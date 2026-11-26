"""Per-product readers: raw year files -> daily, standardized, unit-converted, QC'd datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.common.config import Config
from src.common.grid import standardize_coords
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import daily_index, floor_to_date
from src.preprocess import qc

log = get_logger(__name__)

# Accepted `units` attributes after reading (lower-case). Unknown units are a warning, because
# providers spell them inconsistently; Kelvin SST is converted.
EXPECTED_UNITS = {
    "sst": {"degc", "celsius", "degree_celsius", "degrees_celsius", "°c", "c"},
    "sss": {"psu", "1e-3", "1", "pss-78", "practical_salinity_unit", "pss", "0.001", ""},
    "sla": {"m", "meter", "meters", "metre", "metres"},
    "u_cur": {"m/s", "m s-1", "m.s-1", "meter second-1", "meters/second", "m s**-1"},
    "v_cur": {"m/s", "m s-1", "m.s-1", "meter second-1", "meters/second", "m s**-1"},
    "u_wind": {"m/s", "m s-1", "m.s-1", "meter second-1", "meters/second", "m s**-1"},
    "v_wind": {"m/s", "m s-1", "m.s-1", "meter second-1", "meters/second", "m s**-1"},
    "thetao": {"degrees_c", "degc", "celsius", "°c", "degree_celsius"},
}
KELVIN = {"k", "kelvin", "degrees_k", "degk"}
TIMES_PER_DAY = {"ccmp": 4}


@dataclass
class DailyField:
    """Daily product data on its native grid plus the stream used per date."""

    data: xr.Dataset  # (time, lat, lon) or (time, depth, lat, lon)
    provenance: pd.DataFrame  # date, product, stream, dataset_id


def product_variables(cfg: Config, product: str) -> dict[str, str]:
    """{raw variable name: output name} for ``product`` (channel names; GLORYS keeps thetao)."""
    if product == "glorys":
        return {v: v for v in cfg["products"]["glorys"]["variables"]}
    return {
        cfg["channels"][ch]["variable"]: ch
        for ch in cfg["inputs"]
        if cfg["channels"][ch]["product"] == product
    }


def stream_files(cfg: Config, paths: Paths, product: str, year: int) -> list[tuple[str, str, Path]]:
    """Existing raw year files ``(stream, dataset_id, path)`` in config stream order."""
    out = []
    for s in cfg["products"][product]["streams"]:
        f = paths.raw_dir / product / s["name"] / f"{product}_{s['name']}_{year}.nc"
        if f.exists():
            ident = s.get("dataset_id") or s.get("short_name") or s.get("url_template", "")
            out.append((s["name"], ident, f))
    return out


def _daily(ds: xr.Dataset, product: str) -> xr.Dataset:
    days = floor_to_date(ds["time"].values)
    if product in TIMES_PER_DAY:
        need = TIMES_PER_DAY[product]
        ds = ds.assign_coords(time=days)
        count = ds.notnull().groupby("time").sum()
        mean = ds.groupby("time").mean()
        short = (count < need).any(dim=[d for d in count.dims if d != "time"])
        n_short = int(short.to_array().any("variable").sum())
        if n_short:
            log.warning("%s: %d dates with fewer than %d times -> NaN", product, n_short, need)
        return mean.where(count >= need)
    if not days.is_unique:
        raise ValueError(f"{product}: more than one time step per date")
    return ds.assign_coords(time=days)


def _convert_units(ds: xr.Dataset) -> xr.Dataset:
    for name in ds.data_vars:
        units = str(ds[name].attrs.get("units", "")).strip()
        low = units.lower()
        if name in ("sst", "thetao") and low in KELVIN:
            attrs = dict(ds[name].attrs, units="degC")
            ds[name] = (ds[name] - 273.15).assign_attrs(attrs)
            continue
        if name == "sst" and not units:
            mean = float(ds[name].mean(skipna=True))
            if mean > 200:  # no units attribute but clearly Kelvin
                ds[name] = (ds[name] - 273.15).assign_attrs(units="degC")
                continue
        if name in EXPECTED_UNITS and low not in EXPECTED_UNITS[name]:
            log.warning("%s: unexpected units %r (no conversion applied)", name, units)
    return ds


def read_product_year(cfg: Config, paths: Paths, product: str, year: int) -> DailyField:
    """Read one product for one year following the rules in src/preprocess/IMPLEMENTATION.md."""
    days = daily_index(cfg)
    days = days[days.year == year]
    rename = product_variables(cfg, product)
    files = stream_files(cfg, paths, product, year)
    if not files:
        raise FileNotFoundError(f"no raw files for {product} {year}")
    pieces = []
    for stream, ident, f in files:
        ds = xr.open_dataset(f, chunks={"time": 31} if product == "glorys" else None)
        ds = standardize_coords(ds)
        missing = [v for v in rename if v not in ds.data_vars]
        if missing:
            raise KeyError(f"{f.name}: variables {missing} not found; has {list(ds.data_vars)}")
        ds = _daily(ds[list(rename)], product)
        ds = ds.sel(time=ds["time"].isin(days.values))
        first = next(iter(rename))
        if product == "glorys":
            avail = pd.DatetimeIndex(ds["time"].values)
        else:
            finite = ds[first].notnull().any(dim=["lat", "lon"]).values
            avail = pd.DatetimeIndex(ds["time"].values[finite])
        pieces.append((stream, ident, ds, avail))

    owner = pd.Series([None] * len(days), index=days, dtype=object)
    for stream, _, _, avail in pieces:
        free = owner.isna() & owner.index.isin(avail)
        owner[free] = stream
    parts = []
    ref = pieces[0][2]
    for stream, _, ds, _ in pieces:
        mine = owner.index[owner == stream]
        if len(mine) == 0:
            continue
        if not (np.array_equal(ds["lat"], ref["lat"]) and np.array_equal(ds["lon"], ref["lon"])):
            raise ValueError(f"{product} {year}: stream {stream} is on a different grid")
        parts.append(ds.sel(time=mine.values))
    if not parts:
        raise ValueError(f"{product} {year}: no stream has any data in the period")
    out = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    out = out.sortby("time").reindex(time=days.values)
    out = out.rename(rename)
    out = _convert_units(out)
    if product != "glorys":
        out, rejected = qc.apply_ranges(out, cfg)
        if any(rejected.values()):
            log.info("%s %d QC rejected: %s", product, year, rejected)
    dims = ("time", "depth", "lat", "lon") if product == "glorys" else ("time", "lat", "lon")
    out = out.transpose(*dims).astype("float32")
    idents = {s: i for s, i, _, _ in pieces}
    prov = pd.DataFrame(
        {
            "date": owner.index.strftime("%Y-%m-%d"),
            "product": product,
            "stream": owner.values,
            "dataset_id": [idents.get(s, "") for s in owner.values],
        }
    )
    n_missing = int(owner.isna().sum())
    if n_missing:
        log.warning("%s %d: %d dates have no data in any stream (NaN)", product, year, n_missing)
    return DailyField(data=out, provenance=prov)
