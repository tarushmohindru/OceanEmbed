"""Step 2: raw -> interim channel-group files and interim GLORYS (native depth levels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import file_is_complete, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import CHANNEL_GROUP, Paths
from src.common.timeutil import daily_index
from src.preprocess.readers import read_product_year
from src.preprocess.regrid_core import Regridder, choose_method

log = get_logger(__name__)
GROUPS = ("sst", "sss", "sla", "cur", "wind")
UNITS = {"sst": "degC", "sss": "PSU", "sla": "m", "u_cur": "m s-1", "v_cur": "m s-1",
         "u_wind": "m s-1", "v_wind": "m s-1"}  # fmt: skip


def group_channels(cfg: Config) -> dict[str, list[str]]:
    """{group: [channels]} in config input order."""
    out: dict[str, list[str]] = {}
    for ch in cfg["inputs"]:
        out.setdefault(CHANNEL_GROUP[ch], []).append(ch)
    return out


def group_product(cfg: Config, group: str) -> str:
    products = {cfg["channels"][ch]["product"] for ch in group_channels(cfg)[group]}
    if len(products) != 1:
        raise ValueError(f"group {group} mixes products {products}")
    return products.pop()


def get_regridder(cfg: Config, paths: Paths, product: str, lat, lon) -> Regridder:
    """Build (or load the cached) regridder for a product's native grid."""
    grid = grid_from_config(cfg)
    method = choose_method(lat, lon, grid)
    cache = paths.weights_file(product, method).with_suffix(".npz")
    thres = cfg["preprocess"]["conservative_na_thres"]
    if cache.exists():
        r = Regridder.load(cache)
        if (
            np.array_equal(r.src_lat, np.asarray(lat, "f8"))
            and np.array_equal(r.src_lon, np.asarray(lon, "f8"))
            and np.allclose(r.grid.lat, grid.lat)
            and np.allclose(r.grid.lon, grid.lon)
            and r.na_thres == thres
        ):
            return r
    r = Regridder(lat, lon, grid, thres, method)
    r.save(cache)
    log.info("%s: %s regridder (%d x %d -> %d x %d)", product, method, len(lat), len(lon),
             grid.H, grid.W)  # fmt: skip
    return r


def years_in_period(cfg: Config) -> list[int]:
    return sorted(set(daily_index(cfg).year))


def regrid_inputs_year(cfg: Config, paths: Paths, year: int, force: bool,
                       groups: list[str] | None = None) -> None:  # fmt: skip
    """Regrid every input channel group of one year to the target grid."""
    grid = grid_from_config(cfg)
    for group, channels in group_channels(cfg).items():
        if groups and group not in groups:
            continue
        out = paths.interim_channel_file(group, year)
        if not force and file_is_complete(out):
            log.info("%s %d: exists, skipping", group, year)
            continue
        product = group_product(cfg, group)
        field = read_product_year(cfg, paths, product, year)
        ds = field.data
        r = get_regridder(cfg, paths, product, ds["lat"].values, ds["lon"].values)
        data_vars = {}
        for ch in channels:
            arr = r(ds[ch].values)
            data_vars[ch] = (("time", "lat", "lon"), arr, {"units": UNITS[ch]})
            any_valid = np.isfinite(arr).any(axis=0)
            frac = np.isnan(arr[:, any_valid]).mean() if any_valid.any() else 1.0
            log.info("%s %d: NaN fraction over ever-valid cells %.4f", ch, year, frac)
        streams = sorted({s for s in field.provenance["stream"] if s})
        out_ds = xr.Dataset(
            data_vars,
            coords={"time": ds["time"].values, "lat": grid.lat, "lon": grid.lon},
            attrs={"source_streams": ",".join(streams), "product": product,
                   "regrid_method": r.method},
        )  # fmt: skip
        _assert_interim(out_ds, cfg, year)
        write_netcdf(out_ds, out)
        field.provenance.to_csv(paths.interim_provenance(product, year), index=False)


def regrid_glorys_year(cfg: Config, paths: Paths, year: int, force: bool) -> None:
    """Regrid GLORYS thetao month by month, all native levels within the download range."""
    out = paths.interim_glorys(year)
    if not force and file_is_complete(out):
        log.info("glorys %d: exists, skipping", year)
        return
    grid = grid_from_config(cfg)
    field = read_product_year(cfg, paths, "glorys", year)
    ds = field.data
    r = get_regridder(cfg, paths, "glorys", ds["lat"].values, ds["lon"].values)
    months = []
    times = pd.DatetimeIndex(ds["time"].values)
    for m in range(1, 13):
        sel = times.month == m
        if not sel.any():
            continue
        block = ds["thetao"].isel(time=np.where(sel)[0]).values  # (t, K, Hs, Ws)
        months.append(r(block))
        log.info("glorys %d-%02d regridded", year, m)
    arr = np.concatenate(months, axis=0)
    out_ds = xr.Dataset(
        {"thetao": (("time", "depth", "lat", "lon"), arr, {"units": "degC",
                                                          "long_name": "potential temperature"})},
        coords={"time": times.values, "depth": ds["depth"].values.astype("f8"),
                "lat": grid.lat, "lon": grid.lon},
        attrs={"source_streams": ",".join(sorted({s for s in field.provenance["stream"] if s})),
               "regrid_method": r.method},
    )  # fmt: skip
    _assert_interim(out_ds, cfg, year)
    write_netcdf(out_ds, out)
    field.provenance.to_csv(paths.interim_provenance("glorys", year), index=False)


def _assert_interim(ds: xr.Dataset, cfg: Config, year: int) -> None:
    grid = grid_from_config(cfg)
    days = daily_index(cfg)
    n_days = int((days.year == year).sum())
    assert np.allclose(ds["lat"], grid.lat, atol=1e-6) and np.allclose(
        ds["lon"], grid.lon, atol=1e-6
    )
    assert ds.sizes["time"] == n_days, (ds.sizes["time"], n_days)
    for v in ds.data_vars:
        assert ds[v].dtype == np.float32, (v, ds[v].dtype)


def main() -> None:
    parser = base_parser("Step 2: regrid raw products to the target grid")
    parser.add_argument("--products", nargs="*", default=None,
                        help="groups (sst sss sla cur wind) and/or glorys")  # fmt: skip
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    run_regrid(cfg, paths, args.years, args.products, args.force)


def run_regrid(cfg: Config, paths: Paths, years=None, products=None, force=False) -> None:
    years = years or years_in_period(cfg)
    groups = [p for p in (products or GROUPS) if p in GROUPS]
    for y in years:
        if groups:
            regrid_inputs_year(cfg, paths, y, force, groups)
        if not products or "glorys" in products:
            regrid_glorys_year(cfg, paths, y, force)


if __name__ == "__main__":
    main()
