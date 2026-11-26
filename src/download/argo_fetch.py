"""Fetch ARGO profiles for the test years with ``argopy`` (expert mode, no QC filtering)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import file_is_complete, write_meta, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.download.common import retry, tool_version, update_manifest

log = get_logger(__name__)

# Variables to keep. argopy names differ between versions/sources: edit after the probe.
KEEP_VARS = [
    "PLATFORM_NUMBER", "CYCLE_NUMBER", "DIRECTION", "TIME", "LATITUDE", "LONGITUDE",
    "PRES", "TEMP", "PSAL", "PRES_ADJUSTED", "TEMP_ADJUSTED", "PSAL_ADJUSTED",
    "PRES_QC", "TEMP_QC", "PSAL_QC", "PRES_ADJUSTED_QC", "TEMP_ADJUSTED_QC", "PSAL_ADJUSTED_QC",
    "DATA_MODE", "POSITION_QC", "TIME_QC",
]  # fmt: skip


def raw_argo_file(paths: Paths, year: int) -> Path:
    d = paths.raw_dir / "argo"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"argo_{year}.nc"


@retry(max_tries=3, base_delay_s=30.0)
def _fetch_month(box: list, src: str) -> xr.Dataset:
    from argopy import DataFetcher

    return DataFetcher(src=src, mode="expert").region(box).to_xarray()


def _month_with_fallback(box: list, src: str) -> tuple[xr.Dataset, str]:
    try:
        return _fetch_month(box, src), src
    except Exception as e:  # noqa: BLE001 - any server failure triggers the fallback
        if src == "gdac":
            raise
        log.warning("argopy src=%s failed (%s); switching to gdac", src, e)
        return _fetch_month(box, "gdac"), "gdac"


def fetch_year(cfg: Config, paths: Paths, year: int, force: bool) -> Path:
    """Fetch one year month by month and write ``raw/argo/argo_<year>.nc``."""
    target = raw_argo_file(paths, year)
    if not force and file_is_complete(target):
        log.info("argo %d: already complete", year)
        return target
    a = cfg["products"]["argo"]
    r = cfg["region"]
    src = a["argopy_src"]
    tmp = paths.raw_tmp_dir() / "argo"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    for month in range(1, 13):
        lo = pd.Timestamp(year=year, month=month, day=1)
        hi = lo + pd.offsets.MonthBegin(1)
        cache = tmp / f"argo_{year}_{month:02d}.nc"
        if cache.exists() and file_is_complete(cache):
            parts.append(xr.open_dataset(cache).load())
            continue
        box = [
            r["lon_min"], r["lon_max"], r["lat_min"], r["lat_max"],
            0.0, float(a["max_depth_m"]), f"{lo:%Y-%m-%d}", f"{hi:%Y-%m-%d}",
        ]  # fmt: skip
        ds, src = _month_with_fallback(box, src)
        keep = [v for v in KEEP_VARS if v in ds.variables]
        ds = ds[keep].load()
        write_netcdf(ds, cache)
        parts.append(ds)
        log.info("argo %d-%02d: %d points", year, month, ds.sizes.get("N_POINTS", 0))
    parts = [p for p in parts if p.sizes.get("N_POINTS", 0) > 0]
    if not parts:
        raise RuntimeError(f"argo {year}: no data returned")
    out = xr.concat(parts, dim="N_POINTS")
    out.attrs.update({"product": "argo", "argopy_src": src, "year": year})
    write_netcdf(out, target)
    update_manifest(
        paths,
        target.relative_to(paths.raw_dir).as_posix(),
        {
            "product": "argo", "stream": src, "n_points": int(out.sizes["N_POINTS"]),
            "variables": sorted(out.variables), "time_start": f"{year}-01-01",
            "time_end": f"{year}-12-31", "bytes": target.stat().st_size,
            "tool_versions": tool_version("argopy"),
        },
    )  # fmt: skip
    for f in tmp.glob(f"argo_{year}_*.nc"):
        f.unlink()
    return target


def main() -> None:
    parser = base_parser("Fetch ARGO profiles with argopy")
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    for year in args.years or cfg["products"]["argo"]["fetch_years"]:
        fetch_year(cfg, paths, year, args.force)
    write_meta(paths.raw_dir / "argo", cfg, {"product": "argo"})


if __name__ == "__main__":
    main()
