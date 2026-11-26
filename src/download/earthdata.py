"""Download OSCAR currents and SMAP salinity via NASA Earthdata (``earthaccess``).

Global daily granules are cropped to the region+buffer immediately and deleted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import write_meta, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import floor_to_date
from src.download.common import (
    check_credentials,
    concat_daily,
    crop_to_bbox,
    download_bbox,
    is_done,
    load_probe,
    plan_year_files,
    register_file,
    retry,
    tool_version,
)

log = get_logger(__name__)
PRODUCTS = ("oscar", "smap")
BATCH = 30


def login_earthdata() -> None:
    """Log in with environment credentials, falling back to ``~/.netrc``."""
    import earthaccess

    auth = earthaccess.login(strategy="environment")
    if not getattr(auth, "authenticated", False):
        auth = earthaccess.login(strategy="netrc")
    if not getattr(auth, "authenticated", False):
        raise RuntimeError("Earthdata login failed (check EARTHDATA_USERNAME/PASSWORD or ~/.netrc)")


def crop_granule(
    src: Path, out: Path, variables: list[str], bbox, t0: pd.Timestamp, t1: pd.Timestamp
) -> bool:
    """Crop one global granule to ``bbox``/``variables`` and the dates ``[t0, t1]``.

    Variables are transposed to ``(time, lat, lon)`` by name. Returns False if no time of the
    granule falls inside the range (nothing written).
    """
    with xr.open_dataset(src) as raw:
        ds = crop_to_bbox(raw, bbox)
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise KeyError(f"{src.name}: variables {missing} not found; has {list(ds.data_vars)}")
        ds = ds[variables]
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")
        ds = ds.transpose("time", "lat", "lon", ...)
        days = floor_to_date(ds["time"].values)
        keep = (days >= t0) & (days <= t1)
        if not keep.any():
            return False
        ds = ds.isel(time=keep).load()
        ds.attrs["source_granule"] = src.name
    write_netcdf(ds, out)
    return True


def _order_key(path: Path) -> tuple[str, str]:
    """Order cropped files so that, for duplicate dates, the latest-created one comes last."""
    with xr.open_dataset(path) as d:
        created = str(d.attrs.get("date_created", d.attrs.get("history", "")))
    return (created, path.name)


@retry(max_tries=5, base_delay_s=20.0)
def _search(short_name: str, t0: str, t1: str, bbox) -> list:
    import earthaccess

    return earthaccess.search_data(short_name=short_name, temporal=(t0, t1), bounding_box=bbox)


@retry(max_tries=5, base_delay_s=20.0)
def _download(granules: list, local: Path) -> list[Path]:
    import earthaccess

    return [Path(p) for p in earthaccess.download(granules, local_path=str(local))]


def _fetch_year(cfg: Config, paths: Paths, product: str, entry: dict, stream_cfg: dict) -> None:
    stream, year, start, end = entry["stream"], entry["year"], entry["start"], entry["end"]
    variables = list(cfg["products"][product]["variables"])
    bbox = download_bbox(cfg)
    work = paths.raw_tmp_dir() / product
    dl = work / "dl"
    crops = work / f"crop_{stream}_{year}"
    dl.mkdir(parents=True, exist_ok=True)
    crops.mkdir(parents=True, exist_ok=True)

    granules = _search(stream_cfg["short_name"], f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", bbox)
    log.info("%s/%s %d: %d granules", product, stream, year, len(granules))
    for i in range(0, len(granules), BATCH):
        batch = granules[i : i + BATCH]
        for f in _download(batch, dl):
            out = crops / f"crop_{f.stem}.nc"
            try:
                if not out.exists():
                    crop_granule(f, out, variables, bbox, start, end)
            finally:
                f.unlink(missing_ok=True)  # never keep global files
    files = sorted(crops.glob("crop_*.nc"), key=_order_key)
    if not files:
        raise RuntimeError(f"{product}/{stream} {year}: no granules inside {start}..{end}")
    target = paths.raw_year_file(product, stream, year)
    concat_daily(files, target, product, stream)
    register_file(
        cfg, paths, target, product, stream,
        expected_start=start, expected_end=end,
        ident={
            "dataset_id": stream_cfg["short_name"],
            "dataset_version": entry.get("version", "unknown"),
            "variables": variables,
        },
        tool_versions=tool_version("earthaccess"),
        allow_missing_days=product == "smap",  # SMAP gaps are filled in preprocessing
    )  # fmt: skip
    crops.rmdir()


def download_product(
    cfg: Config, paths: Paths, product: str, years: list[int] | None, force: bool
) -> None:
    """Download one Earthdata product for all streams/years it needs."""
    if product not in PRODUCTS:
        raise ValueError(f"{product} is not an Earthdata product")
    check_credentials(product)
    probe = load_probe(paths)
    streams = {s["name"]: s for s in cfg["products"][product]["streams"]}
    logged_in = False
    for entry in plan_year_files(cfg, product, probe, years):
        stream = entry["stream"]
        entry["version"] = (
            probe["products"][product]["streams"].get(stream, {}).get("version") or "unknown"
        )
        if not force and is_done(
            paths, product, stream, entry["year"], entry["start"], entry["end"]
        ):
            log.info("%s/%s %d: already complete, skipping", product, stream, entry["year"])
            continue
        if not logged_in:
            login_earthdata()
            logged_in = True
        _fetch_year(cfg, paths, product, entry, streams[stream])
    write_meta(paths.raw_dir / product, cfg, {"product": product})


def main() -> None:
    parser = base_parser("Download OSCAR / SMAP via NASA Earthdata")
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    for product in args.products:
        download_product(cfg, paths, product, args.years, args.force)


if __name__ == "__main__":
    main()
