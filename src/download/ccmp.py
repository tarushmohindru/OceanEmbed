"""Download CCMP winds over HTTP (RSS), cropping each global daily file immediately."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import write_meta, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.download.common import (
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
EXTRA_VARS = ("nobs",)
TIMES_PER_DAY = 4


class NotAvailableError(Exception):
    """The server has no file for this date (HTTP 404); never retried."""


def url_for(cfg: Config, stream: dict, date: pd.Timestamp) -> str:
    """Fill ``url_template`` with zero-padded ``yyyy``, ``mm``, ``dd``."""
    return stream["url_template"].format(
        yyyy=f"{date.year:04d}", mm=f"{date.month:02d}", dd=f"{date.day:02d}"
    )


@retry(max_tries=5, base_delay_s=15.0, exceptions=(requests.RequestException, OSError))
def download_day(url: str, tmp: Path) -> Path:
    """Stream ``url`` to ``tmp`` (via ``.part``). Raises ``NotAvailableError`` on 404."""
    tmp = Path(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    part = tmp.with_name(tmp.name + ".part")
    with requests.get(url, stream=True, timeout=(30, 600)) as r:
        if r.status_code == 404:
            raise NotAvailableError(url)
        r.raise_for_status()
        with open(part, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
    part.replace(tmp)
    return tmp


def crop_day(src: Path, out: Path, variables: list[str], bbox) -> None:
    """Crop one global CCMP file to ``bbox`` keeping the wind variables, all 4 times."""
    with xr.open_dataset(src) as raw:
        ds = crop_to_bbox(raw, bbox)
        keep = [v for v in [*variables, *EXTRA_VARS] if v in ds.data_vars]
        missing = [v for v in variables if v not in keep]
        if missing:
            raise KeyError(f"{src.name}: variables {missing} missing; has {list(ds.data_vars)}")
        ds = ds[keep].transpose("time", "lat", "lon", ...).load()
    write_netcdf(ds, out)


def _fetch_year(cfg: Config, paths: Paths, entry: dict, stream_cfg: dict, workers: int) -> None:
    stream, year, start, end = entry["stream"], entry["year"], entry["start"], entry["end"]
    variables = list(cfg["products"]["ccmp"]["variables"])
    bbox = download_bbox(cfg)
    work = paths.raw_tmp_dir() / "ccmp"
    work.mkdir(parents=True, exist_ok=True)
    dates = list(pd.date_range(start, end, freq="D"))
    missing: list[pd.Timestamp] = []

    def crop_path(d: pd.Timestamp) -> Path:
        return work / f"crop_{stream}_{d:%Y%m%d}.nc"

    todo = [d for d in dates if not crop_path(d).exists()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(todo), workers * 4):  # bounded number of global files on disk
            futs = {
                pool.submit(
                    download_day, url_for(cfg, stream_cfg, d), work / f"global_{d:%Y%m%d}.nc"
                ): d
                for d in todo[i : i + workers * 4]
            }
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    glob = fut.result()
                except NotAvailableError:
                    log.warning("ccmp %s: not available on server", d.date())
                    missing.append(d)
                    continue
                try:
                    crop_day(glob, crop_path(d), variables, bbox)
                finally:
                    glob.unlink(missing_ok=True)
    files = sorted(work.glob(f"crop_{stream}_{year}*.nc"))
    if not files:
        raise RuntimeError(f"ccmp {year}: no days downloaded")
    target = paths.raw_year_file("ccmp", stream, year)
    concat_daily(files, target, "ccmp", stream)
    register_file(
        cfg, paths, target, "ccmp", stream,
        expected_start=start, expected_end=end, times_per_day=TIMES_PER_DAY,
        ident={
            "dataset_id": stream_cfg["url_template"],
            "dataset_version": entry.get("version", "unknown"),
            "variables": variables,
        },
        tool_versions=tool_version("requests"),
        allow_missing_days=True,  # reported by check_coverage, not fatal here
    )  # fmt: skip


def download_product(
    cfg: Config, paths: Paths, years: list[int] | None, force: bool, workers: int = 4
) -> None:
    """Download CCMP for all years/streams it needs."""
    probe = load_probe(paths)
    streams = {s["name"]: s for s in cfg["products"]["ccmp"]["streams"]}
    for entry in plan_year_files(cfg, "ccmp", probe, years):
        stream = entry["stream"]
        if not force and is_done(
            paths, "ccmp", stream, entry["year"], entry["start"], entry["end"]
        ):
            log.info("ccmp/%s %d: already complete, skipping", stream, entry["year"])
            continue
        _fetch_year(cfg, paths, entry, streams[stream], workers)
    write_meta(paths.raw_dir / "ccmp", cfg, {"product": "ccmp"})


def main() -> None:
    parser = base_parser("Download CCMP winds from RSS")
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args, cfg = parse_and_load(parser)
    download_product(cfg, Paths(cfg), args.years, args.force, args.workers)


if __name__ == "__main__":
    main()
