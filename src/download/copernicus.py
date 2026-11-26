"""Download GLORYS, OSTIA and DUACS from Copernicus Marine (``copernicusmarine`` toolbox)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import file_is_complete, write_meta
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.download.common import (
    check_credentials,
    concat_daily,
    download_bbox,
    is_done,
    load_probe,
    plan_year_files,
    register_file,
    retry,
    tool_version,
    verify_copernicus_login,
)

log = get_logger(__name__)
PRODUCTS = ("glorys", "ostia", "duacs")


@retry(max_tries=6, base_delay_s=15.0)
def cm_subset(
    dataset_id: str,
    variables: list[str],
    bbox: tuple[float, float, float, float],
    t0: str,
    t1: str,
    out_file: Path,
    depth_range: tuple[float, float] | None = None,
) -> Path:
    """Subset a Copernicus Marine dataset to ``out_file`` (all toolbox-specific args live here).

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        t0, t1: inclusive dates ``YYYY-MM-DD``; the whole of day ``t1`` is requested.
    """
    import copernicusmarine

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lon0, lat0, lon1, lat1 = bbox
    kwargs = {}
    if depth_range is not None:
        kwargs.update(minimum_depth=depth_range[0], maximum_depth=depth_range[1])
    # The toolbox picks its own file name logic; write into a private directory then move.
    with tempfile.TemporaryDirectory(dir=out_file.parent, prefix=".cm_") as tmpdir:
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=list(variables),
            minimum_longitude=lon0,
            maximum_longitude=lon1,
            minimum_latitude=lat0,
            maximum_latitude=lat1,
            start_datetime=f"{t0}T00:00:00",
            end_datetime=f"{t1}T23:59:59",
            output_directory=tmpdir,
            output_filename="data.nc",
            file_format="netcdf",
            netcdf_compression_level=1,
            overwrite=True,
            disable_progress_bar=True,
            **kwargs,
        )
        produced = Path(tmpdir) / "data.nc"
        if not produced.exists():
            raise RuntimeError(f"copernicusmarine produced no file for {dataset_id} {t0}..{t1}")
        part = out_file.with_name(out_file.name + ".part")
        shutil.move(produced, part)
    part.replace(out_file)
    return out_file


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    cur = start.replace(day=1)
    while cur <= end:
        lo = max(cur, start)
        hi = min(cur + pd.offsets.MonthEnd(0), end)
        out.append((lo, hi))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def _fetch_year(cfg: Config, paths: Paths, product: str, entry: dict, stream_cfg: dict) -> None:
    pcfg = cfg["products"][product]
    stream, year, start, end = entry["stream"], entry["year"], entry["start"], entry["end"]
    target = paths.raw_year_file(product, stream, year)
    bbox = download_bbox(cfg)
    depth = tuple(pcfg["depth_range_m"]) if "depth_range_m" in pcfg else None
    tmp = paths.raw_tmp_dir()
    # GLORYS: monthly requests (cheap retries); others: one request per year.
    chunks = _months(start, end) if product == "glorys" else [(start, end)]
    parts: list[Path] = []
    for lo, hi in chunks:
        part = tmp / f"{product}_{stream}_{year}_{lo:%m}.nc"
        if not (part.exists() and file_is_complete(part)):
            log.info("%s/%s: requesting %s .. %s", product, stream, lo.date(), hi.date())
            cm_subset(
                stream_cfg["dataset_id"], pcfg["variables"], bbox,
                f"{lo:%Y-%m-%d}", f"{hi:%Y-%m-%d}", part, depth,
            )  # fmt: skip
        parts.append(part)
    if len(parts) == 1 and product != "glorys":
        part = target.with_name(target.name + ".part")
        shutil.move(parts[0], part)
        part.replace(target)
    else:
        concat_daily(parts, target, product, stream)
    register_file(
        cfg, paths, target, product, stream,
        expected_start=start, expected_end=end,
        ident={
            "dataset_id": stream_cfg["dataset_id"],
            "dataset_version": entry.get("version", "unknown"),
            "variables": list(pcfg["variables"]),
        },
        tool_versions=tool_version("copernicusmarine"),
    )  # fmt: skip
    log.info("%s/%s %d done: %s", product, stream, year, target.name)


def download_product(
    cfg: Config, paths: Paths, product: str, years: list[int] | None, force: bool
) -> None:
    """Download one Copernicus product for all streams/years it needs."""
    if product not in PRODUCTS:
        raise ValueError(f"{product} is not a Copernicus Marine product")
    check_credentials(product)
    verify_copernicus_login()
    probe = load_probe(paths)
    streams = {s["name"]: s for s in cfg["products"][product]["streams"]}
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
        _fetch_year(cfg, paths, product, entry, streams[stream])
    write_meta(paths.raw_dir / product, cfg, {"product": product})


def main() -> None:
    parser = base_parser("Download GLORYS / OSTIA / DUACS from Copernicus Marine")
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    for product in args.products:
        download_product(cfg, paths, product, args.years, args.force)


if __name__ == "__main__":
    main()
