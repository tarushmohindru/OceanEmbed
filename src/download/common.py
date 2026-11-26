"""Shared download helpers: retries, locking, manifest, cropping, concatenation, credentials."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import xarray as xr

from src.common.config import Config
from src.common.grid import standardize_coords
from src.common.io import atomic_write_path, write_json, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import floor_to_date

log = get_logger(__name__)

PROBE_HINT = "run `python -m src.download.probe` first (results/probe/probe.json is missing)"


def retry(
    max_tries: int = 6,
    base_delay_s: float = 10.0,
    max_delay_s: float = 600.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator: retry with exponential backoff and jitter; re-raise after ``max_tries``."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_tries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_tries:
                        log.error("%s failed after %d attempts: %s", fn.__name__, attempt, e)
                        raise
                    delay = min(max_delay_s, base_delay_s * 2 ** (attempt - 1))
                    delay *= 0.5 + random.random()  # noqa: S311 - jitter, not security
                    log.warning(
                        "%s failed (attempt %d/%d): %s; retrying in %.0f s",
                        fn.__name__, attempt, max_tries, e, delay,
                    )  # fmt: skip
                    time.sleep(delay)

        return wrapper

    return deco


@contextlib.contextmanager
def file_lock(path: Path, timeout_s: float = 600.0, stale_s: float = 3600.0) -> Iterator[None]:
    """Cross-process lock held through ``<path>.lock`` (``O_CREAT|O_EXCL``)."""
    lock = Path(path).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not acquire {lock} within {timeout_s:.0f} s") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def read_manifest(paths: Paths) -> dict:
    """Return the manifest dict (empty if it does not exist yet)."""
    p = paths.manifest()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def update_manifest(paths: Paths, rel_path: str, record: dict) -> None:
    """Insert or replace ``record`` under ``rel_path`` (posix, relative to ``raw/``)."""
    with file_lock(paths.manifest()):
        manifest = read_manifest(paths)
        manifest[rel_path] = record
        write_json(manifest, paths.manifest())


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_bbox(cfg: Config) -> tuple[float, float, float, float]:
    """``(lon_min, lat_min, lon_max, lat_max)`` of the region plus ``download_buffer_deg``."""
    r, buf = cfg["region"], float(cfg["download_buffer_deg"])
    return (r["lon_min"] - buf, r["lat_min"] - buf, r["lon_max"] + buf, r["lat_max"] + buf)


def crop_to_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    """Standardize coordinates then select ``bbox = (lon0, lat0, lon1, lat1)`` inclusively."""
    ds = standardize_coords(ds)
    lon0, lat0, lon1, lat1 = bbox
    out = ds.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    if out.sizes.get("lat", 0) == 0 or out.sizes.get("lon", 0) == 0:
        raise ValueError(f"cropping to {bbox} left an empty dataset")
    return out


def concat_daily(files: list[Path], out: Path, product: str, stream: str) -> None:
    """Concatenate cropped files on ``time`` (later files win duplicates), write, delete inputs."""
    if not files:
        raise ValueError("concat_daily: no input files")
    parts = []
    for f in files:
        with xr.open_dataset(f) as d:
            parts.append(d.load())
    ds = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override")
    ds = ds.sortby("time")  # stable, so among equal times the later input stays last
    keep = ~ds.indexes["time"].duplicated(keep="last")
    ds = ds.isel(time=keep)
    ds.attrs.update({"product": product, "stream": stream})
    write_netcdf(ds, out)
    for f in files:
        Path(f).unlink()


def years_needed(start, end) -> list[int]:
    return list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))


def _stream_names(cfg: Config, product: str) -> list[str]:
    return [s["name"] for s in cfg["products"][product]["streams"]]


def load_probe(paths: Paths) -> dict:
    """Read ``results/probe/probe.json`` or raise with the standard hint."""
    p = paths.probe_dir / "probe.json"
    if not p.exists():
        raise SystemExit(PROBE_HINT)
    return json.loads(p.read_text(encoding="utf-8"))


def stream_date_ranges(
    cfg: Config, product: str, probe: dict
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Which stream serves which dates: first stream (config order) wins, no overlap.

    Reads ``probe["products"][product]["streams"][name]["time_start"/"time_end"]``, clips to
    ``dates``, and returns ``(stream, start, end)`` runs sorted by start. Dates covered by no
    stream are absent. A stream can appear more than once if an earlier stream sits inside it.
    """
    days = pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"], freq="D")
    owner = pd.Series([None] * len(days), index=days, dtype=object)
    info = probe["products"][product]["streams"]
    for name in _stream_names(cfg, product):
        rec = info.get(name) or {}
        if not rec.get("time_start") or not rec.get("time_end"):
            continue
        a, b = pd.Timestamp(rec["time_start"]), pd.Timestamp(rec["time_end"])
        free = owner.isna() & (owner.index >= a) & (owner.index <= b)
        owner[free] = name
    runs: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for day, name in owner.items():
        if name is None:
            continue
        if runs and runs[-1][0] == name and runs[-1][2] == day - pd.Timedelta(days=1):
            runs[-1] = (name, runs[-1][1], day)
        else:
            runs.append((name, day, day))
    return runs


def plan_year_files(
    cfg: Config, product: str, probe: dict, years: list[int] | None = None
) -> list[dict]:
    """One entry per (stream, year): ``{"stream", "year", "start", "end"}`` (Timestamps).

    Several runs of the same stream inside a year are merged to their min start / max end.
    """
    merged: dict[tuple[str, int], list[pd.Timestamp]] = {}
    for stream, a, b in stream_date_ranges(cfg, product, probe):
        for y in years_needed(a, b):
            if years is not None and y not in years:
                continue
            lo = max(a, pd.Timestamp(year=y, month=1, day=1))
            hi = min(b, pd.Timestamp(year=y, month=12, day=31))
            cur = merged.setdefault((stream, y), [lo, hi])
            cur[0], cur[1] = min(cur[0], lo), max(cur[1], hi)
    return [
        {"stream": s, "year": y, "start": v[0], "end": v[1]}
        for (s, y), v in sorted(merged.items(), key=lambda kv: (kv[0][1], kv[1][0]))
    ]


def times_in_file(path: Path) -> pd.DatetimeIndex:
    with xr.open_dataset(path) as ds:
        return pd.DatetimeIndex(ds["time"].values)


def is_done(paths: Paths, product: str, stream: str, year: int, start, end) -> bool:
    """True if the year file is complete and the manifest records this requested range."""
    from src.common.io import file_is_complete

    f = paths.raw_year_file(product, stream, year)
    rec = read_manifest(paths).get(f.relative_to(paths.raw_dir).as_posix())
    return bool(
        rec
        and rec.get("requested_start") == pd.Timestamp(start).strftime("%Y-%m-%d")
        and rec.get("requested_end") == pd.Timestamp(end).strftime("%Y-%m-%d")
        and file_is_complete(f)
    )


def register_file(
    cfg: Config,
    paths: Paths,
    path: Path,
    product: str,
    stream: str,
    *,
    expected_start,
    expected_end,
    times_per_day: int = 1,
    ident: dict | None = None,
    tool_versions: dict | None = None,
    allow_missing_days: bool = False,
) -> dict:
    """Verify the year file against its expected dates and write its manifest record.

    Raises ``RuntimeError`` if days are missing (unless ``allow_missing_days``) or the file
    extends outside the expected range.
    """
    times = times_in_file(path)
    days = floor_to_date(times)
    expected = pd.date_range(expected_start, expected_end, freq="D")
    missing = expected.difference(days.unique())
    extra = days.unique().difference(expected)
    if len(extra):
        raise RuntimeError(f"{path.name}: {len(extra)} unexpected dates, first {extra[0].date()}")
    if len(missing) and not allow_missing_days:
        raise RuntimeError(f"{path.name}: {len(missing)} missing days, first {missing[0].date()}")
    if len(missing):
        log.warning("%s: %d missing days (first %s)", path.name, len(missing), missing[0].date())
    if not allow_missing_days and times_per_day > 1:
        counts = pd.Series(days).value_counts()
        if (counts < times_per_day).any():
            raise RuntimeError(f"{path.name}: some days have fewer than {times_per_day} times")
    lon0, lat0, lon1, lat1 = download_bbox(cfg)
    record = {
        **(ident or {}),
        "product": product,
        "stream": stream,
        "time_start": days.min().strftime("%Y-%m-%d"),
        "time_end": days.max().strftime("%Y-%m-%d"),
        "requested_start": pd.Timestamp(expected_start).strftime("%Y-%m-%d"),
        "requested_end": pd.Timestamp(expected_end).strftime("%Y-%m-%d"),
        "n_times": int(len(times)),
        "n_missing_days": int(len(missing)),
        "bbox": [lon0, lat0, lon1, lat1],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "downloaded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool_versions": tool_versions or {},
    }
    update_manifest(paths, path.relative_to(paths.raw_dir).as_posix(), record)
    return record


def check_credentials(product: str) -> None:
    """Raise ``RuntimeError`` with a setup hint if ``product`` needs credentials that are absent."""
    home = Path.home()
    if product in ("glorys", "ostia", "duacs"):
        env = os.environ
        has_env = env.get("COPERNICUSMARINE_SERVICE_USERNAME") and env.get(
            "COPERNICUSMARINE_SERVICE_PASSWORD"
        )
        cred_dir = Path(
            env.get("COPERNICUSMARINE_CREDENTIALS_DIRECTORY", home / ".copernicusmarine")
        )
        if not (has_env or (cred_dir / ".copernicusmarine-credentials").exists()):
            raise RuntimeError(
                f"{product}: no Copernicus Marine credentials. Run `copernicusmarine login`, or set "  # noqa: E501
                "COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD."
            )
    elif product in ("oscar", "smap"):
        env = os.environ
        has_env = (env.get("EARTHDATA_USERNAME") and env.get("EARTHDATA_PASSWORD")) or env.get(
            "EARTHDATA_TOKEN"
        )
        has_netrc = (home / ".netrc").exists() or (home / "_netrc").exists()
        if not (has_env or has_netrc):
            raise RuntimeError(
                f"{product}: no NASA Earthdata credentials. Set EARTHDATA_USERNAME and "
                "EARTHDATA_PASSWORD (Lightning secrets) or create ~/.netrc."
            )
    # ccmp and argo need no credentials


def tool_version(package: str) -> dict:
    """``{package: version}`` or ``{package: "unknown"}``."""
    from importlib import metadata

    try:
        return {package: metadata.version(package)}
    except metadata.PackageNotFoundError:
        return {package: "unknown"}


__all__ = [
    "atomic_write_path",
    "check_credentials",
    "concat_daily",
    "crop_to_bbox",
    "download_bbox",
    "file_lock",
    "load_probe",
    "plan_year_files",
    "read_manifest",
    "register_file",
    "retry",
    "sha256_file",
    "stream_date_ranges",
    "update_manifest",
    "years_needed",
]
