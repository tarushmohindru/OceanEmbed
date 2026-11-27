"""Atomic writes, provenance files and the prediction-store format."""

from __future__ import annotations

import contextlib
import json
import platform
import re
import shutil
import socket
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.common.config import Config
from src.common.gitinfo import git_hash
from src.common.grid import grid_from_config
from src.common.paths import Paths
from src.common.timeutil import split_of

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_ID = re.compile(r"^[a-z0-9_.\-]+$")
_PLAUSIBLE_DEGC = (-5.0, 40.0)


def _remove(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        p.unlink()


@contextlib.contextmanager
def atomic_write_path(final: Path) -> Iterator[Path]:
    """Yield a temporary sibling of ``final``; rename it onto ``final`` on success.

    The temporary is ``<final>.part`` (``<final>.tmp`` for ``.zarr`` directories). On an
    exception it is deleted and ``final`` is left untouched.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + (".tmp" if final.suffix == ".zarr" else ".part"))
    _remove(tmp)
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"writer did not create {tmp}")
        if final.is_dir():
            _remove(final)
        tmp.replace(final)  # atomic overwrite for files
    except BaseException:
        _remove(tmp)
        raise


def write_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write NetCDF4 with zlib level 4 on float variables, atomically.

    Variables with a ``time`` dim are chunked one time step x the full remaining shape; the
    chunking inherited from source files (often tiny) made compressed 4-D writes crawl.
    """
    ds = ds.copy()  # encoding edits below must not leak to the caller
    enc = {}
    for v in ds.data_vars:
        if not np.issubdtype(ds[v].dtype, np.floating):
            continue
        e = {"zlib": True, "complevel": 4}
        if "time" in ds[v].dims and ds[v].ndim >= 2:
            e["chunksizes"] = tuple(1 if d == "time" else ds.sizes[d] for d in ds[v].dims)
        enc[v] = e
        ds[v].encoding.pop("chunksizes", None)
        ds[v].encoding.pop("preferred_chunks", None)
        ds[v].encoding.pop("contiguous", None)
    with atomic_write_path(path) as tmp:
        ds.to_netcdf(tmp, engine="netcdf4", format="NETCDF4", encoding=enc)


def write_zarr(ds: xr.Dataset, path: Path, chunks: dict) -> None:
    """Write ``ds`` chunked as ``chunks`` through ``Dataset.to_zarr``, atomically."""
    with atomic_write_path(path) as tmp:
        ds.chunk(chunks).to_zarr(tmp, mode="w")


def write_json(obj, path: Path) -> None:
    """JSON with indent 2 and sorted keys, atomically."""
    with atomic_write_path(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_meta(directory: Path, cfg: Config, extra: dict | None = None) -> None:
    """Write ``<directory>/_meta.json`` (time, git hash, config hash, host, python, extra)."""
    meta = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_hash": git_hash(_REPO_ROOT),
        "config_hash": cfg.hash(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "extra": extra or {},
    }
    Path(directory).mkdir(parents=True, exist_ok=True)
    write_json(meta, Path(directory) / "_meta.json")


def file_is_complete(path: Path) -> bool:
    """True if ``path`` exists, has no leftover ``.part``/``.tmp`` sibling and opens."""
    path = Path(path)
    if not path.exists():
        return False
    if path.with_name(path.name + ".part").exists() or path.with_name(path.name + ".tmp").exists():
        return False
    try:
        if path.suffix == ".zarr":
            xr.open_zarr(path).close()
        elif path.suffix in {".nc", ".nc4"}:
            xr.open_dataset(path).close()
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - any failure to open means "not complete"
        return False
    return True


def write_prediction_store(
    T_pred: np.ndarray,  # noqa: N803
    dates,
    cfg: Config,
    model_id: str,
    split: str,
    paths: Paths,
    attrs: dict | None = None,
) -> Path:
    """Validate and write ``results/predictions/<model_id>/<split>.zarr``.

    Args:
        T_pred: (len(dates), D, H, W) float32 temperature in degrees C. Cells outside
            ``masks.nc:depth_mask`` are set to NaN; NaN inside the mask is an error.
        dates: the split's dates (gap days excluded), ascending.
        attrs: extra attributes (e.g. ``checkpoint``, ``run_dir``).
    """
    if not _MODEL_ID.match(model_id):
        raise ValueError(f"invalid model_id {model_id!r}; must match {_MODEL_ID.pattern}")
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    idx = pd.DatetimeIndex(dates)
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("dates must be strictly ascending")
    bad = set(split_of(idx, cfg)) - {split}
    if bad:
        raise ValueError(f"dates contain days outside split {split!r}: {sorted(bad)}")
    grid = grid_from_config(cfg)
    depths = np.asarray(cfg["depths_m"], dtype="float64")
    expected = (len(idx), len(depths), grid.H, grid.W)
    if T_pred.shape != expected:
        raise ValueError(f"T_pred shape {T_pred.shape} != expected {expected}")
    if T_pred.dtype != np.float32:
        raise ValueError(f"T_pred must be float32, got {T_pred.dtype}")

    with xr.open_dataset(paths.masks) as m:
        depth_mask = m["depth_mask"].values.astype(bool)
    if depth_mask.shape != expected[1:]:
        raise ValueError(f"depth_mask shape {depth_mask.shape} != {expected[1:]}")
    pred = np.where(depth_mask[None], T_pred, np.float32(np.nan)).astype(np.float32)
    inside = pred[:, depth_mask]
    if np.isnan(inside).any():
        raise ValueError("T_pred has NaN inside depth_mask")
    if not np.isfinite(inside).all():
        raise ValueError("T_pred has infinite values")
    lo, hi = _PLAUSIBLE_DEGC
    if inside.min() < lo or inside.max() > hi:
        raise ValueError(
            f"T_pred outside plausible range [{lo}, {hi}] degC: "
            f"min {inside.min():.2f}, max {inside.max():.2f}"
        )

    meta_attrs = {
        "model_id": model_id,
        "split": split,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_hash": git_hash(_REPO_ROOT),
        "config_hash": cfg.hash(),
        "checkpoint": "",
        "run_dir": "",
    }
    meta_attrs.update({k: str(v) for k, v in (attrs or {}).items()})
    ds = xr.Dataset(
        {"T_pred": (("time", "depth", "lat", "lon"), pred, {"units": "degC"})},
        coords={"time": idx.values, "depth": depths, "lat": grid.lat, "lon": grid.lon},
        attrs=meta_attrs,
    )
    out = paths.prediction_store(model_id, split)
    write_zarr(ds, out, {"time": 1, "depth": len(depths), "lat": grid.H, "lon": grid.W})
    write_meta(out.parent, cfg, {"model_id": model_id, "splits_written": sorted(_written(out))})
    return out


def _written(store: Path) -> list[str]:
    return [s for s in ("val", "test") if (store.parent / f"{s}.zarr").exists()]


def read_prediction_store(paths: Paths, model_id: str, split: str) -> xr.Dataset:
    """Open a prediction store lazily (dask-backed)."""
    p = paths.prediction_store(model_id, split)
    if not p.exists():
        raise FileNotFoundError(f"no prediction store at {p}")
    return xr.open_zarr(p)
