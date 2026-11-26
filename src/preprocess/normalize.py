"""Step 7: normalization statistics from training days only (norm_stats.json)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import file_is_complete, write_json
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import split_of, years_of_split

log = get_logger(__name__)


def masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    """Mean/std (float64) of ``values`` (T, H, W) over cells where ``mask`` (H, W), finite only."""
    v = values[:, mask].astype("f8")
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no values to compute statistics from")
    return float(v.mean()), float(v.std()), int(v.size)


def compute_stats(
    x_raw: np.ndarray,
    x_dates: pd.DatetimeIndex,
    y_degc: np.ndarray,
    y_dates: pd.DatetimeIndex,
    ocean: np.ndarray,
    depth_mask: np.ndarray,
    cfg: Config,
) -> dict:
    """Statistics per the norm_stats.json schema; uses training days only."""
    xt = split_of(x_dates, cfg) == "train"
    yt = split_of(y_dates, cfg) == "train"
    inputs = {}
    for c, ch in enumerate(cfg["inputs"]):
        m, s, _ = masked_mean_std(x_raw[xt, c], ocean)
        assert s > 0, f"zero std for {ch}"
        inputs[ch] = {"mean": m, "std": s}
    targets, counts = {}, {}
    for d, z in enumerate(cfg["depths_m"]):
        m, s, n = masked_mean_std(y_degc[yt, d], depth_mask[d])
        assert s > 0, f"zero std at depth {z}"
        targets[str(int(z))] = {"mean": m, "std": s}
        counts[str(int(z))] = int(depth_mask[d].sum())
    return {
        "years_used": years_of_split(cfg, "train"),
        "inputs": inputs,
        "targets": targets,
        "n_cells_per_depth": counts,
        "config_hash": cfg.hash(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _load_years(paths_fn, var: str, years: list[int]) -> tuple[np.ndarray, pd.DatetimeIndex]:
    arrs, times = [], []
    for y in years:
        with xr.open_dataset(paths_fn(y)) as ds:
            arrs.append(ds[var].values)
            times.append(pd.DatetimeIndex(ds["time"].values))
    return np.concatenate(arrs, 0), (times[0].append(times[1:]) if len(times) > 1 else times[0])


def run_normalize(cfg: Config, paths: Paths, force: bool = False) -> None:
    from src.preprocess.masks import load_masks

    if not force and file_is_complete(paths.norm_stats):
        log.info("norm_stats.json exists, skipping")
        return
    ocean, depth_mask = load_masks(paths)
    years = years_of_split(cfg, "train")
    x, xd = _load_years(paths.filled_x, "X", years)
    y, yd = _load_years(paths.interim_glorys_t15, "thetao", years)
    stats = compute_stats(x, xd, y, yd, ocean, depth_mask, cfg)
    write_json(stats, paths.norm_stats)
    log.info("norm stats written for years %s", stats["years_used"])


def main() -> None:
    args, cfg = parse_and_load(base_parser("Step 7: normalization statistics"))
    run_normalize(cfg, Paths(cfg), args.force)


if __name__ == "__main__":
    main()
