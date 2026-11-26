"""Load processed data as numpy arrays per split (no torch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from src.common.config import Config
from src.common.normalization import NormStats, load_norm_stats, normalize_y
from src.common.paths import Paths
from src.common.timeutil import doy_index
from src.dataset.splits import read_dates, split_indices


@dataclass
class SplitArrays:
    split: str
    t_index: np.ndarray  # (N,)
    dates: pd.DatetimeIndex  # (N,)
    doy: np.ndarray  # (N,)
    X: np.ndarray  # (N, C, H, W) normalized, no NaN, land 0
    Y: np.ndarray  # (N, D, H, W) normalized, NaN outside valid
    Y_degC: np.ndarray | None  # (N, D, H, W) degC
    ocean: np.ndarray  # (H, W) bool
    depth_mask: np.ndarray  # (D, H, W) bool
    clim_Y_norm: np.ndarray  # (N, D, H, W) normalized climatology of each sample's doy


def load_static(cfg: Config, paths: Paths) -> tuple[np.ndarray, np.ndarray, NormStats]:
    """(ocean (H, W) bool, depth_mask (D, H, W) bool, NormStats)."""
    with xr.open_dataset(paths.masks) as m:
        ocean = m["ocean"].values.astype(bool)
        depth_mask = m["depth_mask"].values.astype(bool)
    return ocean, depth_mask, load_norm_stats(paths.norm_stats, cfg)


def load_clim_y(paths: Paths) -> np.ndarray:
    """(366, D, H, W) degC target climatology."""
    with xr.open_dataset(paths.clim_y) as c:
        return c["clim"].values


def load_split_numpy(cfg: Config, paths: Paths, split: str, load_degc: bool = True) -> SplitArrays:
    idx = split_indices(cfg, paths)[split]
    df = read_dates(paths)
    dates = pd.DatetimeIndex(df["date"].to_numpy()[idx])
    ocean, depth_mask, stats = load_static(cfg, paths)
    X = xr.open_zarr(paths.x_zarr)["X"].isel(time=idx).values  # noqa: N806
    Y = xr.open_zarr(paths.y_zarr)["Y"].isel(time=idx).values  # noqa: N806
    Yc = xr.open_zarr(paths.y_degc_zarr)["Y"].isel(time=idx).values if load_degc else None  # noqa: N806
    n, c, h, w = X.shape
    d = len(cfg["depths_m"])
    assert c == len(cfg["inputs"]), (c, cfg["inputs"])
    assert Y.shape == (n, d, h, w), Y.shape
    assert ocean.shape == (h, w) and depth_mask.shape == (d, h, w)
    doy = doy_index(dates)
    clim = load_clim_y(paths)[doy]
    clim_norm = normalize_y(clim.astype("f4"), stats, depth_axis=1).astype("f4")
    return SplitArrays(split, idx, dates, doy, X, Y, Yc, ocean, depth_mask, clim_norm)
