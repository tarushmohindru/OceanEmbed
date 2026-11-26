"""Step 5: day-of-year climatologies of inputs (for filling) and targets (baseline/anomalies).

Training days only; circular centred smoothing over the 366 month-day slots.
"""

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
from src.common.timeutil import doy_index, split_of

log = get_logger(__name__)
N_SLOTS = 366


def _circular_window_sum(a: np.ndarray, window: int) -> np.ndarray:
    """Centred circular moving sum along axis 0 (length 366)."""
    if window % 2 != 1:
        raise ValueError("window_days must be odd")
    h = window // 2
    padded = np.concatenate([a[-h:], a, a[:h]], axis=0) if h else a
    c = np.cumsum(padded, axis=0, dtype="f8")
    c = np.concatenate([np.zeros((1, *a.shape[1:])), c], axis=0)
    return c[window:] - c[:-window] if h else a.astype("f8")


def doy_climatology(
    values: np.ndarray, dates: pd.DatetimeIndex, use: np.ndarray, window_days: int
) -> np.ndarray:
    """(366, ...) smoothed day-of-year mean of ``values`` (T, ...) over days where ``use``."""
    values = np.asarray(values)
    use = np.asarray(use, bool)
    if len(use) != values.shape[0] or len(dates) != values.shape[0]:
        raise ValueError("values, dates and use must share the time axis")
    slots = doy_index(dates)
    s = np.zeros((N_SLOTS, *values.shape[1:]), "f8")
    n = np.zeros_like(s)
    for t in np.flatnonzero(use):
        v = values[t]
        ok = np.isfinite(v)
        s[slots[t]] += np.where(ok, v, 0.0)
        n[slots[t]] += ok
    ss, nn = _circular_window_sum(s, window_days), _circular_window_sum(n, window_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nn > 0, ss / nn, np.nan).astype("f4")


def _train_use(dates, cfg: Config) -> np.ndarray:
    use = split_of(dates, cfg) == "train"
    assert not np.any(np.isin(split_of(dates[use], cfg), ["val", "test", "gap"]))
    return use


def build_clim_inputs(cfg: Config, paths: Paths, force: bool = False) -> None:
    """clim_inputs.nc: per channel (366, H, W) from interim files (after QC, before fill)."""
    from src.common.timeutil import years_of_split

    out = paths.clim_inputs()
    if not force and file_is_complete(out):
        log.info("clim_inputs.nc exists, skipping")
        return
    grid = grid_from_config(cfg)
    years = years_of_split(cfg, "train")
    data_vars = {}
    for ch in cfg["inputs"]:
        arrs, times = [], []
        for y in years:
            with xr.open_dataset(paths.interim_channel_file(CHANNEL_GROUP[ch], y)) as ds:
                arrs.append(ds[ch].values)
                times.append(pd.DatetimeIndex(ds["time"].values))
        dates = times[0].append(times[1:]) if len(times) > 1 else times[0]
        vals = np.concatenate(arrs, axis=0)
        clim = doy_climatology(
            vals, dates, _train_use(dates, cfg), cfg["preprocess"]["clim_window_days"]
        )
        data_vars[ch] = (("doy", "lat", "lon"), clim)
    ds = xr.Dataset(data_vars, coords={"doy": np.arange(N_SLOTS), "lat": grid.lat, "lon": grid.lon},
                    attrs={"years_used": ",".join(map(str, years))})  # fmt: skip
    write_netcdf(ds, out)


def build_clim_y(cfg: Config, paths: Paths, force: bool = False) -> None:
    """climatology_Y.nc: target climatology (366, D, H, W) in degC, train years, masked."""
    from src.common.timeutil import years_of_split
    from src.preprocess.masks import load_masks

    out = paths.clim_y
    if not force and file_is_complete(out):
        log.info("climatology_Y.nc exists, skipping")
        return
    grid = grid_from_config(cfg)
    _, depth_mask = load_masks(paths)
    years = years_of_split(cfg, "train")
    arrs, times = [], []
    for y in years:
        with xr.open_dataset(paths.interim_glorys_t15(y)) as ds:
            arrs.append(ds["thetao"].values)
            times.append(pd.DatetimeIndex(ds["time"].values))
    dates = times[0].append(times[1:]) if len(times) > 1 else times[0]
    vals = np.concatenate(arrs, axis=0)
    vals = np.where(depth_mask[None], vals, np.nan)
    clim = doy_climatology(
        vals, dates, _train_use(dates, cfg), cfg["preprocess"]["clim_window_days"]
    )
    clim = np.where(depth_mask[None], clim, np.nan).astype("f4")
    ds = xr.Dataset(
        {"clim": (("doy", "depth", "lat", "lon"), clim, {"units": "degC"})},
        coords={"doy": np.arange(N_SLOTS), "depth": np.asarray(cfg["depths_m"], "f8"),
                "lat": grid.lat, "lon": grid.lon},
        attrs={"years_used": ",".join(map(str, years))},
    )  # fmt: skip
    write_netcdf(ds, out)


def main() -> None:
    parser = base_parser("Step 5: climatologies")
    parser.add_argument("--which", choices=["inputs", "y", "both"], default="both")
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    if args.which in ("inputs", "both"):
        build_clim_inputs(cfg, paths, args.force)
    if args.which in ("y", "both"):
        build_clim_y(cfg, paths, args.force)


if __name__ == "__main__":
    main()
