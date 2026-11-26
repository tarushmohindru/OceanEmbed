"""Step 6: gap-fill the inputs (targets are never filled).

Order per channel and ocean cell: short-gap time interpolation -> day-of-year climatology ->
nearest valid ocean cell on the same day. Outside the ocean the values are set to NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import file_is_complete, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import CHANNEL_GROUP, Paths
from src.common.timeutil import doy_index, split_of

log = get_logger(__name__)


def time_interp(a: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill interior NaN gaps of length <= ``max_gap`` along axis 0.

    Args:
        a: (T, N) values.
    Returns:
        filled copy and a (T, N) bool array of the cells filled by this step.
    """
    a = np.asarray(a, "f8")
    t_len = a.shape[0]
    valid = np.isfinite(a)
    t = np.arange(t_len)[:, None]
    prev = np.maximum.accumulate(np.where(valid, t, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(valid, t, t_len)[::-1], axis=0)[::-1]
    gap = nxt - prev - 1
    can = ~valid & (prev >= 0) & (nxt < t_len) & (gap <= max_gap)
    cols = np.broadcast_to(np.arange(a.shape[1]), a.shape)
    p = np.clip(prev, 0, t_len - 1)
    n = np.clip(nxt, 0, t_len - 1)
    vp, vn = a[p, cols], a[n, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (t - prev) / (nxt - prev)
    out = a.copy()
    out[can] = (vp + (vn - vp) * w)[can]
    return out, can


def nearest_fill(field: np.ndarray, ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN ocean cells of a 2-D field with the nearest finite ocean cell."""
    valid = np.isfinite(field) & ocean
    need = ocean & ~valid
    if not need.any():
        return field, need
    if not valid.any():
        raise ValueError("no valid ocean cell to fill from")
    _, (ii, jj) = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = field.copy()
    out[need] = field[ii[need], jj[need]]
    return out, need


def fill_channel(
    x: np.ndarray, clim: np.ndarray, dates: pd.DatetimeIndex, ocean: np.ndarray, max_gap: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fill one channel ``x`` (T, H, W) inside ``ocean``; land -> NaN.

    Returns the filled array (float32) and per-day fractions of ocean cells filled by each step.
    """
    t_len = x.shape[0]
    n_ocean = max(int(ocean.sum()), 1)
    cells = x[:, ocean].astype("f8")  # (T, N)
    filled, by_time = time_interp(cells, max_gap)
    slots = doy_index(dates)
    c = clim[slots][:, ocean]
    by_clim = ~np.isfinite(filled) & np.isfinite(c)
    filled[by_clim] = c[by_clim]
    out = np.full(x.shape, np.nan, "f4")
    out[:, ocean] = filled
    by_spatial = np.zeros(t_len)
    for t in range(t_len):
        if np.isnan(out[t][ocean]).any():
            out[t], need = nearest_fill(out[t], ocean)
            by_spatial[t] = need.sum() / n_ocean
    assert not np.isnan(out[:, ocean]).any(), "NaN left inside the ocean after filling"
    stats = {
        "frac_time_interp": by_time.sum(axis=1) / n_ocean,
        "frac_clim": by_clim.sum(axis=1) / n_ocean,
        "frac_spatial": by_spatial,
    }
    return out, stats


def run_fill(cfg: Config, paths: Paths, force: bool = False) -> None:
    from src.preprocess.masks import load_masks
    from src.preprocess.regrid import years_in_period

    years = years_in_period(cfg)
    if not force and all(file_is_complete(paths.filled_x(y)) for y in years) and \
            paths.fill_stats.exists():  # fmt: skip
        log.info("filled X exists, skipping")
        return
    ocean, _ = load_masks(paths)
    grid = grid_from_config(cfg)
    with xr.open_dataset(paths.clim_inputs()) as c:
        clim = {ch: c[ch].values for ch in cfg["inputs"]}
    lo_hi = cfg["preprocess"]["qc_ranges"]
    chans, rows = [], []
    dates = None
    for ch in cfg["inputs"]:
        arrs, times = [], []
        for y in years:
            with xr.open_dataset(paths.interim_channel_file(CHANNEL_GROUP[ch], y)) as ds:
                arrs.append(ds[ch].values)
                times.append(pd.DatetimeIndex(ds["time"].values))
        dates = times[0].append(times[1:]) if len(times) > 1 else times[0]
        x = np.concatenate(arrs, axis=0)
        missing = np.isnan(x[:, ocean]).mean(axis=1)
        out, st = fill_channel(x, clim[ch], dates, ocean, cfg["preprocess"]["max_interp_gap_days"])
        lo, hi = lo_hi.get(ch, (-np.inf, np.inf))
        out_of_range = ((out[:, ocean] < lo - 1e-6) | (out[:, ocean] > hi + 1e-6)).mean()
        if out_of_range:
            log.warning("%s: %.4f of filled values outside the QC range", ch, out_of_range)
        chans.append(out)
        rows.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"), "channel": ch, "frac_missing": missing,
            "frac_qc_rejected": np.nan,  # QC counts are logged by regrid; not recounted here
            "frac_time_interp": st["frac_time_interp"], "frac_clim": st["frac_clim"],
            "frac_spatial": st["frac_spatial"],
        }))  # fmt: skip
    stats = pd.concat(rows, ignore_index=True)
    split = pd.Series(split_of(dates, cfg), index=dates.strftime("%Y-%m-%d"))
    stats["split"] = stats["date"].map(split)
    for (ch, sp), g in stats.groupby(["channel", "split"]):
        total = g[["frac_time_interp", "frac_clim", "frac_spatial"]].sum(axis=1).mean()
        (log.warning if total > 0.2 else log.info)(
            "fill %s %s: %.3f of ocean values filled", ch, sp, total
        )
    stats.to_csv(paths.fill_stats, index=False)
    X = np.stack(chans, axis=1)  # (T, C, H, W)  # noqa: N806
    for y in years:
        sel = dates.year == y
        ds = xr.Dataset(
            {"X": (("time", "channel", "lat", "lon"), X[sel])},
            coords={"time": dates[sel].values, "channel": list(cfg["inputs"]),
                    "lat": grid.lat, "lon": grid.lon},
            attrs={"note": "physical units, filled inside the ocean mask, NaN on land"},
        )  # fmt: skip
        write_netcdf(ds, paths.filled_x(y))


def main() -> None:
    args, cfg = parse_and_load(base_parser("Step 6: gap filling of inputs"))
    run_fill(cfg, Paths(cfg), args.force)


if __name__ == "__main__":
    main()
