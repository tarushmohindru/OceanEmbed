"""Step 8: model-ready tensors X.zarr, Y.zarr, Y_degC.zarr, dates.csv, provenance.csv."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import file_is_complete, write_meta, write_zarr
from src.common.logging_utils import get_logger
from src.common.normalization import load_norm_stats, normalize_x, normalize_y
from src.common.paths import Paths
from src.common.timeutil import doy_index, split_of

log = get_logger(__name__)


def make_tensors(
    x_raw: np.ndarray, y_degc: np.ndarray, ocean: np.ndarray, depth_mask: np.ndarray, stats
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X normalized, land = 0, no NaN), (Y normalized), (Y degC), both NaN off-mask."""
    x = normalize_x(x_raw.astype("f4"), stats, channel_axis=1)
    x[:, :, ~ocean] = 0.0
    if np.isnan(x).any():
        raise ValueError("X has NaN after normalization (fill step incomplete?)")
    y_c = np.where(depth_mask[None], y_degc, np.nan).astype("f4")
    n_hole = int((np.isnan(y_c) & depth_mask[None]).sum())
    if n_hole:
        log.info("Y has %d NaN values inside depth_mask (kept as NaN)", n_hole)
    y = normalize_y(y_c, stats, depth_axis=1).astype("f4")
    return x.astype("f4"), y, y_c


def run_build(cfg: Config, paths: Paths, force: bool = False) -> None:
    from src.preprocess.masks import load_masks
    from src.preprocess.normalize import _load_years
    from src.preprocess.regrid import years_in_period

    outs = [paths.x_zarr, paths.y_zarr, paths.y_degc_zarr]
    if not force and all(file_is_complete(p) for p in outs) and paths.dates_csv.exists():
        log.info("tensors exist, skipping")
        return
    grid = grid_from_config(cfg)
    ocean, depth_mask = load_masks(paths)
    stats = load_norm_stats(paths.norm_stats, cfg)
    years = years_in_period(cfg)
    x_raw, xd = _load_years(paths.filled_x, "X", years)
    y_degc, yd = _load_years(paths.interim_glorys_t15, "thetao", years)
    if not xd.equals(yd):
        raise ValueError("X and Y time axes differ")
    x, y, y_c = make_tensors(x_raw, y_degc, ocean, depth_mask, stats)
    depths = np.asarray(cfg["depths_m"], "f8")
    t, c, h, w = x.shape
    coords = {"time": xd.values, "lat": grid.lat, "lon": grid.lon}
    write_zarr(
        xr.Dataset({"X": (("time", "channel", "lat", "lon"), x)},
                   coords={**coords, "channel": np.array(cfg["inputs"], dtype=object)}),
        paths.x_zarr, {"time": 1, "channel": c, "lat": h, "lon": w},
    )  # fmt: skip
    for arr, path, units in ((y, paths.y_zarr, "normalized"), (y_c, paths.y_degc_zarr, "degC")):
        write_zarr(
            xr.Dataset({"Y": (("time", "depth", "lat", "lon"), arr, {"units": units})},
                       coords={**coords, "depth": depths}),
            path, {"time": 1, "depth": len(depths), "lat": h, "lon": w},
        )  # fmt: skip
    pd.DataFrame({
        "t_index": np.arange(t), "date": xd.strftime("%Y-%m-%d"), "year": xd.year,
        "doy_index": doy_index(xd), "split": split_of(xd, cfg),
    }).to_csv(paths.dates_csv, index=False)  # fmt: skip
    prov = sorted(paths.interim_dir.glob("provenance/*.csv"))
    if prov:
        pd.concat([pd.read_csv(p) for p in prov], ignore_index=True).sort_values(
            ["product", "date"]
        ).to_csv(paths.provenance, index=False)
    write_meta(paths.processed_dir, cfg, {"T": t, "C": c, "H": h, "W": w, "D": len(depths)})
    log.info("tensors written: X %s, Y %s", x.shape, y.shape)


def main() -> None:
    args, cfg = parse_and_load(base_parser("Step 8: build tensors"))
    run_build(cfg, Paths(cfg), args.force)


if __name__ == "__main__":
    main()
