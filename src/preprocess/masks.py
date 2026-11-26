"""Step 4: ocean mask and per-depth target mask (rules: data/processed/IMPLEMENTATION.md)."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import file_is_complete, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths

log = get_logger(__name__)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected component of a 2-D boolean mask."""
    labels, n = ndimage.label(mask)  # default structure = 4-connectivity in 2-D
    if n <= 1:
        return mask.copy()
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def compute_masks(
    glorys_valid_frac: np.ndarray,
    sst_valid_frac: np.ndarray,
    thres: float,
    keep_largest: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``ocean`` (H, W), ``depth_mask`` (D, H, W) and the cells removed as disconnected.

    Args:
        glorys_valid_frac: (D, H, W) fraction of days with finite GLORYS at each target depth.
        sst_valid_frac: (H, W) fraction of days with finite regridded SST.
    """
    ocean = (glorys_valid_frac[0] >= thres) & (sst_valid_frac >= thres)
    removed = np.zeros_like(ocean)
    if keep_largest:
        kept = largest_component(ocean)
        removed = ocean & ~kept
        ocean = kept
    depth_mask = ocean[None] & (glorys_valid_frac >= thres)
    nested = np.logical_and.accumulate(depth_mask, axis=0)
    n_fix = int((depth_mask & ~nested).sum())
    if n_fix:
        log.warning("depth_mask: %d cells valid below an invalid level; removed for nesting", n_fix)
    depth_mask = nested
    assert all((depth_mask[d + 1] <= depth_mask[d]).all() for d in range(len(depth_mask) - 1))
    return ocean, depth_mask, removed


def _valid_fractions(cfg: Config, paths: Paths) -> tuple[np.ndarray, np.ndarray]:
    from src.preprocess.regrid import years_in_period

    g_count = s_count = None
    n_days = 0
    for y in years_in_period(cfg):
        with xr.open_dataset(paths.interim_glorys_t15(y)) as g:
            gc = np.isfinite(g["thetao"].values).sum(axis=0)
        with xr.open_dataset(paths.interim_channel_file("sst", y)) as s:
            sc = np.isfinite(s["sst"].values).sum(axis=0)
            n = s.sizes["time"]
        g_count = gc if g_count is None else g_count + gc
        s_count = sc if s_count is None else s_count + sc
        n_days += n
    return g_count / n_days, s_count / n_days


def build_masks(cfg: Config, paths: Paths, force: bool = False) -> None:
    if not force and file_is_complete(paths.masks):
        log.info("masks.nc exists, skipping")
        return
    p = cfg["preprocess"]
    gfrac, sfrac = _valid_fractions(cfg, paths)
    ocean, depth_mask, removed = compute_masks(
        gfrac, sfrac, p["ocean_valid_fraction"], p["keep_largest_ocean_component"]
    )
    grid = grid_from_config(cfg)
    if removed.any():
        ii, jj = np.nonzero(removed)
        log.info("removed %d disconnected ocean cells: lat %.2f..%.2f lon %.2f..%.2f",
                 removed.sum(), grid.lat[ii].min(), grid.lat[ii].max(),
                 grid.lon[jj].min(), grid.lon[jj].max())  # fmt: skip
    log.info("ocean cells: %d; depth_mask counts: %s", ocean.sum(),
             [int(c) for c in depth_mask.sum(axis=(1, 2))])  # fmt: skip
    ds = xr.Dataset(
        {"ocean": (("lat", "lon"), ocean.astype("u1")),
         "depth_mask": (("depth", "lat", "lon"), depth_mask.astype("u1"))},
        coords={"depth": np.asarray(cfg["depths_m"], "f8"), "lat": grid.lat, "lon": grid.lon},
        attrs={"rule": f"valid on >= {p['ocean_valid_fraction']} of days (GLORYS depth 0 and SST); "
                       f"largest 4-connected component kept: {p['keep_largest_ocean_component']}; "
                       "depth_mask nested with depth"},
    )  # fmt: skip
    write_netcdf(ds, paths.masks)


def load_masks(paths: Paths) -> tuple[np.ndarray, np.ndarray]:
    """(ocean (H, W) bool, depth_mask (D, H, W) bool)."""
    with xr.open_dataset(paths.masks) as m:
        return m["ocean"].values.astype(bool), m["depth_mask"].values.astype(bool)


def main() -> None:
    args, cfg = parse_and_load(base_parser("Step 4: masks"))
    build_masks(cfg, Paths(cfg), args.force)


if __name__ == "__main__":
    main()
