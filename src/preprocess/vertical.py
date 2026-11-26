"""Step 3: GLORYS native levels -> the 15 target depths (linear in depth, never extrapolated)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import file_is_complete, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths

log = get_logger(__name__)


def interp_to_targets(
    T: np.ndarray,  # noqa: N803
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    top_level_max_m: float,
) -> np.ndarray:
    """Interpolate ``T`` (..., K, H, W) on ascending ``z_src`` to ``z_tgt`` -> (..., D, H, W).

    Depth 0 copies the shallowest level if it is within ``top_level_max_m``. A target is NaN if
    either bracketing level is NaN. Targets deeper than the deepest level raise.
    """
    z_src = np.asarray(z_src, "f8")
    z_tgt = np.asarray(z_tgt, "f8")
    if np.any(np.diff(z_src) <= 0):
        raise ValueError("z_src must be strictly ascending")
    if z_tgt.max() > z_src[-1] + 1e-9:
        raise ValueError(
            f"target depth {z_tgt.max()} m deeper than the deepest level {z_src[-1]} m"
        )
    T = np.asarray(T)  # noqa: N806
    out = []
    for d in z_tgt:
        if d == 0:
            if z_src[0] > top_level_max_m:
                raise ValueError(f"shallowest level {z_src[0]} m > {top_level_max_m} m; no 0 m")
            out.append(T[..., 0, :, :])
            continue
        if d < z_src[0]:
            raise ValueError(f"target depth {d} m above the shallowest level {z_src[0]} m")
        exact = np.flatnonzero(np.abs(z_src - d) < 1e-9)
        if exact.size:
            out.append(T[..., exact[0], :, :])
            continue
        k = int(np.searchsorted(z_src, d, side="right") - 1)
        w = (d - z_src[k]) / (z_src[k + 1] - z_src[k])
        out.append(T[..., k, :, :] * (1.0 - w) + T[..., k + 1, :, :] * w)  # NaN propagates
    return np.stack(out, axis=-3).astype("f4")


def vertical_year(cfg: Config, paths: Paths, year: int, force: bool) -> None:
    out = paths.interim_glorys_t15(year)
    if not force and file_is_complete(out):
        log.info("glorys_t15 %d: exists, skipping", year)
        return
    with xr.open_dataset(paths.interim_glorys(year)) as ds:
        ds = ds.load()
    z_tgt = np.asarray(cfg["depths_m"], "f8")
    arr = interp_to_targets(
        ds["thetao"].values, ds["depth"].values, z_tgt, cfg["preprocess"]["glorys_top_level_max_m"]
    )
    out_ds = xr.Dataset(
        {"thetao": (("time", "depth", "lat", "lon"), arr, {"units": "degC"})},
        coords={"time": ds["time"].values, "depth": z_tgt, "lat": ds["lat"].values,
                "lon": ds["lon"].values},
        attrs={"top_level_m": float(ds["depth"].values[0])},
    )  # fmt: skip
    write_netcdf(out_ds, out)
    log.info("glorys_t15 %d written (top native level %.3f m)", year, ds["depth"].values[0])


def main() -> None:
    parser = base_parser("Step 3: vertical interpolation to target depths")
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args, cfg = parse_and_load(parser)
    from src.preprocess.regrid import years_in_period

    paths = Paths(cfg)
    for y in args.years or years_in_period(cfg):
        vertical_year(cfg, paths, y, args.force)


if __name__ == "__main__":
    main()
