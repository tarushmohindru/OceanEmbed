"""Pure-numpy separable regridding for regular lat-lon grids (primary regridder).

All sources are regular rectilinear grids, so conservative remapping factorizes into a latitude
matrix ``A_lat`` (H_t, H_s) and a longitude matrix ``A_lon`` (W_t, W_s):
``out = A_lat @ field @ A_lon.T``, with a NaN-aware normalization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.grid import Grid, regular_bounds
from src.common.logging_utils import get_logger

log = get_logger(__name__)


def _overlap(src_edges: np.ndarray, tgt_edges: np.ndarray) -> np.ndarray:
    """(N_t, N_s) overlap length between target and source intervals (edges ascending)."""
    s0, s1 = src_edges[:-1][None, :], src_edges[1:][None, :]
    t0, t1 = tgt_edges[:-1][:, None], tgt_edges[1:][:, None]
    return np.clip(np.minimum(s1, t1) - np.maximum(s0, t0), 0.0, None)


def overlap_matrix_lon(src_edges: np.ndarray, tgt_edges: np.ndarray) -> np.ndarray:
    """(W_t, W_s) overlap in degrees of longitude."""
    return _overlap(np.asarray(src_edges, "f8"), np.asarray(tgt_edges, "f8"))


def overlap_matrix_lat(src_edges: np.ndarray, tgt_edges: np.ndarray) -> np.ndarray:
    """(H_t, H_s) overlap in sin(latitude) units (proportional to spherical area)."""
    s = np.sin(np.deg2rad(np.asarray(src_edges, "f8")))
    t = np.sin(np.deg2rad(np.asarray(tgt_edges, "f8")))
    return _overlap(s, t)


def _apply(field: np.ndarray, a_lat: np.ndarray, a_lon: np.ndarray, na_thres: float) -> np.ndarray:
    f = np.asarray(field, dtype="f8")
    m = np.isfinite(f)
    v = np.where(m, f, 0.0)
    num = a_lat @ v @ a_lon.T
    den = a_lat @ m.astype("f8") @ a_lon.T
    cover = a_lat.sum(axis=1)[:, None] * a_lon.sum(axis=1)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
        ok = (den / cover) >= na_thres - 1e-12
    return np.where(ok & (den > 0), out, np.nan).astype("f4")


def conservative(field: np.ndarray, A_lat, A_lon, na_thres: float) -> np.ndarray:  # noqa: N803
    """Area-weighted (first-order conservative) remap of ``field`` (..., H_s, W_s), NaN-aware.

    A target cell is NaN when the valid fraction of its source coverage is below ``na_thres``.
    """
    return _apply(field, A_lat, A_lon, na_thres)


def bilinear_matrix(src_centres: np.ndarray, tgt_centres: np.ndarray) -> np.ndarray:
    """(N_t, N_s) linear-interpolation weights (2 non-zeros per row; clamped at the ends)."""
    s = np.asarray(src_centres, "f8")
    t = np.asarray(tgt_centres, "f8")
    k = np.clip(np.searchsorted(s, t, side="right") - 1, 0, s.size - 2)
    w = np.clip((t - s[k]) / (s[k + 1] - s[k]), 0.0, 1.0)
    b = np.zeros((t.size, s.size))
    b[np.arange(t.size), k] = 1.0 - w
    b[np.arange(t.size), k + 1] += w
    return b


def bilinear(field, B_lat, B_lon, na_thres: float) -> np.ndarray:  # noqa: N803
    """Bilinear interpolation with the same NaN normalization as ``conservative``."""
    return _apply(field, B_lat, B_lon, na_thres)


def _select_index(src: np.ndarray, tgt: np.ndarray) -> np.ndarray | None:
    idx = np.clip(np.searchsorted(src, tgt), 0, src.size - 1)
    for shift in (0, -1):
        j = np.clip(idx + shift, 0, src.size - 1)
        better = np.abs(src[j] - tgt) < np.abs(src[idx] - tgt)
        idx = np.where(better, j, idx)
    return idx if np.all(np.abs(src[idx] - tgt) < 1e-6) else None


def choose_method(src_lat: np.ndarray, src_lon: np.ndarray, grid: Grid) -> str:
    """``select`` (target centres are source centres), ``conservative`` (finer source) or
    ``bilinear`` (coarser source, not expected)."""
    src_lat, src_lon = np.asarray(src_lat, "f8"), np.asarray(src_lon, "f8")
    if (
        _select_index(src_lat, grid.lat) is not None
        and _select_index(src_lon, grid.lon) is not None
    ):
        return "select"
    step = max(np.median(np.abs(np.diff(src_lat))), np.median(np.abs(np.diff(src_lon))))
    if step <= grid.res + 1e-9:
        return "conservative"
    log.warning("source grid (%.4f deg) is coarser than the target; using bilinear", step)
    return "bilinear"


class Regridder:
    """Callable regridder from a regular source grid to the target ``Grid``."""

    def __init__(
        self,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        grid: Grid,
        na_thres: float,
        method: str | None = None,
    ):
        self.src_lat = np.asarray(src_lat, "f8")
        self.src_lon = np.asarray(src_lon, "f8")
        if np.any(np.diff(self.src_lat) <= 0) or np.any(np.diff(self.src_lon) <= 0):
            raise ValueError("source lat/lon must be strictly ascending (standardize_coords first)")
        self.grid = grid
        self.na_thres = float(na_thres)
        self.method = method or choose_method(self.src_lat, self.src_lon, grid)
        self.shape_out = (grid.H, grid.W)
        if self.method == "select":
            self.i_lat = _select_index(self.src_lat, grid.lat)
            self.i_lon = _select_index(self.src_lon, grid.lon)
            if self.i_lat is None or self.i_lon is None:
                raise ValueError("select: target centres are not all source centres")
        elif self.method == "conservative":
            se_lat, se_lon = regular_bounds(self.src_lat), regular_bounds(self.src_lon)
            te_lat, te_lon = grid.lat_bounds(), grid.lon_bounds()
            if se_lat[0] > te_lat[0] + 1e-9 or se_lat[-1] < te_lat[-1] - 1e-9:
                raise ValueError("source latitudes do not cover the target box")
            if se_lon[0] > te_lon[0] + 1e-9 or se_lon[-1] < te_lon[-1] - 1e-9:
                raise ValueError("source longitudes do not cover the target box")
            self.a_lat = overlap_matrix_lat(se_lat, te_lat)
            self.a_lon = overlap_matrix_lon(se_lon, te_lon)
        elif self.method == "bilinear":
            self.a_lat = bilinear_matrix(self.src_lat, grid.lat)
            self.a_lon = bilinear_matrix(self.src_lon, grid.lon)
        else:
            raise ValueError(f"unknown method {self.method!r}")

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """(..., H_s, W_s) -> (..., H, W) float32."""
        f = np.asarray(field)
        if f.shape[-2:] != (self.src_lat.size, self.src_lon.size):
            raise ValueError(f"field shape {f.shape} does not match source grid")
        if self.method == "select":
            return f[..., self.i_lat, :][..., self.i_lon].astype("f4")
        if self.method == "conservative":
            return conservative(f, self.a_lat, self.a_lon, self.na_thres)
        return bilinear(f, self.a_lat, self.a_lon, self.na_thres)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            src_lat=self.src_lat, src_lon=self.src_lon, lat=self.grid.lat, lon=self.grid.lon,
            res=self.grid.res, na_thres=self.na_thres, method=self.method,
        )  # fmt: skip

    @classmethod
    def load(cls, path: Path) -> Regridder:
        d = np.load(Path(path), allow_pickle=False)
        grid = Grid(lat=d["lat"], lon=d["lon"], res=float(d["res"]))
        return cls(d["src_lat"], d["src_lon"], grid, float(d["na_thres"]), str(d["method"]))
