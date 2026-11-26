"""Metric definitions (single source of truth). Pure numpy, no IO.

Everything is built from per-day, per-depth sufficient statistics so that overall numbers,
seasons and bootstrap resamples share one code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUM_KEYS = ("n", "se", "e", "sp", "st", "spp", "stt", "spt", "sa", "sb", "saa", "sbb", "sab")


def per_depth_sums(p: np.ndarray, t: np.ndarray, c: np.ndarray, mask: np.ndarray) -> dict:
    """Per-day, per-depth sums, each (N, D). p, t, c: (N, D, H, W) degC; mask (D,H,W) or (N,D,H,W).

    Cells counted: mask & finite truth. Raises if the prediction is NaN at any counted cell.
    """
    m = np.broadcast_to(mask, t.shape) & np.isfinite(t)
    if np.isnan(p[m]).any():
        raise ValueError("prediction is NaN inside the evaluation mask")
    mf = m.astype("f8")
    pz = np.where(m, p, 0.0).astype("f8")
    tz = np.where(m, t, 0.0).astype("f8")
    cz = np.where(m, c, 0.0).astype("f8")
    a, b = pz - cz, tz - cz
    ax = (2, 3)
    return {
        "n": mf.sum(ax), "se": ((pz - tz) ** 2).sum(ax), "e": (pz - tz).sum(ax),
        "sp": pz.sum(ax), "st": tz.sum(ax), "spp": (pz * pz).sum(ax), "stt": (tz * tz).sum(ax),
        "spt": (pz * tz).sum(ax), "sa": a.sum(ax), "sb": b.sum(ax), "saa": (a * a).sum(ax),
        "sbb": (b * b).sum(ax), "sab": (a * b).sum(ax),
    }  # fmt: skip


def _pearson(n, sx, sy, sxx, syy, sxy):
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sxx / n - (sx / n) ** 2
    vy = syy / n - (sy / n) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vx * vy)
    return np.where((vx > 1e-12) & (vy > 1e-12), r, np.nan)


def metrics_from_sums(sums: dict, day_index: np.ndarray | None = None,
                      depths=None) -> pd.DataFrame:  # fmt: skip
    """One row per depth: depth, n, rmse, bias, corr, acc (aggregated over selected days)."""
    s = {k: (v if day_index is None else v[day_index]).sum(axis=0) for k, v in sums.items()}
    n = np.maximum(s["n"], 1)
    d = len(n)
    return pd.DataFrame({
        "depth": depths if depths is not None else np.arange(d),
        "n": s["n"].astype(int),
        "rmse": np.sqrt(s["se"] / n),
        "bias": s["e"] / n,
        "corr": _pearson(n, s["sp"], s["st"], s["spp"], s["stt"], s["spt"]),
        "acc": _pearson(n, s["sa"], s["sb"], s["saa"], s["sbb"], s["sab"]),
    })  # fmt: skip


def add_skill(df_model: pd.DataFrame, df_clim: pd.DataFrame) -> pd.DataFrame:
    out = df_model.copy()
    out["skill"] = 1.0 - out["rmse"].to_numpy() / df_clim["rmse"].to_numpy()
    return out


def summarize(df: pd.DataFrame, bands: dict, clim: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows: overall + each band (mean of per-depth values)."""
    rows = []
    scopes = {"overall": list(df["depth"])}
    scopes.update({k: list(v) for k, v in bands.items()})
    for scope, ds in scopes.items():
        sub = df[df["depth"].isin(ds)]
        row = {"scope": scope, **{k: float(sub[k].mean()) for k in ("rmse", "bias", "corr", "acc")}}
        if clim is not None:
            row["skill"] = 1.0 - row["rmse"] / float(clim[clim["depth"].isin(ds)]["rmse"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def rmse_map(p: np.ndarray, t: np.ndarray, mask_d: np.ndarray) -> np.ndarray:
    """(H, W) RMSE over days for one depth (p, t: (N, H, W))."""
    ok = mask_d[None] & np.isfinite(t)
    se = np.where(ok, (p - t) ** 2, 0.0).sum(0)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.sqrt(se / n), np.nan)


def rmse_per_depth(p: np.ndarray, t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(D,) pooled RMSE per depth."""
    s = per_depth_sums(p, t, np.zeros_like(t), mask)
    return metrics_from_sums(s)["rmse"].to_numpy()
