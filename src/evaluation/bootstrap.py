"""Moving-block bootstrap over days (keeps day-to-day autocorrelation)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_days(n_days: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_days) day indices from contiguous circular blocks with random starts."""
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_days // block)
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)[None, None]) % n_days
    return idx.reshape(n_boot, -1)[:, :n_days]


def _rmse(sums: dict, idx: np.ndarray) -> np.ndarray:
    """(n_boot, D) RMSE for each resample."""
    se = sums["se"][idx].sum(axis=1)
    n = np.maximum(sums["n"][idx].sum(axis=1), 1)
    return np.sqrt(se / n)


def bootstrap_rmse(sums: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-depth (n_boot, D) and overall (n_boot,) RMSE (overall = mean over depths)."""
    r = _rmse(sums, idx)
    return r, r.mean(axis=1)


def paired_difference_ci(sums_a: dict, sums_b: dict, idx: np.ndarray, ci: float = 0.95,
                         depths=None) -> pd.DataFrame:  # fmt: skip
    """RMSE_a - RMSE_b per depth and overall using the same resampled days for both."""
    ra, rb = _rmse(sums_a, idx), _rmse(sums_b, idx)
    diff = ra - rb
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    point = np.sqrt(sums_a["se"].sum(0) / np.maximum(sums_a["n"].sum(0), 1)) - np.sqrt(
        sums_b["se"].sum(0) / np.maximum(sums_b["n"].sum(0), 1)
    )
    labels = list(depths) if depths is not None else list(range(diff.shape[1]))
    rows = [{"depth": d, "diff": point[k], "ci_low": np.quantile(diff[:, k], lo),
             "ci_high": np.quantile(diff[:, k], hi)} for k, d in enumerate(labels)]  # fmt: skip
    ov = diff.mean(axis=1)
    rows.append({"depth": "overall", "diff": point.mean(), "ci_low": np.quantile(ov, lo),
                 "ci_high": np.quantile(ov, hi)})  # fmt: skip
    return pd.DataFrame(rows)
