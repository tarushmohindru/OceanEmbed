"""Gross-error range checks on input channels."""

from __future__ import annotations

import xarray as xr

from src.common.config import Config


def apply_ranges(ds: xr.Dataset, cfg: Config) -> tuple[xr.Dataset, dict[str, int]]:
    """Set values outside ``preprocess.qc_ranges[channel]`` (inclusive bounds) to NaN.

    Returns the dataset and the number of rejected finite values per channel.
    """
    ranges = cfg["preprocess"]["qc_ranges"]
    counts: dict[str, int] = {}
    out = ds.copy()
    for ch, (lo, hi) in ranges.items():
        if ch not in out.data_vars:
            continue
        v = out[ch]
        bad = v.notnull() & ((v < lo) | (v > hi))
        counts[ch] = int(bad.sum())
        out[ch] = v.where(~bad)
    return out, counts
