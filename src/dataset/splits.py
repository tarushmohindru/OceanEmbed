"""Split logic and leakage assertions (no torch)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.paths import Paths
from src.common.timeutil import split_of

SPLITS = ("train", "val", "test")


def read_dates(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.dates_csv)
    df["date"] = pd.to_datetime(df["date"])
    return df


def assert_no_leakage(idx: dict[str, np.ndarray], dates: pd.DatetimeIndex, cfg: Config) -> None:
    """Raise ``AssertionError`` if splits overlap, leave their years or violate the gap."""
    names = list(idx)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not np.intersect1d(idx[a], idx[b]).size, f"splits {a} and {b} share days"
    for name, ix in idx.items():
        lo, hi = cfg["splits"][name]
        years = dates[ix].year
        assert ((years >= lo) & (years <= hi)).all(), f"{name} has days outside {lo}-{hi}"
    gap = int(cfg["splits"]["gap_days"])
    for early, late in (("train", "val"), ("val", "test")):
        if (
            idx.get(early) is not None
            and idx.get(late) is not None
            and len(idx[early])
            and len(idx[late])
        ):
            delta = (dates[idx[late]].min() - dates[idx[early]].max()).days
            assert delta > gap, f"only {delta} days between {early} and {late} (gap {gap})"
    used = np.concatenate([v for v in idx.values()]) if idx else np.array([], int)
    for name in ("val", "test"):
        start = pd.Timestamp(year=cfg["splits"][name][0], month=1, day=1)
        banned = (dates >= start) & (dates < start + pd.Timedelta(days=gap))
        assert not np.isin(np.flatnonzero(banned), used).any(), f"gap days of {name} are used"


def split_indices(cfg: Config, paths: Paths) -> dict[str, np.ndarray]:
    """Sorted ``t_index`` arrays for train/val/test from ``processed/dates.csv``."""
    df = read_dates(paths)
    dates = pd.DatetimeIndex(df["date"])
    labels = split_of(dates, cfg)
    if not np.array_equal(labels, df["split"].astype(str).to_numpy()):
        raise AssertionError(
            "dates.csv split labels differ from the config (processed data built with another "
            "config?) - rebuild data/processed"
        )
    t_index = df["t_index"].to_numpy()
    idx = {s: np.sort(t_index[labels == s]) for s in SPLITS}
    assert_no_leakage(idx, dates, cfg)
    return idx
