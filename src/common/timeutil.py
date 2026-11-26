"""Daily calendar, month-day slot index, split membership and seasons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import Config


def daily_index(cfg: Config) -> pd.DatetimeIndex:
    """Every day from ``dates.start`` to ``dates.end`` inclusive."""
    return pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"], freq="D")


def floor_to_date(times) -> pd.DatetimeIndex:
    """Timestamps -> UTC midnight of their date, timezone-naive."""
    idx = pd.DatetimeIndex(times)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def doy_index(dates) -> np.ndarray:
    """Month-day slot 0..365: 1 Jan = 0, 29 Feb = 59, 1 Mar = 60 in every year."""
    idx = pd.DatetimeIndex(dates)
    slot = idx.day_of_year.to_numpy() - 1
    skip = (~idx.is_leap_year) & (idx.month.to_numpy() >= 3)
    return np.where(skip, slot + 1, slot).astype(int)


def years_of_split(cfg: Config, split: str) -> list[int]:
    """Inclusive year range of ``train``/``val``/``test`` as a list."""
    lo, hi = cfg["splits"][split]
    return list(range(lo, hi + 1))


def split_of(dates, cfg: Config) -> np.ndarray:
    """Per date: train|val|test, gap (first ``gap_days`` of val/test) or none."""
    idx = floor_to_date(dates)
    out = np.full(len(idx), "none", dtype=object)
    gap = pd.Timedelta(days=int(cfg["splits"]["gap_days"]))
    start, end = pd.Timestamp(cfg["dates"]["start"]), pd.Timestamp(cfg["dates"]["end"])
    in_range = (idx >= start) & (idx <= end)
    for name in ("train", "val", "test"):
        lo, hi = cfg["splits"][name]
        mask = in_range & (idx.year >= lo) & (idx.year <= hi)
        out[mask] = name
        if name != "train":
            out[mask & (idx < pd.Timestamp(year=lo, month=1, day=1) + gap)] = "gap"
    return out.astype(str)


def season_of(dates, cfg: Config) -> np.ndarray:
    """Season name per date from ``eval.seasons`` (month lists)."""
    month_to_season = {m: s for s, months in cfg["eval"]["seasons"].items() for m in months}
    months = pd.DatetimeIndex(dates).month
    return np.array([month_to_season[int(m)] for m in months], dtype=str)
