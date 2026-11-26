"""Report missing days per product and which stream serves each date range."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import daily_index, floor_to_date, split_of

log = get_logger(__name__)
TIMES_PER_DAY = {"ccmp": 4}
ALLOWED_GAPS = {"smap"}  # filled during preprocessing; listed but not fatal


def compress_ranges(dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Consecutive-day runs as ``(first, last, n_days)``."""
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for d in pd.DatetimeIndex(dates).sort_values():
        if runs and d - runs[-1][1] == pd.Timedelta(days=1):
            runs[-1] = (runs[-1][0], d, runs[-1][2] + 1)
        else:
            runs.append((d, d, 1))
    return runs


def format_ranges(dates: pd.DatetimeIndex) -> list[str]:
    out = []
    for a, b, n in compress_ranges(dates):
        out.append(f"{a.date()}" if n == 1 else f"{a.date()} … {b.date()} ({n} days)")
    return out


def day_counts(files: list[Path]) -> pd.Series:
    """Number of time steps per calendar day over ``files``."""
    days = []
    for f in files:
        with xr.open_dataset(f) as ds:
            days.append(floor_to_date(ds["time"].values))
    if not days:
        return pd.Series(dtype=int)
    allv = days[0].append(days[1:]) if len(days) > 1 else days[0]
    return pd.Series(allv).value_counts().sort_index()


def coverage_for_files(
    files_by_stream: dict[str, list[Path]], expected: pd.DatetimeIndex, times_per_day: int = 1
) -> pd.Series:
    """Per expected date the first stream (dict order) that fully covers it, else ``None``."""
    owner = pd.Series([None] * len(expected), index=expected, dtype=object)
    for stream, files in files_by_stream.items():
        counts = day_counts(files)
        full = counts.index[counts >= times_per_day]
        free = owner.isna() & owner.index.isin(full)
        owner[free] = stream
    return owner


def switch_dates(owner: pd.Series) -> list[tuple[pd.Timestamp, str, str]]:
    """Dates where the serving stream changes: ``(date, from_stream, to_stream)``."""
    covered = owner.dropna()
    changes = covered != covered.shift()
    out = []
    for i in range(1, len(covered)):
        if changes.iloc[i]:
            out.append((covered.index[i], covered.iloc[i - 1], covered.iloc[i]))
    return out


def compute_coverage(cfg: Config, paths: Paths) -> dict[str, pd.Series]:
    """Coverage of every non-ARGO product from the raw year files on disk."""
    expected = daily_index(cfg)
    result = {}
    for product, pcfg in cfg["products"].items():
        if product == "argo":
            continue
        files = {
            s["name"]: sorted((paths.raw_dir / product / s["name"]).glob(f"{product}_*_*.nc"))
            for s in pcfg["streams"]
        }
        result[product] = coverage_for_files(files, expected, TIMES_PER_DAY.get(product, 1))
    return result


def render_report(cfg: Config, coverage: dict[str, pd.Series]) -> tuple[str, pd.DataFrame, bool]:
    """Markdown report, long-format dataframe ``product,date,stream`` and overall OK flag."""
    lines = ["# Download coverage", ""]
    rows = []
    ok = True
    for product, owner in coverage.items():
        missing = owner.index[owner.isna()]
        lines += [f"## {product}", ""]
        if len(missing):
            lines.append(f"**{len(missing)} missing days**")
            lines += [f"- {r}" for r in format_ranges(missing)]
            if product not in ALLOWED_GAPS:
                ok = False
        else:
            lines.append("No missing days.")
        lines.append("")
        covered = owner.dropna()
        if len(covered):
            lines.append("Streams used:")
            run_start = covered.index[0]
            for i in range(1, len(covered) + 1):
                end_run = i == len(covered) or covered.iloc[i] != covered.iloc[i - 1]
                if end_run:
                    lines.append(
                        f"- `{covered.iloc[i - 1]}`: {run_start.date()} … {covered.index[i - 1].date()}"  # noqa: E501
                    )
                    if i < len(covered):
                        run_start = covered.index[i]
            for d, a, b in switch_dates(owner):
                sp = split_of([d], cfg)[0]
                warn = (
                    "  **WARNING: switch inside the test split**" if sp in ("test", "gap") else ""
                )
                lines.append(f"- switch {a} → {b} on {d.date()} (split: {sp}){warn}")
            lines.append("")
        rows += [(product, d, s) for d, s in covered.items()]
    df = pd.DataFrame(rows, columns=["product", "date", "stream"])
    return "\n".join(lines), df, ok


def main() -> None:
    parser = base_parser("Report missing days per product")
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    coverage = compute_coverage(cfg, paths)
    text, df, ok = render_report(cfg, coverage)
    (paths.probe_dir / "coverage.md").write_text(text, encoding="utf-8")
    df.to_csv(paths.probe_dir / "coverage.csv", index=False)
    sys.stdout.write(text + "\n")
    if not ok:
        log.error("missing days found in a non-SMAP product")
        sys.exit(1)


if __name__ == "__main__":
    main()
