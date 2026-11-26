"""Validate data/interim/ against its contract (grid, dates, dtype, units)."""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import daily_index

log = get_logger(__name__)


def check_interim(cfg: Config, paths: Paths) -> list[str]:
    """Return a list of problems (empty = OK)."""
    from src.preprocess.regrid import group_channels, years_in_period

    grid = grid_from_config(cfg)
    days = daily_index(cfg)
    problems = []
    for y in years_in_period(cfg):
        n = int((days.year == y).sum())
        files = {
            g: (paths.interim_channel_file(g, y), chs) for g, chs in group_channels(cfg).items()
        }
        files["glorys_t15"] = (paths.interim_glorys_t15(y), ["thetao"])
        for _name, (f, variables) in files.items():
            if not f.exists():
                problems.append(f"{f} missing")
                continue
            with xr.open_dataset(f) as ds:
                if not (np.allclose(ds["lat"], grid.lat, atol=1e-6)
                        and np.allclose(ds["lon"], grid.lon, atol=1e-6)):  # fmt: skip
                    problems.append(f"{f.name}: grid differs from target")
                if ds.sizes["time"] != n:
                    problems.append(f"{f.name}: {ds.sizes['time']} dates, expected {n}")
                for v in variables:
                    if v not in ds:
                        problems.append(f"{f.name}: variable {v} missing")
                        continue
                    if ds[v].dtype != np.float32:
                        problems.append(f"{f.name}: {v} dtype {ds[v].dtype}")
                    if not ds[v].attrs.get("units"):
                        problems.append(f"{f.name}: {v} has no units attribute")
    return problems


def main() -> None:
    _, cfg = parse_and_load(base_parser("Validate data/interim"))
    problems = check_interim(cfg, Paths(cfg))
    for p in problems:
        log.error(p)
    if problems:
        sys.exit(1)
    log.info("interim OK")


if __name__ == "__main__":
    main()
