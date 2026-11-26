"""Step 1: write the target grid to processed/grid.nc and interim/grid.nc."""

from __future__ import annotations

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths

log = get_logger(__name__)


def make_grid(cfg: Config, paths: Paths, force: bool = False) -> None:
    grid = grid_from_config(cfg)
    ds = grid.to_dataset()
    for p in (paths.grid, paths.interim_grid):
        if p.exists() and not force:
            continue
        write_netcdf(ds, p)
    log.info(
        "grid H=%d W=%d lat %.3f..%.3f lon %.3f..%.3f",
        grid.H, grid.W, grid.lat[0], grid.lat[-1], grid.lon[0], grid.lon[-1],
    )  # fmt: skip


def main() -> None:
    args, cfg = parse_and_load(base_parser("Step 1: target grid"))
    make_grid(cfg, Paths(cfg), args.force)


if __name__ == "__main__":
    main()
