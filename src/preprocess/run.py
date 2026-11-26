"""Run preprocessing steps 1-9 in order (or a subset with --steps)."""

from __future__ import annotations

import sys

from src.common.cli import base_parser, parse_and_load
from src.common.logging_utils import get_logger
from src.common.paths import Paths

log = get_logger(__name__)
STEPS = ["grid", "regrid", "vertical", "masks", "clim", "fill", "normalize", "tensors", "sanity"]


def main() -> None:
    parser = base_parser("Run preprocessing steps in order")
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    steps = [s for s in args.steps.split(",") if s]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    from src.preprocess import (
        build_tensors,
        climatology,
        fill,
        make_grid,
        masks,
        normalize,
        regrid,  # fmt: skip
        sanity,
        vertical,
    )

    years = args.years or regrid.years_in_period(cfg)
    for step in [s for s in STEPS if s in steps]:
        log.info("=== preprocess step: %s ===", step)
        if step == "grid":
            make_grid.make_grid(cfg, paths, args.force)
        elif step == "regrid":
            regrid.run_regrid(cfg, paths, years, None, args.force)
        elif step == "vertical":
            for y in years:
                vertical.vertical_year(cfg, paths, y, args.force)
        elif step == "masks":
            masks.build_masks(cfg, paths, args.force)
        elif step == "clim":
            climatology.build_clim_inputs(cfg, paths, args.force)
            climatology.build_clim_y(cfg, paths, args.force)
        elif step == "fill":
            fill.run_fill(cfg, paths, args.force)
        elif step == "normalize":
            normalize.run_normalize(cfg, paths, args.force)
        elif step == "tensors":
            build_tensors.run_build(cfg, paths, args.force)
        elif step == "sanity":
            rows = sanity.run_sanity(cfg, paths)
            for status, name, detail in rows:
                log.info("%-8s %s  %s", status, name, detail)
            if any(r[0] == "FAIL" for r in rows):
                log.error("sanity checks failed")
                sys.exit(1)


if __name__ == "__main__":
    main()
