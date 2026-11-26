"""GLORYS truth copied into a store (model_id ``glorys``); used only as the ARGO reference."""

from __future__ import annotations

from src.common.cli import base_parser, parse_and_load
from src.common.io import write_prediction_store
from src.common.paths import Paths
from src.dataset.arrays import load_split_numpy


def main() -> None:
    p = base_parser("GLORYS reference store")
    p.add_argument("--splits", nargs="+", default=["test"])
    args, cfg = parse_and_load(p)
    paths = Paths(cfg)
    for split in args.splits:
        a = load_split_numpy(cfg, paths, split, load_degc=True)
        y = a.Y_degC.copy()
        y[:, ~a.depth_mask] = float("nan")
        # Rare NaN inside the mask (missing truth) cannot be stored; use the climatology there.
        hole = y != y
        hole &= a.depth_mask[None]
        if hole.any():
            from src.common.normalization import denormalize_y
            from src.dataset.arrays import load_static

            stats = load_static(cfg, paths)[2]
            y[hole] = denormalize_y(a.clim_Y_norm, stats, depth_axis=1)[hole]
        write_prediction_store(y.astype("f4"), a.dates, cfg, "glorys", split, paths)


if __name__ == "__main__":
    main()
