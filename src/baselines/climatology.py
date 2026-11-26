"""Climatology baseline: training-year day-of-year mean (model_id ``climatology``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import write_prediction_store
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import doy_index
from src.dataset.arrays import load_clim_y, load_static
from src.dataset.splits import read_dates, split_indices

log = get_logger(__name__)


def predict_climatology(
    cfg: Config, paths: Paths, split: str
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    idx = split_indices(cfg, paths)[split]
    dates = pd.DatetimeIndex(read_dates(paths)["date"].to_numpy()[idx])
    _, depth_mask, _ = load_static(cfg, paths)
    pred = load_clim_y(paths)[doy_index(dates)].astype("f4")
    if np.isnan(pred[:, depth_mask]).any():
        raise ValueError("climatology has holes inside depth_mask")
    pred[:, ~depth_mask] = np.nan
    return pred, dates


def main() -> None:
    p = base_parser("Climatology baseline")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    args, cfg = parse_and_load(p)
    paths = Paths(cfg)
    for split in args.splits:
        pred, dates = predict_climatology(cfg, paths, split)
        write_prediction_store(pred, dates, cfg, "climatology", split, paths)
        log.info("climatology %s written", split)


if __name__ == "__main__":
    main()
