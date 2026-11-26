"""Per-cell ridge regression from the 7 surface inputs to the 15 depths (model_id ``linreg``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import write_prediction_store
from src.common.logging_utils import get_logger
from src.common.normalization import denormalize_y
from src.common.paths import Paths
from src.dataset.arrays import load_split_numpy

log = get_logger(__name__)
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def fit_per_cell(X: np.ndarray, Y: np.ndarray, valid: np.ndarray, alpha: float  # noqa: N803
                 ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Ridge per cell: X (N,C,H,W), Y (N,D,H,W), valid (N,D,H,W) -> W (H,W,C,D), b (H,W,D).

    Per cell, days are kept when every depth that is ever valid there is finite that day.
    """
    n, c, h, w = X.shape
    d = Y.shape[1]
    Wt = np.full((h, w, c, d), np.nan)  # noqa: N806
    b = np.full((h, w, d), np.nan)
    ever = valid.any(axis=0)  # (D, H, W)
    cells = np.argwhere(ever.any(axis=0))
    if not len(cells):
        return Wt, b
    xi = X[:, :, cells[:, 0], cells[:, 1]].transpose(2, 0, 1).astype("f8")  # (P, N, C)
    yi = Y[:, :, cells[:, 0], cells[:, 1]].transpose(2, 0, 1).astype("f8")  # (P, N, D)
    vi = valid[:, :, cells[:, 0], cells[:, 1]].transpose(2, 0, 1)  # (P, N, D)
    ev = ever[:, cells[:, 0], cells[:, 1]].T  # (P, D)
    keep = np.all(vi | ~ev[:, None, :], axis=2)  # (P, N)
    dropped = int((~keep).sum())
    if dropped:
        log.info("linreg: dropped %d (cell, day) pairs with missing targets", dropped)
    k = keep[..., None].astype("f8")
    cnt = k.sum(axis=1).clip(min=1)  # (P, 1)
    xm = (xi * k).sum(1) / cnt  # (P, C)
    ym = (np.nan_to_num(yi) * k).sum(1) / cnt  # (P, D)
    xc = (xi - xm[:, None]) * k
    yc = (np.nan_to_num(yi) - ym[:, None]) * k
    xtx = np.einsum("pnc,pne->pce", xc, xc) + alpha * cnt[:, :, None] * np.eye(c)[None]
    xty = np.einsum("pnc,pnd->pcd", xc, yc)
    wp = np.linalg.solve(xtx, xty)  # (P, C, D)
    bp = ym - np.einsum("pc,pcd->pd", xm, wp)
    wp = np.where(ev[:, None, :], wp, np.nan)
    bp = np.where(ev, bp, np.nan)
    Wt[cells[:, 0], cells[:, 1]] = wp
    b[cells[:, 0], cells[:, 1]] = bp
    return Wt, b


def predict_per_cell(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:  # noqa: N803
    """(N, D, H, W) normalized predictions; NaN where the cell/depth has no model."""
    return (np.einsum("nchw,hwcd->ndhw", X.astype("f8"), np.nan_to_num(W))
            + np.moveaxis(b, -1, 0)[None]).astype("f4")  # fmt: skip


def _rmse_mean_degc(pred_n, arrays, stats) -> float:
    p = denormalize_y(pred_n, stats, depth_axis=1)
    t = denormalize_y(arrays.Y, stats, depth_axis=1)
    ok = np.isfinite(t) & arrays.depth_mask[None]
    r = [np.sqrt(np.mean((p[:, k][ok[:, k]] - t[:, k][ok[:, k]]) ** 2)) for k in range(p.shape[1])]
    return float(np.mean(r))


def select_alpha(cfg: Config, paths: Paths, alphas=ALPHAS, train=None, val=None) -> float:
    """Pick alpha by mean val RMSE (degC); fit on train only. Never touches test."""
    from src.dataset.arrays import load_static

    _, _, stats = load_static(cfg, paths)
    train = train or load_split_numpy(cfg, paths, "train", load_degc=False)
    val = val or load_split_numpy(cfg, paths, "val", load_degc=False)
    valid = train.depth_mask[None] & np.isfinite(train.Y)
    rows = []
    for a in alphas:
        W, b = fit_per_cell(train.X, train.Y, valid, a)  # noqa: N806
        rows.append({"alpha": a, "val_rmse_mean": _rmse_mean_degc(predict_per_cell(val.X, W, b),
                                                                   val, stats)})  # fmt: skip
        log.info("alpha %g: val RMSE %.4f", a, rows[-1]["val_rmse_mean"])
    df = pd.DataFrame(rows)
    df.to_csv(paths.metrics_dir / "linreg_alpha_selection.csv", index=False)
    return float(df.sort_values("val_rmse_mean").iloc[0]["alpha"])


def main() -> None:
    from src.dataset.arrays import load_static

    p = base_parser("Per-cell ridge regression baseline")
    p.add_argument("--variant", default="per_cell", choices=["per_cell"])
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    args, cfg = parse_and_load(p)
    paths = Paths(cfg)
    _, _, stats = load_static(cfg, paths)
    train = load_split_numpy(cfg, paths, "train", load_degc=False)
    val = load_split_numpy(cfg, paths, "val", load_degc=False)
    alpha = select_alpha(cfg, paths, train=train, val=val)
    log.info("selected alpha %g", alpha)
    W, b = fit_per_cell(train.X, train.Y, train.depth_mask[None] & np.isfinite(train.Y), alpha)  # noqa: N806
    for split in args.splits:
        arrays = val if split == "val" else load_split_numpy(cfg, paths, split, load_degc=False)
        pred = denormalize_y(predict_per_cell(arrays.X, W, b), stats, depth_axis=1).astype("f4")
        pred[:, ~arrays.depth_mask] = np.nan
        write_prediction_store(pred, arrays.dates, cfg, "linreg", split, paths,
                               {"alpha": alpha})  # fmt: skip


if __name__ == "__main__":
    main()
