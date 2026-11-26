"""ENTRY: score every prediction store of a split against withheld GLORYS."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import read_prediction_store, write_netcdf
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import season_of
from src.dataset.arrays import load_clim_y, load_split_numpy
from src.evaluation.bootstrap import block_bootstrap_days, bootstrap_rmse, paired_difference_ci
from src.evaluation.metrics import add_skill, metrics_from_sums, per_depth_sums, rmse_map, summarize

log = get_logger(__name__)


def available_models(paths: Paths, split: str) -> list[str]:
    root = paths.results_dir / "predictions"
    order = ["climatology", "linreg", "cnn_unet", "vit_cnn", "vit_cnn_ens"]
    found = [p.parent.name for p in root.glob(f"*/{split}.zarr") if p.parent.name != "glorys"]
    return sorted(found, key=lambda m: (order.index(m) if m in order else 99, m))


def load_store(paths: Paths, model: str, split: str, dates, grid, depths) -> np.ndarray:
    ds = read_prediction_store(paths, model, split)
    if not pd.DatetimeIndex(ds["time"].values).equals(pd.DatetimeIndex(dates)):
        raise ValueError(f"{model}/{split}: store dates differ from the truth's")
    if not (np.allclose(ds["lat"], grid.lat) and np.allclose(ds["lon"], grid.lon)):
        raise ValueError(f"{model}/{split}: store grid differs from the target grid")
    if not np.allclose(ds["depth"], depths):
        raise ValueError(f"{model}/{split}: store depths differ from config")
    return ds["T_pred"].values


def evaluate(
    cfg: Config, paths: Paths, split: str, models: list[str] | None = None
) -> pd.DataFrame:
    a = load_split_numpy(cfg, paths, split, load_degc=True)
    t = a.Y_degC
    c = load_clim_y(paths)[a.doy]
    grid = grid_from_config(cfg)
    depths = np.asarray(cfg["depths_m"], "f8")
    models = models or available_models(paths, split)
    if not models:
        raise SystemExit(f"no prediction stores for split {split}")
    out = paths.metrics_dir / split
    out.mkdir(parents=True, exist_ok=True)
    bs = cfg["eval"]["bootstrap"]
    idx = block_bootstrap_days(len(a.dates), bs["block_days"], bs["n"], int(cfg["seed"]))
    seasons = season_of(a.dates, cfg)
    sums, per_depth, summary, per_season, daily = {}, [], [], [], []
    maps = {}
    map_d = [cfg["depths_m"].index(z) for z in cfg["eval"]["map_depths_m"]]
    preds = {}
    for m in models:
        p = load_store(paths, m, split, a.dates, grid, depths)
        preds[m] = p
        sums[m] = per_depth_sums(p, t, c, a.depth_mask)
    clim_df = (
        metrics_from_sums(sums["climatology"], depths=depths) if "climatology" in sums else None
    )
    for m in models:
        df = metrics_from_sums(sums[m], depths=depths)
        if clim_df is not None:
            df = add_skill(df, clim_df)
        per_depth.append(df.assign(model=m))
        sm = summarize(df, {k: v for k, v in cfg["depth_bands_m"].items()}, clim_df)
        _, ov = bootstrap_rmse(sums[m], idx)
        lo, hi = np.quantile(ov, [(1 - bs["ci"]) / 2, 1 - (1 - bs["ci"]) / 2])
        sm["rmse_ci_low"] = np.where(sm["scope"] == "overall", lo, np.nan)
        sm["rmse_ci_high"] = np.where(sm["scope"] == "overall", hi, np.nan)
        summary.append(sm.assign(model=m))
        for s in np.unique(seasons):
            dsel = np.flatnonzero(seasons == s)
            sdf = metrics_from_sums(sums[m], dsel, depths)
            per_season.append(sdf.assign(model=m, season=s))
            per_season.append(pd.DataFrame([{"depth": "overall", "rmse": sdf["rmse"].mean(),
                                             "bias": sdf["bias"].mean(), "acc": sdf["acc"].mean(),
                                             "model": m, "season": s}]))  # fmt: skip
        dr = np.sqrt(sums[m]["se"] / np.maximum(sums[m]["n"], 1)).mean(axis=1)
        daily.append(pd.DataFrame({"model": m, "date": a.dates.strftime("%Y-%m-%d"), "rmse": dr}))
        maps[m] = np.stack([rmse_map(preds[m][:, d], t[:, d], a.depth_mask[d]) for d in map_d])
    pd.concat(per_depth).to_csv(out / "per_depth.csv", index=False)
    summ = pd.concat(summary)
    summ.to_csv(out / "summary.csv", index=False)
    pd.concat(per_season).to_csv(out / "per_season.csv", index=False)
    pd.concat(daily).to_csv(out / "daily_rmse.csv", index=False)
    for a_name, b_name in (("vit_cnn", "cnn_unet"), ("vit_cnn_ens", "cnn_unet")):
        if a_name in sums and b_name in sums:
            pdiff = paired_difference_ci(sums[a_name], sums[b_name], idx, bs["ci"], depths)
            pdiff.to_csv(out / f"paired_{a_name}_vs_{b_name}.csv", index=False)
    mds = xr.Dataset(
        {"rmse": (("model", "depth", "lat", "lon"), np.stack([maps[m] for m in models]))},
        coords={"model": models, "depth": [cfg["depths_m"][d] for d in map_d], "lat": grid.lat,
                "lon": grid.lon},
    )  # fmt: skip
    if "climatology" in maps:
        mds["skill"] = 1 - mds["rmse"] / mds["rmse"].sel(model="climatology")
    write_netcdf(mds, out / "rmse_maps.nc")
    view = summ[summ["scope"] == "overall"][["model", "rmse", "bias", "acc"]]
    log.info("%s overall:\n%s", split, view.to_string(index=False))
    return summ


def main() -> None:
    p = base_parser("Evaluate prediction stores against GLORYS")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--models", nargs="*", default=None)
    args, cfg = parse_and_load(p)
    evaluate(cfg, Paths(cfg), args.split, args.models)
    sys.exit(0)


if __name__ == "__main__":
    main()
