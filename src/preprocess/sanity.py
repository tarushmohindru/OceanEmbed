"""Step 9: sanity checks on data/processed (PASS/FAIL/WARNING) and diagnostic figures."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import years_of_split

log = get_logger(__name__)
PROFILE_POINTS = {"central Bay": (15, 88), "Andaman Sea": (10, 95), "northern Bay": (20, 89),
                  "south": (7, 85)}  # fmt: skip


def run_sanity(cfg: Config, paths: Paths, figures: bool = True) -> list[tuple[str, str, str]]:
    """Return ``(status, item, detail)`` rows; status in PASS/FAIL/WARNING."""
    from src.preprocess.masks import load_masks

    rows: list[tuple[str, str, str]] = []

    def item(ok: bool, name: str, detail: str = "", warn: bool = False):
        rows.append(("PASS" if ok else ("WARNING" if warn else "FAIL"), name, detail))

    ocean, depth_mask = load_masks(paths)
    X = xr.open_zarr(paths.x_zarr)["X"].values  # noqa: N806
    Y = xr.open_zarr(paths.y_zarr)["Y"].values  # noqa: N806
    dates = pd.read_csv(paths.dates_csv)
    train = (dates["split"] == "train").to_numpy()
    item(not np.isnan(X).any(), "X has no NaN")
    holes = np.isnan(Y) & depth_mask[None]
    item(holes.mean() <= 0.001, "Y NaN inside depth_mask <= 0.1%", f"{holes.mean():.5f}")
    stats = json.loads(paths.norm_stats.read_text())
    item(stats["years_used"] == years_of_split(cfg, "train"), "norm_stats years = train years",
         str(stats["years_used"]))  # fmt: skip
    for c, ch in enumerate(cfg["inputs"]):
        v = X[train, c][:, ocean]
        item(abs(v.mean()) < 0.05 and abs(v.std() - 1) < 0.05, f"X[{ch}] train mean~0 std~1",
             f"mean {v.mean():.3f} std {v.std():.3f}")  # fmt: skip
    for d, z in enumerate(cfg["depths_m"]):
        v = Y[train, d][:, depth_mask[d]]
        v = v[np.isfinite(v)]
        item(abs(v.mean()) < 0.05 and abs(v.std() - 1) < 0.05, f"Y[{z} m] train mean~0 std~1",
             f"mean {v.mean():.3f} std {v.std():.3f}")  # fmt: skip
    counts = depth_mask.sum(axis=(1, 2))
    item(bool(np.all(np.diff(counts) <= 0)), "depth_mask sizes non-increasing",
         str([int(c) for c in counts]))  # fmt: skip
    if paths.provenance.exists():
        prov = pd.read_csv(paths.provenance)
        split = dict(zip(dates["date"], dates["split"], strict=True))
        for product, g in prov.groupby("product"):
            g = g.sort_values("date")
            sw = g[g["stream"] != g["stream"].shift()].iloc[1:]
            for _, r in sw.iterrows():
                sp = split.get(r["date"], "none")
                item(sp not in ("test", "gap"), f"{product} stream switch on {r['date']}",
                     f"-> {r['stream']} (split {sp})", warn=True)  # fmt: skip
    if figures:
        _figures(cfg, paths, X, ocean, dates)
    return rows


def _figures(cfg: Config, paths: Paths, X, ocean, dates) -> None:  # noqa: N803
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = paths.figures_dir / "sanity"
    out.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(paths.grid) as g:
        lat, lon = g["lat"].values, g["lon"].values
    t = len(dates) // 2
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (c, ch) in zip(axes.flat, enumerate(cfg["inputs"]), strict=False):
        im = ax.pcolormesh(lon, lat, np.where(ocean, X[t, c], np.nan), shading="auto")
        ax.contour(lon, lat, ocean.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        ax.set_title(f"{ch} (normalized) {dates['date'][t]}")
        fig.colorbar(im, ax=ax)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "channels_one_day.png", dpi=200)
    plt.close(fig)
    Yc = xr.open_zarr(paths.y_degc_zarr)["Y"]  # noqa: N806
    fig, ax = plt.subplots(figsize=(6, 7))
    for name, (la, lo) in PROFILE_POINTS.items():
        prof = Yc.isel(time=t).sel(lat=la, lon=lo, method="nearest").values
        ax.plot(prof, cfg["depths_m"], marker="o", label=name)
    ax.invert_yaxis()
    ax.set_xlabel("temperature (degC)")
    ax.set_ylabel("depth (m)")
    ax.legend()
    fig.savefig(out / "glorys_profiles.png", dpi=200)
    plt.close(fig)
    if paths.fill_stats.exists():
        fs = pd.read_csv(paths.fill_stats)
        s = fs[fs["channel"] == "sss"]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(pd.to_datetime(s["date"]),
                s[["frac_time_interp", "frac_clim", "frac_spatial"]].sum(axis=1))  # fmt: skip
        ax.set_ylabel("SSS fill fraction")
        fig.savefig(out / "sss_fill_fraction.png", dpi=200)
        plt.close(fig)


def main() -> None:
    _, cfg = parse_and_load(base_parser("Step 9: sanity checks"))
    rows = run_sanity(cfg, Paths(cfg))
    for status, name, detail in rows:
        sys.stdout.write(f"{status:8s} {name}  {detail}\n")
    if any(r[0] == "FAIL" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
