"""OceanEmbed viewer: subsurface temperature from surface data (Bay of Bengal box).

Run:  streamlit run app/streamlit_app.py -- --config config.yaml
Reads prediction stores (results/predictions/<model>/<split>.zarr), GLORYS truth and metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.paths import Paths  # noqa: E402

st.set_page_config(page_title="OceanEmbed", layout="wide")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--set", action="append", default=[])
    return p.parse_known_args(sys.argv[1:])[0]


@st.cache_resource
def get_paths(config: str, overrides: tuple[str, ...]):
    cfg = load_config(config, list(overrides))
    return cfg, Paths(cfg)


@st.cache_resource
def open_store(path: str) -> xr.Dataset:
    return xr.open_zarr(path)


@st.cache_resource
def open_truth(path: str) -> xr.Dataset:
    return xr.open_zarr(path)


def list_models(paths: Paths, split: str) -> list[str]:
    root = paths.results_dir / "predictions"
    return sorted(p.parent.name for p in root.glob(f"*/{split}.zarr"))


def heatmap(arr: np.ndarray, lat, lon, title: str, cmap: str, vmin=None, vmax=None):
    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(z=arr, x=lon, y=lat, colorscale=cmap, zmin=vmin, zmax=vmax,
                               colorbar={"thickness": 12}, hoverongaps=False))  # fmt: skip
    fig.update_layout(title=title, height=420, margin={"l": 10, "r": 10, "t": 40, "b": 10},
                      xaxis_title="lon (°E)", yaxis_title="lat (°N)")  # fmt: skip
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def main() -> None:
    a = _args()
    cfg, paths = get_paths(a.config, tuple(a.set))
    st.title("OceanEmbed — subsurface temperature from surface observations")
    st.caption("Bay of Bengal, 5–23°N, 80–100°E, 0.25°. Truth: GLORYS12 reanalysis (withheld).")

    split = st.sidebar.selectbox("Split", ["test", "val"])
    models = [m for m in list_models(paths, split) if m != "glorys"]
    if not models:
        st.warning(f"No prediction stores for '{split}' under {paths.results_dir / 'predictions'}.")
        st.stop()
    default = models.index("vit_cnn") if "vit_cnn" in models else 0
    model = st.sidebar.selectbox("Model", models, index=default)
    ds = open_store(str(paths.prediction_store(model, split)))
    truth = open_truth(str(paths.y_degc_zarr))
    times = pd.DatetimeIndex(ds["time"].values)
    date = st.sidebar.select_slider("Date", options=list(times.date), value=times.date[0])
    depths = [float(z) for z in ds["depth"].values]
    depth = st.sidebar.select_slider(
        "Depth (m)", options=depths, value=depths[min(5, len(depths) - 1)]
    )
    t = pd.Timestamp(date)

    pred = ds["T_pred"].sel(time=t, depth=depth).values
    true = truth["Y"].sel(time=t, depth=depth).values
    true = np.where(np.isfinite(pred), true, np.nan)
    lat, lon = ds["lat"].values, ds["lon"].values
    lo = float(np.nanpercentile(np.concatenate([pred.ravel(), true.ravel()]), 1))
    hi = float(np.nanpercentile(np.concatenate([pred.ravel(), true.ravel()]), 99))

    tab_maps, tab_profile, tab_metrics = st.tabs(["Maps", "Profile", "Metrics"])
    with tab_maps:
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(heatmap(pred, lat, lon, f"{model} — {depth:g} m", "Turbo", lo, hi),
                        width="stretch")  # fmt: skip
        c2.plotly_chart(heatmap(true, lat, lon, f"GLORYS — {depth:g} m", "Turbo", lo, hi),
                        width="stretch")  # fmt: skip
        err = pred - true
        m = float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else 1.0
        c3.plotly_chart(heatmap(err, lat, lon, "Error (model − GLORYS) °C", "RdBu_r", -m, m),
                        width="stretch")  # fmt: skip
        rmse = float(np.sqrt(np.nanmean(err**2)))
        st.metric(f"RMSE on {date} at {depth:g} m", f"{rmse:.3f} °C",
                  help="Over ocean cells valid at this depth")  # fmt: skip

    with tab_profile:
        import plotly.graph_objects as go

        c1, c2 = st.columns(2)
        plat = c1.slider(
            "Latitude", float(lat.min()), float(lat.max()), float(np.median(lat)), 0.25
        )
        plon = c2.slider(
            "Longitude", float(lon.min()), float(lon.max()), float(np.median(lon)), 0.25
        )
        fig = go.Figure()
        prof_t = truth["Y"].sel(time=t).sel(lat=plat, lon=plon, method="nearest").values
        fig.add_trace(go.Scatter(x=prof_t, y=depths, name="GLORYS", line={"color": "black"}))
        for mm in models:
            dsm = open_store(str(paths.prediction_store(mm, split)))
            prof = dsm["T_pred"].sel(time=t).sel(lat=plat, lon=plon, method="nearest").values
            fig.add_trace(go.Scatter(x=prof, y=depths, name=mm, mode="lines+markers"))
        fig.update_yaxes(autorange="reversed", title="depth (m)")
        fig.update_layout(xaxis_title="temperature (°C)", height=550,
                          title=f"Profile at {plat:.2f}°N {plon:.2f}°E on {date}")  # fmt: skip
        st.plotly_chart(fig, width="stretch")
        if not np.isfinite(prof_t).any():
            st.info("This point is land or outside the valid mask.")

    with tab_metrics:
        mdir = paths.metrics_dir / split
        if (mdir / "summary.csv").exists():
            s = pd.read_csv(mdir / "summary.csv")
            st.subheader("Summary (mean over depths / bands)")
            st.dataframe(s.pivot_table(index="model", columns="scope", values="rmse").round(3))
            pdp = pd.read_csv(mdir / "per_depth.csv")
            import plotly.express as px

            fig = px.line(pdp, x="rmse", y="depth", color="model", markers=True)
            fig.update_yaxes(autorange="reversed", title="depth (m)")
            fig.update_layout(xaxis_title="RMSE (°C)", height=500)
            st.plotly_chart(fig, width="stretch")
            if (mdir / "daily_rmse.csv").exists():
                d = pd.read_csv(mdir / "daily_rmse.csv", parse_dates=["date"])
                st.plotly_chart(px.line(d, x="date", y="rmse", color="model",
                                        title="Daily RMSE (mean over depths)"),
                                width="stretch")  # fmt: skip
        else:
            st.info(f"No metrics yet: run `python -m src.evaluation.evaluate --split {split}`.")


main()
