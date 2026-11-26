"""Synthetic, contract-conforming data/processed directory for tests and smoke runs (no torch).

Climatology, normalization statistics and tensors are built by the real preprocessing functions.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import write_json, write_netcdf, write_zarr
from src.common.normalization import load_norm_stats
from src.common.paths import Paths
from src.common.timeutil import daily_index, doy_index, split_of


def synthetic_config(root: Path, cfg: Config, years=(2016, 2017, 2018), H: int = 24,  # noqa: N803
                     W: int = 32) -> Config:  # noqa: N803  # fmt: skip
    raw = copy.deepcopy(cfg.raw)
    res = float(raw["grid_res"])
    raw["paths"]["data_root"] = str(Path(root))
    raw["region"].update(lat_min=5.0, lat_max=5.0 + H * res, lon_min=80.0, lon_max=80.0 + W * res)
    raw["dates"] = {"start": f"{years[0]}-01-01", "end": f"{years[-1]}-12-31"}
    raw["splits"].update(train=[years[0], years[0]], val=[years[1], years[1]],
                         test=[years[2], years[2]])  # fmt: skip
    return Config(raw=raw, path=cfg.path)


def _smooth_fields(rng, t_len: int, h: int, w: int, k: int = 3) -> np.ndarray:
    """(T, H, W) sum of k slowly drifting 2-D sinusoids, roughly unit amplitude."""
    t = np.arange(t_len)[:, None, None]
    ii = np.arange(h)[None, :, None]
    jj = np.arange(w)[None, None, :]
    out = np.zeros((t_len, h, w))
    for _ in range(k):
        kx, ky = rng.uniform(0.1, 0.5, 2)
        om = rng.uniform(0.01, 0.05)
        ph = rng.uniform(0, 2 * np.pi)
        out += np.sin(kx * jj + ky * ii + om * t + ph)
    return out / np.sqrt(k / 2)


def make_synthetic_processed(root: Path, cfg: Config, years=(2016, 2017, 2018), H: int = 24,  # noqa: N803
                             W: int = 32, seed: int = 0) -> Config:  # noqa: N803  # fmt: skip
    """Write a complete processed/ directory under ``root/processed``; return its Config."""
    from src.preprocess.build_tensors import make_tensors
    from src.preprocess.climatology import doy_climatology
    from src.preprocess.normalize import compute_stats

    scfg = synthetic_config(root, cfg, years, H, W)
    paths = Paths(scfg)
    rng = np.random.default_rng(seed)
    grid = grid_from_config(scfg)
    dates = daily_index(scfg)
    t_len = len(dates)
    depths = np.asarray(scfg["depths_m"], "f8")

    # geometry: continent in the north-east corner + 2-cell shelf (depths >= 200 m masked)
    land = np.zeros((H, W), bool)
    land[H - H // 3 :, W - W // 3 :] = True
    near = np.zeros_like(land)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            near |= np.roll(np.roll(land, di, 0), dj, 1)
    shelf = near & ~land
    ocean = ~land
    depth_mask = np.repeat(ocean[None], len(depths), 0)
    depth_mask[depths >= 200] &= ~shelf

    # inputs (physical units)
    season = np.sin(2 * np.pi * doy_index(dates) / 366.0)[:, None, None]
    fields = {
        "sst": 28 + 1.5 * season + 0.8 * _smooth_fields(rng, t_len, H, W),
        "sss": 33 - 0.5 * season + 0.5 * _smooth_fields(rng, t_len, H, W),
        "sla": 0.1 * _smooth_fields(rng, t_len, H, W),
        "u_cur": 0.3 * _smooth_fields(rng, t_len, H, W),
        "v_cur": 0.3 * _smooth_fields(rng, t_len, H, W),
        "u_wind": 5 * _smooth_fields(rng, t_len, H, W),
        "v_wind": 5 * _smooth_fields(rng, t_len, H, W),
    }
    x_raw = np.stack([fields[c] for c in scfg["inputs"]], axis=1).astype("f4")
    x_raw[:, :, land] = np.nan

    # targets (degC)
    z = depths[None, :, None, None]
    g = np.exp(-(((z - 100.0) / 50.0) ** 2))
    sst, sla = fields["sst"][:, None], fields["sla"][:, None] / 0.1
    y = sst * np.exp(-z / 150.0) + 0.5 * sla * g + 4.0
    y = y + 0.05 * rng.standard_normal(y.shape)
    y = np.clip(y, 2, 32).astype("f4")
    y[:, ~depth_mask] = np.nan

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    write_netcdf(grid.to_dataset(), paths.grid)
    write_netcdf(
        xr.Dataset(
            {"ocean": (("lat", "lon"), ocean.astype("u1")),
             "depth_mask": (("depth", "lat", "lon"), depth_mask.astype("u1"))},
            coords={"depth": depths, "lat": grid.lat, "lon": grid.lon}, attrs={"rule": "synthetic"},
        ),
        paths.masks,
    )  # fmt: skip
    stats = compute_stats(x_raw, dates, y, dates, ocean, depth_mask, scfg)
    write_json(stats, paths.norm_stats)
    ns = load_norm_stats(paths.norm_stats, scfg)
    train = split_of(dates, scfg) == "train"
    clim = doy_climatology(y, dates, train, scfg["preprocess"]["clim_window_days"])
    clim[:, ~depth_mask] = np.nan
    write_netcdf(
        xr.Dataset({"clim": (("doy", "depth", "lat", "lon"), clim, {"units": "degC"})},
                   coords={"doy": np.arange(366), "depth": depths, "lat": grid.lat,
                           "lon": grid.lon}),
        paths.clim_y,
    )  # fmt: skip
    x, yn, yc = make_tensors(x_raw, y, ocean, depth_mask, ns)
    coords = {"time": dates.values, "lat": grid.lat, "lon": grid.lon}
    c = len(scfg["inputs"])
    write_zarr(xr.Dataset({"X": (("time", "channel", "lat", "lon"), x)},
                          coords={**coords, "channel": np.array(scfg["inputs"], dtype=object)}),
               paths.x_zarr, {"time": 1, "channel": c, "lat": H, "lon": W})  # fmt: skip
    for arr, p in ((yn, paths.y_zarr), (yc, paths.y_degc_zarr)):
        write_zarr(xr.Dataset({"Y": (("time", "depth", "lat", "lon"), arr)},
                              coords={**coords, "depth": depths}),
                   p, {"time": 1, "depth": len(depths), "lat": H, "lon": W})  # fmt: skip
    pd.DataFrame({"t_index": np.arange(t_len), "date": dates.strftime("%Y-%m-%d"),
                  "year": dates.year, "doy_index": doy_index(dates),
                  "split": split_of(dates, scfg)}).to_csv(paths.dates_csv, index=False)  # fmt: skip
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "product": "synthetic", "stream": "s",
                  "dataset_id": "synthetic"}).to_csv(paths.provenance, index=False)  # fmt: skip
    pd.DataFrame({"date": np.repeat(dates.strftime("%Y-%m-%d"), c),
                  "channel": np.tile(scfg["inputs"], t_len), "frac_qc_rejected": 0.0,
                  "frac_time_interp": 0.0, "frac_clim": 0.0,
                  "frac_spatial": 0.0}).to_csv(paths.fill_stats, index=False)  # fmt: skip
    return scfg
