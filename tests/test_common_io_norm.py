from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.common.config import Config
from src.common.grid import grid_from_config
from src.common.io import (
    atomic_write_path,
    file_is_complete,
    read_prediction_store,
    write_json,
    write_meta,
    write_netcdf,
    write_prediction_store,
    write_zarr,
)
from src.common.normalization import (
    NormStats,
    denormalize_y,
    load_norm_stats,
    normalize_x,
    normalize_y,
)
from src.common.paths import CHANNEL_GROUP, Paths


def _stats(c=7, d=15):
    r = np.random.default_rng(1)
    return NormStats(
        in_mean=r.normal(size=c).astype("f4"), in_std=r.uniform(0.5, 2, c).astype("f4"),
        out_mean=r.normal(size=d).astype("f4"), out_std=r.uniform(0.5, 2, d).astype("f4"),
        years_used=[2016],
    )  # fmt: skip


@pytest.mark.parametrize("axis", [1, -3])
def test_norm_roundtrip_numpy(axis):
    s = _stats()
    y = np.random.default_rng(2).normal(size=(3, 15, 4, 5)).astype("f4")
    z = normalize_y(y, s, axis)
    assert z.dtype == np.float32
    np.testing.assert_allclose(denormalize_y(z, s, axis), y, atol=1e-5)
    # a different layout: depth on the last axis
    y2 = np.moveaxis(y, 1, -1)
    np.testing.assert_allclose(normalize_y(y2, s, -1), np.moveaxis(z, 1, -1), atol=1e-6)


def test_norm_roundtrip_torch():
    torch = pytest.importorskip("torch")
    s = _stats()
    y = torch.randn(2, 15, 4, 5)
    z = normalize_y(y, s, 1)
    assert torch.is_tensor(z)
    torch.testing.assert_close(denormalize_y(z, s, 1), y, atol=1e-5, rtol=1e-5)
    x = torch.randn(2, 7, 4, 5)
    zx = normalize_x(x, s, 1)
    np.testing.assert_allclose(zx.numpy(), normalize_x(x.numpy(), s, 1), atol=1e-6)


def test_norm_wrong_axis_size():
    with pytest.raises(ValueError):
        normalize_x(np.zeros((2, 5, 3, 3), "f4"), _stats(), 1)


def test_load_norm_stats(cfg, tmp_path):
    d = {
        "years_used": [2016, 2017],
        "inputs": {c: {"mean": float(i), "std": 2.0} for i, c in enumerate(cfg["inputs"])},
        "targets": {str(z): {"mean": 20.0, "std": 3.0} for z in cfg["depths_m"]},
    }
    p = tmp_path / "n.json"
    write_json(d, p)
    s = load_norm_stats(p, cfg)
    assert s.in_mean.shape == (7,) and s.out_std.shape == (15,) and s.years_used == [2016, 2017]
    d["inputs"].pop("sst")
    write_json(d, p)
    with pytest.raises(ValueError):
        load_norm_stats(p, cfg)


def test_atomic_write_success_and_failure(tmp_path):
    f = tmp_path / "sub" / "a.txt"
    with atomic_write_path(f) as tmp:
        assert tmp.name == "a.txt.part"
        tmp.write_text("hi")
        assert not f.exists()
    assert f.read_text() == "hi" and not (tmp_path / "sub" / "a.txt.part").exists()

    with pytest.raises(RuntimeError), atomic_write_path(f) as tmp:
        tmp.write_text("broken")
        raise RuntimeError("boom")
    assert f.read_text() == "hi"  # previous content intact, no .part left
    assert not (tmp_path / "sub" / "a.txt.part").exists()

    g = tmp_path / "never.txt"
    with pytest.raises(RuntimeError), atomic_write_path(g) as tmp:
        tmp.write_text("x")
        raise RuntimeError
    assert not g.exists()


def test_write_netcdf_zarr_json_meta(tiny_cfg, tmp_path):
    ds = xr.Dataset(
        {"a": (("t", "x"), np.ones((3, 4), "f4"))}, coords={"t": [0, 1, 2], "x": range(4)}
    )
    write_netcdf(ds, tmp_path / "a.nc")
    assert file_is_complete(tmp_path / "a.nc")
    write_zarr(ds, tmp_path / "a.zarr", {"t": 1, "x": 4})
    assert file_is_complete(tmp_path / "a.zarr")
    assert not (tmp_path / "a.zarr.tmp").exists()
    write_json({"b": 1, "a": 2}, tmp_path / "x.json")
    assert list(json.loads((tmp_path / "x.json").read_text())) == ["a", "b"]
    write_meta(tmp_path / "m", tiny_cfg, {"k": 1})
    meta = json.loads((tmp_path / "m" / "_meta.json").read_text())
    assert meta["config_hash"] == tiny_cfg.hash() and meta["extra"] == {"k": 1}
    assert {"created_utc", "git_hash", "hostname", "python"} <= set(meta)
    assert not file_is_complete(tmp_path / "missing.nc")
    (tmp_path / "a.nc.part").write_text("x")
    assert not file_is_complete(tmp_path / "a.nc")


def _tiny_pred_cfg(tiny_cfg) -> Config:
    raw = copy.deepcopy(tiny_cfg.raw)
    raw["region"].update(lat_min=5.0, lat_max=6.0, lon_min=80.0, lon_max=81.0)  # 4 x 4
    return Config(raw=raw, path=tiny_cfg.path)


def _write_masks(paths, cfg, fill_inside=True):
    g = grid_from_config(cfg)
    d = len(cfg["depths_m"])
    mask = np.zeros((d, g.H, g.W), "u1")
    mask[:, :3, :3] = 1
    mask[5:, 0, 0] = 0
    ds = xr.Dataset({"depth_mask": (("depth", "lat", "lon"), mask)},
                    coords={"depth": cfg["depths_m"], "lat": g.lat, "lon": g.lon})  # fmt: skip
    ds.to_netcdf(paths.masks)
    return mask.astype(bool)


def test_prediction_store_roundtrip(tiny_cfg):
    cfg = _tiny_pred_cfg(tiny_cfg)
    paths = Paths(cfg)
    mask = _write_masks(paths, cfg)
    dates = pd.date_range("2021-02-01", periods=3)
    pred = np.random.default_rng(0).uniform(5, 30, (3, 15, 4, 4)).astype("f4")
    out = write_prediction_store(pred, dates, cfg, "vit_cnn_seed1", "val", paths, {"run_dir": "r"})
    assert out == paths.prediction_store("vit_cnn_seed1", "val")
    assert (out.parent / "_meta.json").exists()
    ds = read_prediction_store(paths, "vit_cnn_seed1", "val")
    t = ds["T_pred"].values
    assert t.dtype == np.float32 and t.shape == (3, 15, 4, 4)
    np.testing.assert_array_equal(
        np.isnan(t), ~np.broadcast_to(mask, t.shape)
    )  # NaN exactly outside
    np.testing.assert_allclose(t[:, mask], pred[:, mask])
    assert ds.attrs["model_id"] == "vit_cnn_seed1" and ds.attrs["run_dir"] == "r"
    assert ds.attrs["checkpoint"] == "" and ds.attrs["config_hash"] == cfg.hash()
    assert list(ds["depth"].values) == cfg["depths_m"]
    assert pd.DatetimeIndex(ds["time"].values).equals(dates)


def test_prediction_store_validation(tiny_cfg):
    cfg = _tiny_pred_cfg(tiny_cfg)
    paths = Paths(cfg)
    _write_masks(paths, cfg)
    dates = pd.date_range("2021-02-01", periods=2)
    ok = np.full((2, 15, 4, 4), 10.0, "f4")

    def call(pred=ok, d=dates, mid="m", split="val"):
        return write_prediction_store(pred, d, cfg, mid, split, paths)

    with pytest.raises(ValueError, match="model_id"):
        call(mid="Bad Name")
    with pytest.raises(ValueError, match="split"):
        call(split="train")
    with pytest.raises(ValueError, match="shape"):
        call(pred=ok[:1])
    with pytest.raises(ValueError, match="float32"):
        call(pred=ok.astype("f8"))
    with pytest.raises(ValueError, match="outside split"):
        call(d=pd.date_range("2021-01-01", periods=2))  # gap days
    with pytest.raises(ValueError, match="outside split"):
        call(split="test")
    with pytest.raises(ValueError, match="plausible"):
        call(pred=ok + 100)
    nan = ok.copy()
    nan[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        call(pred=nan)
    assert not paths.prediction_store("m", "val").exists()
    with pytest.raises(FileNotFoundError):
        read_prediction_store(paths, "absent", "val")


def test_paths(tiny_cfg):
    p = Paths(tiny_cfg)
    assert p.raw_year_file("glorys", "my", 2016).name == "glorys_my_2016.nc"
    assert p.raw_year_file("glorys", "my", 2016).parent.is_dir()
    assert p.filled_x(2020).name == "X_raw_2020.nc"
    assert p.interim_channel_file("cur", 2018).name == "cur_2018.nc"
    assert p.prediction_store("climatology", "test").name == "test.zarr"
    assert p.x_zarr.name == "X.zarr" and p.norm_stats.name == "norm_stats.json"
    assert CHANNEL_GROUP["u_cur"] == CHANNEL_GROUP["v_cur"] == "cur"
    assert set(CHANNEL_GROUP) == set(tiny_cfg["inputs"])
