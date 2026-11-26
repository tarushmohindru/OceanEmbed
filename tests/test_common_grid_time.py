from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.common.grid import Grid, grid_from_config, regular_bounds, standardize_coords
from src.common.timeutil import (
    daily_index,
    doy_index,
    floor_to_date,
    season_of,
    split_of,
    years_of_split,
)


def test_default_grid(cfg):
    g = grid_from_config(cfg)
    assert (g.H, g.W) == (72, 80)
    assert g.lat[0] == 5.125 and g.lat[-1] == 22.875
    assert g.lon[0] == 80.125 and g.lon[-1] == 99.875
    assert np.all(np.diff(g.lat) > 0)
    assert g.padded_shape(8) == (72, 80)
    assert g.padded_shape(32) == (96, 96)


def test_regular_bounds_roundtrip(cfg):
    g = grid_from_config(cfg)
    b = g.lat_bounds()
    assert b.shape == (73,) and b[0] == pytest.approx(5.0) and b[-1] == pytest.approx(23.0)
    np.testing.assert_allclose(0.5 * (b[:-1] + b[1:]), g.lat)
    with pytest.raises(ValueError):
        regular_bounds(np.array([0.0, 1.0, 3.0]))


def test_index_of_edges(cfg):
    g = grid_from_config(cfg)
    i, j = g.index_of(
        np.array([5.0, 22.99, 23.0, 4.9, 10.0]), np.array([80.0, 99.99, 90.0, 90, 100])
    )
    assert list(i) == [0, 71, -1, -1, -1]
    assert list(j) == [0, 79, -1, -1, -1]
    i, j = g.index_of(np.array([5.125]), np.array([80.125]))
    assert (i[0], j[0]) == (0, 0)


def test_cell_area(cfg):
    a = grid_from_config(cfg).cell_area_km2()
    assert a.shape == (72, 80) and (a > 0).all()
    assert (np.diff(a[:, 0]) < 0).all()  # shrinks toward the pole
    assert a[0, 0] == pytest.approx(
        0.25 * 111.19 * 0.25 * 111.19 * np.cos(np.deg2rad(5.125)), rel=1e-3
    )


def test_to_dataset(cfg):
    ds = grid_from_config(cfg).to_dataset()
    assert ds["lat_bnds"].shape == (72, 2) and ds["cell_area_km2"].shape == (72, 80)


def test_standardize_coords():
    ds = xr.Dataset(
        {"v": (("depth", "latitude", "longitude"), np.arange(1 * 2 * 4).reshape(1, 2, 4))},
        coords={"latitude": [10.0, 5.0], "longitude": [0.0, 90.0, 180.0, 270.0], "depth": [0.5]},
    )
    out = standardize_coords(ds)
    assert list(out["lon"].values) == [-180.0, -90.0, 0.0, 90.0]
    assert list(out["lat"].values) == [5.0, 10.0]
    assert "depth" not in out.dims
    assert float(out["v"].sel(lat=10.0, lon=90.0)) == 1.0  # (lat 10 = first row, lon 90 = col 1)


def test_doy_reference_values():
    dates = pd.to_datetime(
        [
            "2016-01-01",
            "2016-02-29",
            "2016-03-01",
            "2017-02-28",
            "2017-03-01",
            "2016-12-31",
            "2017-12-31",
        ]  # fmt: skip
    )
    assert list(doy_index(dates)) == [0, 59, 60, 58, 60, 365, 365]


def test_doy_same_slot_across_years():
    assert (
        doy_index(pd.to_datetime(["2016-07-04"]))[0] == doy_index(pd.to_datetime(["2017-07-04"]))[0]
    )


def test_split_counts(cfg):
    s = split_of(daily_index(cfg), cfg)
    counts = pd.Series(s).value_counts().to_dict()
    assert counts == {"train": 1827, "val": 350, "test": 350, "gap": 30}
    assert split_of(pd.to_datetime(["2015-06-01", "2023-06-01"]), cfg).tolist() == ["none", "none"]
    # gap days come first in each of val and test
    idx = daily_index(cfg)
    assert s[idx == "2021-01-15"][0] == "gap" and s[idx == "2021-01-16"][0] == "val"
    assert s[idx == "2022-01-15"][0] == "gap" and s[idx == "2022-01-16"][0] == "test"


def test_years_and_seasons(cfg):
    assert years_of_split(cfg, "train") == [2016, 2017, 2018, 2019, 2020]
    seasons = season_of(
        pd.to_datetime(["2022-01-10", "2022-04-01", "2022-07-01", "2022-11-30"]), cfg
    )
    assert list(seasons) == ["winter_monsoon", "pre_monsoon", "summer_monsoon", "post_monsoon"]


def test_floor_to_date():
    t = pd.DatetimeIndex(["2020-01-01 12:00", "2020-01-02 23:59"])
    assert list(floor_to_date(t)) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    tz = pd.DatetimeIndex(["2020-01-01 23:00"]).tz_localize("UTC").tz_convert("Asia/Kolkata")
    assert floor_to_date(tz)[0] == pd.Timestamp("2020-01-01")


def test_grid_dataclass_direct():
    g = Grid(lat=np.array([0.5, 1.5]), lon=np.array([0.5, 1.5, 2.5]), res=1.0)
    assert (g.H, g.W) == (2, 3)
