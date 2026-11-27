from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.common.io import write_netcdf
from src.common.paths import Paths
from src.download import common
from src.download.ccmp import url_for
from src.download.check_coverage import (
    compress_ranges,
    coverage_for_files,
    format_ranges,
    switch_dates,
)
from src.download.check_coverage import (
    render_report as cov_report,
)
from src.download.common import (
    concat_daily,
    crop_to_bbox,
    download_bbox,
    plan_year_files,
    register_file,
    retry,
    stream_date_ranges,
    update_manifest,
)
from src.download.probe import (
    coverage_plan,
    estimate_sizes,
    extension_option,
    finalize,
    render_report,
    summarize_dataset,
    to_timestamp,
    verify_rows,
)
from src.download.run import parse_products


def _probe(**streams):
    return {"products": {p: {"streams": s} for p, s in streams.items()}}


def test_stream_date_ranges_first_wins(cfg):
    probe = _probe(glorys={
        "my": {"time_start": "2015-01-01", "time_end": "2021-06-30"},
        "myint": {"time_start": "2021-01-01", "time_end": "2023-12-31"},
    })  # fmt: skip
    runs = stream_date_ranges(cfg, "glorys", probe)
    assert [(s, a.date().isoformat(), b.date().isoformat()) for s, a, b in runs] == [
        ("my", cfg["dates"]["start"], "2021-06-30"),
        ("myint", "2021-07-01", "2022-12-31"),
    ]
    # no gaps, no overlaps
    days = [d for _, a, b in runs for d in pd.date_range(a, b)]
    assert len(days) == len(set(days)) == len(pd.date_range(cfg["dates"]["start"], "2022-12-31"))


def test_stream_date_ranges_gap_and_missing_stream(cfg):
    probe = _probe(glorys={
        "my": {"time_start": "2016-01-01", "time_end": "2019-12-31"},
        "myint": {"time_start": None, "time_end": None},
    })  # fmt: skip
    runs = stream_date_ranges(cfg, "glorys", probe)
    assert len(runs) == 1 and runs[0][2] == pd.Timestamp("2019-12-31")
    plan = coverage_plan(cfg, probe)["glorys"]
    assert plan["n_uncovered"] == 1096 and plan["uncovered"] == [
        "2020-01-01 … 2022-12-31 (1096 days)"
    ]


def test_plan_year_files(cfg):
    probe = _probe(glorys={
        "my": {"time_start": "2016-01-01", "time_end": "2021-06-30"},
        "myint": {"time_start": "2021-01-01", "time_end": "2023-12-31"},
    })  # fmt: skip
    plan = plan_year_files(cfg, "glorys", probe)
    keys = [(p["stream"], p["year"]) for p in plan]
    assert keys[-3:] == [("my", 2021), ("myint", 2021), ("myint", 2022)]
    p21 = {p["stream"]: p for p in plan if p["year"] == 2021}
    assert p21["my"]["end"] == pd.Timestamp("2021-06-30")
    assert p21["myint"]["start"] == pd.Timestamp("2021-07-01")
    assert [p["year"] for p in plan_year_files(cfg, "glorys", probe, [2018])] == [2018]


def test_crop_to_bbox_global_0_360():
    lat = np.arange(-89.875, 90, 0.25)
    lon = np.arange(0.125, 360, 0.25)
    ds = xr.Dataset({"v": (("time", "latitude", "longitude"), np.zeros((1, lat.size, lon.size), "f4"))},
                    coords={"time": [0], "latitude": lat[::-1], "longitude": lon})  # fmt: skip
    out = crop_to_bbox(ds, (79.0, 4.0, 101.0, 24.0))
    assert out["lat"].min() >= 4.0 and out["lat"].max() <= 24.0
    assert out["lon"].min() >= 79.0 and out["lon"].max() <= 101.0
    assert (np.diff(out["lat"].values) > 0).all() and (np.diff(out["lon"].values) > 0).all()
    assert out.sizes["lat"] == 80 and out.sizes["lon"] == 88
    with pytest.raises(ValueError):
        crop_to_bbox(
            ds, (200.0, 4.0, 201.0, 24.0)
        )  # outside after wrapping? lon wraps to -160..-159


def _day(date: str, value: float) -> xr.Dataset:
    return xr.Dataset({"v": (("time", "lat", "lon"), np.full((1, 2, 2), value, "f4"))},
                      coords={"time": pd.to_datetime([date]), "lat": [1.0, 2.0], "lon": [3.0, 4.0]})  # fmt: skip


def test_concat_daily_sorts_dedups_and_deletes(tmp_path):
    spec = [("2020-01-03", 3), ("2020-01-01", 1), ("2020-01-02", 2), ("2020-01-02", 22)]
    files = []
    for i, (d, v) in enumerate(spec):
        f = tmp_path / f"f{i}.nc"
        write_netcdf(_day(d, v), f)
        files.append(f)
    out = tmp_path / "year.nc"
    concat_daily(files, out, "prod", "s1")
    with xr.open_dataset(out) as ds:
        assert list(pd.DatetimeIndex(ds["time"].values).strftime("%Y-%m-%d")) == [
            "2020-01-01", "2020-01-02", "2020-01-03",
        ]  # fmt: skip
        assert ds["v"].isel(time=1).mean() == 22  # later file wins the duplicate
        assert ds.attrs["stream"] == "s1"
    assert not any(f.exists() for f in files)


def test_retry_calls_three_times(monkeypatch):
    sleeps = []
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    calls = []

    @retry(max_tries=5, base_delay_s=1.0)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("boom")
        return "ok"

    assert flaky() == "ok" and len(calls) == 3 and len(sleeps) == 2


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda s: None)

    @retry(max_tries=2, exceptions=(OSError,))
    def always():
        raise OSError("x")

    with pytest.raises(OSError):
        always()

    @retry(max_tries=3, exceptions=(OSError,))
    def other():
        raise KeyError("not retried")

    with pytest.raises(KeyError):
        other()


def test_update_manifest_threads(tiny_cfg):
    paths = Paths(tiny_cfg)
    errors = []

    def work(i):
        try:
            for j in range(5):
                update_manifest(paths, f"p/f_{i}_{j}.nc", {"i": i, "j": j})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(common.read_manifest(paths)) == 20
    assert not paths.manifest().with_suffix(".lock").exists()


def test_url_for(cfg):
    stream = cfg["products"]["ccmp"]["streams"][0]
    url = url_for(cfg, stream, pd.Timestamp("2016-03-05"))
    assert "Y2016/M03/" in url and url.endswith("CCMP_Wind_Analysis_20160305_V03.1_L4.nc")


def test_download_bbox(cfg):
    assert download_bbox(cfg) == (79.0, 4.0, 101.0, 24.0)


def _year_file(path: Path, days: pd.DatetimeIndex) -> None:
    ds = xr.Dataset({"v": (("time", "lat", "lon"), np.zeros((len(days), 2, 2), "f4"))},
                    coords={"time": days, "lat": [1.0, 2.0], "lon": [3.0, 4.0]})  # fmt: skip
    write_netcdf(ds, path)


def test_check_coverage_reports_hole(tmp_path):
    expected = pd.date_range("2020-01-01", "2020-01-10")
    days = expected.delete([3, 4, 5])  # 3-day hole
    f = tmp_path / "x_s1_2020.nc"
    _year_file(f, days)
    owner = coverage_for_files({"s1": [f]}, expected)
    missing = owner.index[owner.isna()]
    assert len(missing) == 3
    assert format_ranges(missing) == ["2020-01-04 … 2020-01-06 (3 days)"]
    assert compress_ranges(pd.DatetimeIndex(["2020-01-01", "2020-01-03"]))[1][2] == 1


def test_check_coverage_streams_and_ccmp_counts(tmp_path):
    expected = pd.date_range("2020-01-01", "2020-01-06")
    f1, f2 = tmp_path / "a.nc", tmp_path / "b.nc"
    _year_file(f1, expected[:3])
    _year_file(f2, expected[2:])
    owner = coverage_for_files({"s1": [f1], "s2": [f2]}, expected)
    assert list(owner) == ["s1", "s1", "s1", "s2", "s2", "s2"]
    assert switch_dates(owner) == [(expected[3], "s1", "s2")]
    # 4 times/day requirement: a daily file does not cover
    assert coverage_for_files({"s1": [f1]}, expected, times_per_day=4).isna().all()


def test_check_coverage_report_flags_missing(cfg):
    days = pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"])
    ok = pd.Series(["s1"] * len(days), index=days, dtype=object)
    bad = ok.copy()
    bad.iloc[10:13] = None
    text, df, flag = cov_report(cfg, {"ostia": bad, "smap": ok})
    assert not flag and "3 missing days" in text and set(df["product"]) == {"ostia", "smap"}
    smap_gap = ok.copy()
    smap_gap.iloc[5] = None
    assert cov_report(cfg, {"smap": smap_gap})[2]  # SMAP gaps are allowed


def test_register_file_checks_days(tiny_cfg):
    paths = Paths(tiny_cfg)
    f = paths.raw_year_file("ostia", "rep", 2020)
    days = pd.date_range("2020-01-01 12:00", periods=5)
    _year_file(f, days)
    rec = register_file(tiny_cfg, paths, f, "ostia", "rep",
                        expected_start="2020-01-01", expected_end="2020-01-05")  # fmt: skip
    assert rec["n_times"] == 5 and rec["time_end"] == "2020-01-05"
    assert common.is_done(paths, "ostia", "rep", 2020, "2020-01-01", "2020-01-05")
    assert not common.is_done(paths, "ostia", "rep", 2020, "2020-01-01", "2020-01-06")
    with pytest.raises(RuntimeError, match="missing"):
        register_file(tiny_cfg, paths, f, "ostia", "rep",
                      expected_start="2020-01-01", expected_end="2020-01-06")  # fmt: skip
    with pytest.raises(RuntimeError, match="unexpected"):
        register_file(tiny_cfg, paths, f, "ostia", "rep",
                      expected_start="2020-01-01", expected_end="2020-01-04")  # fmt: skip


def test_check_credentials(monkeypatch, tmp_path):
    for k in ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD",
              "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("COPERNICUSMARINE_CREDENTIALS_DIRECTORY", str(tmp_path / "cm"))
    with pytest.raises(RuntimeError, match="copernicusmarine login"):
        common.check_credentials("glorys")
    with pytest.raises(RuntimeError, match="EARTHDATA"):
        common.check_credentials("oscar")
    common.check_credentials("ccmp")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "u")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "p")
    common.check_credentials("glorys")


def test_parse_products():
    assert parse_products(["all"])[0] == "duacs"
    assert parse_products(["glorys,ostia"]) == ["ostia", "glorys"]  # run order, not input order
    with pytest.raises(ValueError):
        parse_products(["nope"])


def test_to_timestamp():
    assert to_timestamp(1451606400000, "milliseconds since 1970-01-01") == pd.Timestamp(
        "2016-01-01"
    )
    assert to_timestamp("2016-01-01T00:00:00Z") == pd.Timestamp("2016-01-01")
    assert to_timestamp(None) is None


def test_summarize_dataset():
    lat = np.arange(10.0, 5.0, -0.25)
    lon = np.arange(280.0, 285.0, 0.25)
    ds = xr.Dataset({"u": (("time", "longitude", "latitude"),
                           np.zeros((1, lon.size, lat.size), "f4"), {"units": "m/s"})},
                    coords={"time": pd.to_datetime(["2016-01-01 12:00"]), "latitude": lat,
                            "longitude": lon})  # fmt: skip
    s = summarize_dataset(ds)
    assert s["variables"]["u"]["dims"] == ["time", "longitude", "latitude"]
    assert s["variables"]["u"]["units"] == "m/s"
    assert s["lat"]["order"] == "descending" and s["lon"]["convention"] == "0-360"
    assert s["lat"]["step"] == 0.25 and s["time"]["hours"] == [12]


def test_probe_planning_and_report(cfg):
    sample = {"date": "2016-01-01", "bytes": 1_000_000, "dims": {}, "coords": [],
              "variables": {"thetao": {"dims": ["time", "depth", "lat", "lon"], "units": "degC"}},
              "lat": {"step": 0.083, "order": "ascending"},
              "lon": {"step": 0.083, "order": "ascending", "convention": "-180-180"},
              "depth": {"values": [0.494, 1.54]}}  # fmt: skip
    probe = {
        "created_utc": "now",
        "products": {
            "glorys": {"streams": {
                "my": {"ok": True, "dataset_id": "a", "time_start": "1993-01-01",
                       "time_end": "2021-06-30", "samples": [sample], "depths": [0.494, 1.54]},
                "myint": {"ok": True, "dataset_id": "b", "time_start": "2021-07-01",
                          "time_end": "2024-12-31", "samples": [sample]},
            }},
            "oscar": {"streams": {
                "final": {"ok": False, "error": "LookupError: nope", "short_name": "X"},
            }},
        },
    }  # fmt: skip
    finalize(cfg, probe)
    plan = probe["coverage_plan"]
    assert plan["glorys"]["switches"] == [
        {"date": "2021-07-01", "from": "my", "to": "myint", "split": "val"}
    ]
    assert plan["oscar"]["n_uncovered"] == len(
        pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"])
    )
    assert probe["extension"]["latest_common_end"] is None
    assert probe["sizes"]["glorys"]["gb"] > 0
    rows = verify_rows(cfg, probe, plan, "x: 1  # VERIFY\n")
    assert any(r["item"] == "glorys shallowest level (m)" and r["status"] == "OK" for r in rows)
    assert any(r["item"] == "channel u_cur: variable" and r["status"] == "CHANGE" for r in rows)
    text = render_report(cfg, probe, "x: 1  # VERIFY\n")
    assert "## 5. Extension option" in text and "CHANGE ->" in text
    # when a switch falls in the test split it is flagged
    probe["products"]["glorys"]["streams"]["myint"]["time_start"] = "2022-03-01"
    probe["products"]["glorys"]["streams"]["my"]["time_end"] = "2022-02-28"
    finalize(cfg, probe)
    assert "inside the test period" in render_report(cfg, probe, "")
    assert estimate_sizes(cfg, probe, probe["coverage_plan"])["glorys"]["gb"] > 0
    assert extension_option(cfg, probe)["per_product_end"]["glorys"] == "2024-12-31"


def test_oscar_like_granule_crops(tmp_path):
    """lat/lon on dims latitude/longitude, lon 0..360, time labelled julian."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from src.download.earthdata import crop_granule

    lat = np.arange(-89.75, 90, 0.25)
    lon = np.arange(0, 360, 0.25)
    u = np.zeros((1, lon.size, lat.size), "f4")
    ds = xr.Dataset(
        {"u": (("time", "longitude", "latitude"), u), "v": (("time", "longitude", "latitude"), u)},
        coords={"lat": ("latitude", lat), "lon": ("longitude", lon), "time": ("time", [10228.0])},
    )
    ds["time"].attrs.update(units="days since 1990-1-1", calendar="julian")
    src = tmp_path / "oscar_currents_final_20180102.nc"
    ds.to_netcdf(src)
    out = tmp_path / "out.nc"
    day = pd.Timestamp("2018-01-02")
    assert crop_granule(src, out, ["u", "v"], (79.0, 4.0, 101.0, 24.0), day, day)
    with xr.open_dataset(out) as r:
        assert r["u"].dims == ("time", "lat", "lon")
        assert pd.Timestamp(r["time"].values[0]) == day
        assert float(r.lon.min()) >= 79 and float(r.lat.max()) <= 24


def test_granule_date_patterns():
    import pandas as pd

    from src.download.earthdata import granule_date

    assert granule_date("RSS_smap_SSS_L3_8day_running_2018_005_FNL_v05.0.nc") == pd.Timestamp(
        "2018-01-05"
    )
    assert granule_date("oscar_currents_final_20180102.nc") == pd.Timestamp("2018-01-02")
    assert granule_date("nothing_here.nc") is None


def test_smap_like_granule_without_time_axis(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from src.download.earthdata import crop_granule

    lat = np.arange(-89.875, 90, 0.25)
    lon = np.arange(0.125, 360, 0.25)
    z = np.zeros((lat.size, lon.size), "f4")
    ds = xr.Dataset({"sss_smap": (("lat", "lon"), z), "time": (("lat", "lon"), z.astype("f8"))},
                    coords={"lat": lat, "lon": lon})  # fmt: skip
    ds["time"].attrs["units"] = "seconds since 2000-01-01"
    src = tmp_path / "RSS_smap_SSS_L3_8day_running_2018_005_FNL_v05.0.nc"
    ds.to_netcdf(src)
    out = tmp_path / "o.nc"
    day = pd.Timestamp("2018-01-05")
    assert crop_granule(src, out, ["sss_smap"], (79.0, 4.0, 101.0, 24.0), day, day)
    with xr.open_dataset(out) as r:
        assert r["sss_smap"].dims == ("time", "lat", "lon")
        assert pd.Timestamp(r.time.values[0]) == day


def test_concat_daily_keeps_packed_int16_lossless(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    from src.download.common import concat_daily

    rng = np.random.default_rng(0)
    files = []
    for m, n in ((1, 3), (2, 2)):
        t = pd.date_range(f"2018-0{m}-01", periods=n)
        v = (20 + rng.standard_normal((n, 2, 3, 4))).astype("f8")
        v[0, 0, 0, 0] = np.nan
        ds = xr.Dataset({"thetao": (("time", "depth", "latitude", "longitude"), v)},
                        coords={"time": t, "depth": [0.5, 10.0], "latitude": [1.0, 2, 3],
                                "longitude": [1.0, 2, 3, 4]})  # fmt: skip
        f = tmp_path / f"g_{m}.nc"
        ds.to_netcdf(f, encoding={"thetao": {"dtype": "int16", "scale_factor": 0.001,
                                             "add_offset": 20.0, "_FillValue": -32767}})  # fmt: skip
        files.append(f)
    expected = xr.concat([xr.open_dataset(f).load() for f in files], "time")
    out = tmp_path / "year.nc"
    concat_daily(files, out, "glorys", "my")
    with xr.open_dataset(out) as r:
        assert r.sizes["time"] == 5
        np.testing.assert_allclose(r.thetao.values, expected.thetao.values, equal_nan=True)
        assert np.isnan(r.thetao.values[0, 0, 0, 0])
    with xr.open_dataset(out, mask_and_scale=False) as raw:
        assert raw.thetao.dtype == np.int16
    assert not any(f.exists() for f in files)
