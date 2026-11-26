"""Probe every product/stream so that all ``# VERIFY`` items in config.yaml rest on evidence.

Writes ``results/probe/probe.json`` and ``results/probe/probe_report.md``. After running it,
STOP and review the report with the user before any full download.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import xarray as xr

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.io import write_json
from src.common.logging_utils import get_logger
from src.common.paths import Paths
from src.common.timeutil import split_of
from src.download.check_coverage import format_ranges
from src.download.common import (
    check_credentials,
    download_bbox,
    stream_date_ranges,
    tool_version,
)

log = get_logger(__name__)

LAT_NAMES = ("lat", "latitude", "nav_lat", "LATITUDE", "Latitude")
LON_NAMES = ("lon", "longitude", "nav_lon", "LONGITUDE", "Longitude")
GLOBAL_AREA_DEG2 = 360.0 * 180.0


# ---------------------------------------------------------------- pure helpers


def _date(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def to_timestamp(value, unit: str | None = None) -> pd.Timestamp | None:
    """Parse a catalogue time value (epoch number or ISO string) into a naive UTC Timestamp."""
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        u = (unit or "").lower()
        code = "s" if "second" in u and "milli" not in u else "D" if "day" in u else "ms"
        return pd.Timestamp(value, unit=code)
    ts = pd.Timestamp(str(value))
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def _first_name(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    return next((n for n in names if n in ds.variables or n in ds.dims), None)


def _axis_info(values: pd.Series | list | None) -> dict:
    import numpy as np

    v = np.asarray(values, dtype="float64")
    d = np.diff(v)
    return {
        "n": int(v.size),
        "first": float(v[0]),
        "last": float(v[-1]),
        "step": float(np.median(np.abs(d))) if d.size else None,
        "order": "ascending" if d.size and d[0] > 0 else "descending" if d.size else "single",
    }


def summarize_dataset(ds: xr.Dataset) -> dict:
    """Describe a dataset as delivered: names, units, dims order, grid, time, depths."""
    out: dict = {
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "coords": sorted(str(c) for c in ds.coords),
        "variables": {},
    }
    for name, var in ds.data_vars.items():
        out["variables"][str(name)] = {
            "dims": [str(d) for d in var.dims],
            "units": var.attrs.get("units"),
            "long_name": var.attrs.get("long_name"),
            "dtype": str(var.dtype),
            "fill_value": _jsonable(var.encoding.get("_FillValue", var.attrs.get("_FillValue"))),
        }
    lat, lon = _first_name(ds, LAT_NAMES), _first_name(ds, LON_NAMES)
    if lat and ds[lat].ndim == 1:
        out["lat"] = {"name": lat, **_axis_info(ds[lat].values)}
    if lon and ds[lon].ndim == 1:
        info = _axis_info(ds[lon].values)
        lo, hi = float(ds[lon].min()), float(ds[lon].max())
        info["convention"] = "0-360" if hi > 180 else "-180-180" if lo < 0 else "ambiguous (0..180)"
        out["lon"] = {"name": lon, **info}
    if "time" in ds.variables and ds["time"].size:
        t = pd.DatetimeIndex(ds["time"].values.reshape(-1))
        out["time"] = {
            "n": int(t.size),
            "first": t[0].isoformat(),
            "last": t[-1].isoformat(),
            "hours": sorted({int(h) for h in t.hour}),
        }
    for dname in ("depth", "lev", "deptht"):
        if dname in ds.variables and ds[dname].ndim == 1:
            vals = [round(float(x), 4) for x in ds[dname].values]
            out["depth"] = {"name": dname, "n": len(vals), "values": vals}
            break
    return out


def _jsonable(v):
    try:
        return v.item() if hasattr(v, "item") else v
    except (ValueError, AttributeError):
        return str(v)


def coverage_plan(cfg: Config, probe: dict) -> dict:
    """Per product: runs of stream use, uncovered date ranges and stream-switch dates."""
    days = pd.date_range(cfg["dates"]["start"], cfg["dates"]["end"], freq="D")
    plan = {}
    for product, rec in probe["products"].items():
        if "streams" not in rec:
            continue
        runs = stream_date_ranges(cfg, product, probe)
        covered = pd.DatetimeIndex([d for _, a, b in runs for d in pd.date_range(a, b)])
        uncovered = days.difference(covered)
        switches = []
        for (s0, _, _), (s1, a1, _) in zip(runs, runs[1:], strict=False):
            switches.append(
                {"date": _date(a1), "from": s0, "to": s1, "split": str(split_of([a1], cfg)[0])}
            )
        plan[product] = {
            "runs": [{"stream": s, "start": _date(a), "end": _date(b)} for s, a, b in runs],
            "uncovered": format_ranges(uncovered),
            "n_uncovered": int(len(uncovered)),
            "switches": switches,
        }
    return plan


def extension_option(cfg: Config, probe: dict) -> dict:
    """Latest date covered by every product through any stream."""
    ends = {}
    for product, rec in probe["products"].items():
        stream_ends = [
            pd.Timestamp(s["time_end"])
            for s in rec.get("streams", {}).values()
            if s.get("time_end")
        ]
        if "streams" in rec:
            ends[product] = max(stream_ends) if stream_ends else None
    known = [e for e in ends.values() if e is not None]
    latest = min(known) if len(known) == len(ends) and known else None
    return {
        "per_product_end": {p: _date(e) for p, e in ends.items()},
        "latest_common_end": _date(latest),
        "config_end": cfg["dates"]["end"],
        "can_extend": bool(latest is not None and latest > pd.Timestamp(cfg["dates"]["end"])),
    }


def estimate_sizes(cfg: Config, probe: dict, plan: dict) -> dict:
    """Full-period size estimates in GB: sample bytes x days (area-scaled for global files)."""
    lon0, lat0, lon1, lat1 = download_bbox(cfg)
    frac = (lon1 - lon0) * (lat1 - lat0) / GLOBAL_AREA_DEG2
    out = {}
    for product, rec in probe["products"].items():
        if "streams" not in rec:
            continue
        total = 0.0
        note = ""
        for run in plan.get(product, {}).get("runs", []):
            sample = rec["streams"].get(run["stream"], {}).get("samples") or []
            if not sample:
                continue
            per_day = sample[0].get("bytes", 0)
            if sample[0].get("global_file"):
                per_day *= frac
                note = "scaled by bbox area; global file sampled"
            days = (pd.Timestamp(run["end"]) - pd.Timestamp(run["start"])).days + 1
            total += per_day * days
        out[product] = {"gb": round(total / 1e9, 2), "note": note}
    return out


def verify_rows(cfg: Config, probe: dict, plan: dict, config_text: str) -> list[dict]:
    """One row per checkable ``# VERIFY`` item: ``{item, config, observed, status}``."""
    rows: list[dict] = []

    def add(item, conf, obs, ok):
        rows.append(
            {"item": item, "config": conf, "observed": obs, "status": "OK" if ok else "CHANGE"}
        )

    for product, rec in probe["products"].items():
        for name, s in rec.get("streams", {}).items():
            ident = s.get("dataset_id") or s.get("short_name") or s.get("url_template")
            add(
                f"{product}/{name} exists",
                ident,
                "found" if s.get("ok") else s.get("error"),
                s.get("ok"),
            )
    seen_vars: dict[str, dict] = {}
    for product, rec in probe["products"].items():
        for s in rec.get("streams", {}).values():
            for sample in s.get("samples", []):
                seen_vars.setdefault(product, {}).update(sample.get("variables", {}))
    for ch in cfg["inputs"]:
        product, var = cfg["channels"][ch]["product"], cfg["channels"][ch]["variable"]
        have = seen_vars.get(product)
        if have is None:
            add(f"channel {ch}: variable", var, "no sample downloaded", False)
        else:
            add(f"channel {ch}: variable", var, sorted(have), var in have)
    for product, var in (("ostia", "analysed_sst"), ("duacs", "sla"), ("smap", "sss_smap")):
        u = seen_vars.get(product, {}).get(var, {}).get("units")
        rows.append({"item": f"{product}.{var} units", "config": "see §5.3", "observed": u,
                     "status": "INFO"})  # fmt: skip
    for s in probe["products"].get("duacs", {}).get("streams", {}).values():
        if s.get("resolution_deg"):
            rows.append({"item": "duacs resolution (deg)", "config": "0.125 in dataset id",
                         "observed": s["resolution_deg"], "status": "INFO"})  # fmt: skip
            break
    for s in probe["products"].get("glorys", {}).get("streams", {}).values():
        depths = s.get("depths") or []
        if depths:
            top = depths[0]
            add("glorys shallowest level (m)", f"<= {cfg['preprocess']['glorys_top_level_max_m']}",
                top, top <= cfg["preprocess"]["glorys_top_level_max_m"])  # fmt: skip
            break
    for product in ("oscar", "smap"):
        p = plan.get(product)
        if p is not None:
            add(f"{product}: covers full period", f"{cfg['dates']['start']}..{cfg['dates']['end']}",
                p["uncovered"] or "yes", p["n_uncovered"] == 0)  # fmt: skip
    for product, p in plan.items():
        if p["switches"]:
            sw = ", ".join(f"{s['date']} ({s['split']})" for s in p["switches"])
            rows.append({"item": f"{product}: stream switches", "config": "-", "observed": sw,
                         "status": "INFO"})  # fmt: skip
    for product, rec in probe["products"].items():
        for name, s in rec.get("streams", {}).items():
            for sample in s.get("samples", []):
                if "lon" in sample:
                    rows.append({
                        "item": f"{product}/{name} grid as delivered",
                        "config": "-",
                        "observed": (f"lon {sample['lon']['convention']}, lat "
                                     f"{sample.get('lat', {}).get('order')}, dims "
                                     f"{[v['dims'] for v in sample['variables'].values()][:1]}"),
                        "status": "INFO",
                    })  # fmt: skip
                    break
    verify_lines = [
        f"L{i}: {line.strip()}"
        for i, line in enumerate(config_text.splitlines(), 1)
        if "VERIFY" in line
    ]
    rows.append({"item": "config lines tagged VERIFY", "config": "\n".join(verify_lines),
                 "observed": "-", "status": "INFO"})  # fmt: skip
    return rows


# ---------------------------------------------------------------- reporting


def _table(headers: list[str], rows: list[list]) -> list[str]:
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return lines + [""]


def render_report(cfg: Config, probe: dict, config_text: str = "") -> str:
    """Human-readable ``probe_report.md`` from a probe dict (sections 1-5 of the spec)."""
    plan, ext = probe["coverage_plan"], probe["extension"]
    L = [
        "# Probe report",
        "",
        f"Created {probe['created_utc']}. Review with the user, then STOP.",
        "",
    ]
    L += ["## 1. Products and streams", ""]
    rows = []
    for product, rec in probe["products"].items():
        for name, s in rec.get("streams", {}).items():
            vars_ = {
                v: i.get("units")
                for smp in s.get("samples", [])[:1]
                for v, i in smp["variables"].items()
            }
            smp = (s.get("samples") or [{}])[0]
            grid = ""
            if "lat" in smp:
                grid = f"lat step {smp['lat']['step']}, lon step {smp['lon']['step']}"
            dims = [i["dims"] for i in smp.get("variables", {}).values()][:1]
            rows.append([
                product, name, s.get("dataset_id") or s.get("short_name") or s.get("url_template"),
                s.get("version"), f"{s.get('time_start')} .. {s.get('time_end')}",
                vars_ or s.get("variables"), grid, dims,
                "ok" if s.get("ok") else f"ERROR: {s.get('error')}",
            ])  # fmt: skip
    L += _table(["product", "stream", "id", "version", "coverage", "variables (units)", "grid",
                 "dims", "status"], rows)  # fmt: skip
    if "argo" in probe["products"]:
        L += ["ARGO probe:", "```", str(probe["products"]["argo"]), "```", ""]
    L += ["## 2. Coverage plan (first stream wins)", ""]
    for product, p in plan.items():
        L.append(
            f"**{product}**: "
            + "; ".join(f"`{r['stream']}` {r['start']}..{r['end']}" for r in p["runs"])
        )
        if p["uncovered"]:
            L.append(f"- UNCOVERED ({p['n_uncovered']} days): " + ", ".join(p["uncovered"]))
        for s in p["switches"]:
            warn = " **<-- inside the test period**" if s["split"] in ("test", "gap") else ""
            L.append(
                f"- switch {s['from']} -> {s['to']} on {s['date']} (split: {s['split']}){warn}"
            )
        L.append("")
    L += ["## 3. VERIFY items", ""]
    L += _table(["item", "config", "observed", "status"],
                [[r["item"], r["config"], r["observed"],
                  r["status"] if r["status"] != "CHANGE" else f"CHANGE -> {r['observed']}"]
                 for r in verify_rows(cfg, probe, plan, config_text)])  # fmt: skip
    L += ["## 4. Size estimates for the configured period", ""]
    L += _table(["product", "GB", "note"],
                [[p, v["gb"], v["note"]] for p, v in probe["sizes"].items()])  # fmt: skip
    L += ["## 5. Extension option", ""]
    L.append(f"Per-product last date: {ext['per_product_end']}")
    L.append(f"Latest date covered by every product: **{ext['latest_common_end']}** "
             f"(configured end {ext['config_end']}; can extend: {ext['can_extend']}).")  # fmt: skip
    L += ["", "## Decisions needed (STOP)", "",
          "1. Corrected config.yaml values (section 3).",
          "2. Keep the configured period or extend (section 5).",
          "3. What to do about any stream switch inside the test year (section 2).",
          "4. Record the outcome in docs/decisions_log.md, then `touch results/probe/APPROVED`."]  # fmt: skip  # noqa: E501
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- network probes


def describe_cm(dataset_id: str) -> dict:
    """Catalogue metadata of a Copernicus Marine dataset (all toolbox access is here)."""
    import copernicusmarine

    cat = copernicusmarine.describe(dataset_id=dataset_id)
    for product in cat.products:
        for ds in product.datasets:
            if ds.dataset_id != dataset_id:
                continue
            labels = [v.label for v in ds.versions]
            version = sorted(ds.versions, key=lambda v: str(v.label))[-1]
            info: dict = {"version": version.label, "versions": labels, "variables": {}}
            for part in sorted(version.parts, key=lambda p: p.name != "default"):
                for svc in part.services:
                    if not svc.variables:
                        continue
                    for var in svc.variables:
                        info["variables"][var.short_name] = var.units
                        for c in var.coordinates:
                            cid = c.coordinate_id
                            if cid == "time" and "time_start" not in info:
                                info["time_start"] = _date(
                                    to_timestamp(c.minimum_value, c.coordinate_unit)
                                )
                                info["time_end"] = _date(
                                    to_timestamp(c.maximum_value, c.coordinate_unit)
                                )
                            elif cid in ("latitude", "longitude"):
                                key = "lat" if cid == "latitude" else "lon"
                                info.setdefault("resolution_deg", {})[key] = c.step
                            elif cid == "depth" and "depths" not in info:
                                info["depths"] = (
                                    [float(x) for x in c.values] if c.values else
                                    [c.minimum_value, c.maximum_value]
                                )  # fmt: skip
                    if "time_start" in info:
                        return info
            return info
    raise LookupError(f"dataset {dataset_id} not found in the Copernicus Marine catalogue")


def _sample_dates(cfg: Config, rec: dict) -> list[pd.Timestamp]:
    """Dates to sample: dates.start and dates.end clipped to the stream's own coverage."""
    a, b = pd.Timestamp(cfg["dates"]["start"]), pd.Timestamp(cfg["dates"]["end"])
    ts, te = rec.get("time_start"), rec.get("time_end")
    lo = max(a, pd.Timestamp(ts)) if ts else a
    hi = min(b, pd.Timestamp(te)) if te else b
    if hi < lo:
        return []
    return sorted({lo, hi})


def probe_cm_stream(cfg: Config, paths: Paths, product: str, stream: dict) -> dict:
    from src.download.copernicus import cm_subset

    pcfg = cfg["products"][product]
    rec: dict = {"dataset_id": stream["dataset_id"], "ok": False, "samples": []}
    try:
        rec.update(describe_cm(stream["dataset_id"]))
        depth = tuple(pcfg["depth_range_m"]) if "depth_range_m" in pcfg else None
        for d in _sample_dates(cfg, rec):
            out = paths.probe_dir / "raw" / f"{product}_{stream['name']}_{d:%Y%m%d}.nc"
            out.parent.mkdir(parents=True, exist_ok=True)
            cm_subset(stream["dataset_id"], pcfg["variables"], download_bbox(cfg),
                      f"{d:%Y-%m-%d}", f"{d:%Y-%m-%d}", out, depth)  # fmt: skip
            with xr.open_dataset(out) as ds:
                smp = {"date": _date(d), "file": out.name, "bytes": out.stat().st_size,
                       **summarize_dataset(ds)}  # fmt: skip
            rec["samples"].append(smp)
            if "depth" in smp and "depths" not in rec:
                rec["depths"] = smp["depth"]["values"]
        if "depth" in (rec["samples"] or [{}])[0]:
            rec["depths"] = rec["samples"][0]["depth"]["values"]
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 - recorded in the report, never fatal
        log.exception("probe %s/%s failed", product, stream["name"])
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def probe_earthdata_stream(cfg: Config, paths: Paths, product: str, stream: dict) -> dict:
    import earthaccess

    from src.download.earthdata import login_earthdata

    rec: dict = {"short_name": stream["short_name"], "ok": False, "samples": []}
    try:
        login_earthdata()
        cols = earthaccess.search_datasets(short_name=stream["short_name"])
        if not cols:
            raise LookupError(f"no collection with short_name {stream['short_name']}")
        umm = cols[0]["umm"]
        rec["version"] = umm.get("Version")
        rng = (umm.get("TemporalExtents") or [{}])[0].get("RangeDateTimes", [{}])[0]
        rec["time_start"] = _date(to_timestamp(rng.get("BeginningDateTime")))
        end = rng.get("EndingDateTime")
        rec["time_end"] = (
            _date(to_timestamp(end)) if end else _date(pd.Timestamp.now(UTC).tz_localize(None))
        )
        rec["ongoing"] = end is None
        bbox = download_bbox(cfg)
        for d in _sample_dates(cfg, rec):
            granules = earthaccess.search_data(
                short_name=stream["short_name"],
                temporal=(f"{d:%Y-%m-%d}", f"{d + pd.Timedelta(days=2):%Y-%m-%d}"),
                bounding_box=bbox,
            )
            rec.setdefault("granule_counts", {})[_date(d)] = len(granules)
            if not granules:
                continue
            local = paths.probe_dir / "raw" / f"{product}_{stream['name']}"
            files = earthaccess.download(granules[:1], local_path=str(local))
            with xr.open_dataset(files[0]) as ds:
                rec["samples"].append({"date": _date(d), "file": Path(files[0]).name,
                                       "bytes": Path(files[0]).stat().st_size, "global_file": True,
                                       **summarize_dataset(ds)})  # fmt: skip
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        log.exception("probe %s/%s failed", product, stream["name"])
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _url_exists(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=(30, 60))
    if r.status_code in (405, 501):
        with requests.get(url, stream=True, timeout=(30, 60)) as g:
            return g.status_code
    return r.status_code


def probe_ccmp_stream(cfg: Config, paths: Paths, stream: dict) -> dict:
    from src.download.ccmp import download_day, url_for

    rec: dict = {"url_template": stream["url_template"], "ok": False, "samples": [],
                 "months_present": {}}  # fmt: skip
    try:
        a, b = pd.Timestamp(cfg["dates"]["start"]), pd.Timestamp(cfg["dates"]["end"])
        checks = sorted({a, b, *pd.date_range(a, b, freq="MS")})
        status = {d: _url_exists(url_for(cfg, stream, d)) for d in checks}
        rec["months_present"] = {f"{d:%Y-%m-%d}": s for d, s in status.items()}
        rec["auth_required"] = any(s in (401, 403) for s in status.values())
        present = [d for d, s in status.items() if s == 200]
        if present:
            rec["time_start"] = _date(a if status[a] == 200 else min(present))
            end = b if status[b] == 200 else max(present) + pd.offsets.MonthEnd(0)
            rec["time_end"] = _date(min(end, b))
        for d in (
            sorted({pd.Timestamp(rec["time_start"]), pd.Timestamp(rec["time_end"])})
            if present
            else []
        ):
            f = download_day(
                url_for(cfg, stream, d), paths.probe_dir / "raw" / f"ccmp_{d:%Y%m%d}.nc"
            )
            with xr.open_dataset(f) as ds:
                rec["samples"].append({"date": _date(d), "file": f.name, "bytes": f.stat().st_size,
                                       "global_file": True, **summarize_dataset(ds)})  # fmt: skip
            f.unlink()
        rec["ok"] = bool(present)
        if not present:
            rec["error"] = f"no URL answered 200: {sorted(set(status.values()))}"
    except Exception as e:  # noqa: BLE001
        log.exception("probe ccmp failed")
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def probe_argo(cfg: Config) -> dict:
    from argopy import DataFetcher

    a, r = cfg["products"]["argo"], cfg["region"]
    year = a["fetch_years"][0]
    box = [r["lon_min"], r["lon_max"], r["lat_min"], r["lat_max"], 0.0, float(a["max_depth_m"]),
           f"{year}-01-01", f"{year}-01-08"]  # fmt: skip
    rec: dict = {"ok": False, "box": box}
    try:
        ds = DataFetcher(src=a["argopy_src"], mode="expert").region(box).to_xarray()
        rec.update(
            ok=True, n_points=int(ds.sizes.get("N_POINTS", 0)), variables=sorted(ds.variables)
        )
    except Exception as e:  # noqa: BLE001
        log.exception("probe argo failed")
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# ---------------------------------------------------------------- entry point


def run_probe(cfg: Config, paths: Paths, products: list[str] | None = None) -> dict:
    """Probe the requested products (default all) and return the probe dict."""
    probe: dict = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_dates": cfg["dates"],
        "tool_versions": {
            **tool_version("copernicusmarine"),
            **tool_version("earthaccess"),
            **tool_version("argopy"),
            **tool_version("xarray"),
        },  # fmt: skip
        "products": {},
    }
    for product, pcfg in cfg["products"].items():
        if products and product not in products:
            continue
        log.info("probing %s", product)
        if product == "argo":
            probe["products"]["argo"] = {"source": pcfg["source"], "info": probe_argo(cfg)}
            continue
        entry: dict = {"source": pcfg["source"], "streams": {}}
        try:
            check_credentials(product)
        except RuntimeError as e:
            log.warning("%s", e)
            entry["credentials_warning"] = str(e)
        for stream in pcfg["streams"]:
            if pcfg["source"] == "copernicusmarine":
                rec = probe_cm_stream(cfg, paths, product, stream)
            elif pcfg["source"] == "earthaccess":
                rec = probe_earthdata_stream(cfg, paths, product, stream)
            elif pcfg["source"] == "http":
                rec = probe_ccmp_stream(cfg, paths, stream)
            else:
                raise ValueError(f"unknown source {pcfg['source']!r} for {product}")
            entry["streams"][stream["name"]] = rec
        probe["products"][product] = entry
    finalize(cfg, probe)
    return probe


def finalize(cfg: Config, probe: dict) -> None:
    """Add the coverage plan, extension option and size estimates to ``probe`` in place."""
    probe["coverage_plan"] = coverage_plan(cfg, probe)
    probe["extension"] = extension_option(cfg, probe)
    probe["sizes"] = estimate_sizes(cfg, probe, probe["coverage_plan"])


def main() -> None:
    parser = base_parser("Probe every product and stream; write results/probe/probe_report.md")
    parser.add_argument("--products", nargs="*", default=None, help="subset of products")
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    probe = run_probe(cfg, paths, args.products)
    write_json(probe, paths.probe_dir / "probe.json")
    text = cfg.path.read_text(encoding="utf-8")
    report = render_report(cfg, probe, text)
    (paths.probe_dir / "probe_report.md").write_text(report, encoding="utf-8")
    log.info("wrote %s", paths.probe_dir / "probe_report.md")
    log.info("STOP: review probe_report.md with the user before downloading anything else.")


if __name__ == "__main__":
    main()
