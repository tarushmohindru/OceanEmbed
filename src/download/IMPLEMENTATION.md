# `src/download/` — Data acquisition

Purpose: fetch every product for region ± buffer and the study period into `data/raw/`
(contract: `data/raw/IMPLEMENTATION.md`), resumably, with a manifest — and, **before that**,
probe every product so that all **Verify** items in `config.yaml` are settled by evidence.

Runs on: Lightning CPU machine. Never on the authoring machine (except unit tests with mocks).

## Files

```text
src/download/
  __init__.py
  common.py            # retries, locking, manifest, cropping, concatenation, credential checks
  probe.py             # ENTRY: inspect every product/stream, write probe report
  copernicus.py        # ENTRY: GLORYS, OSTIA, DUACS via copernicusmarine
  earthdata.py         # ENTRY: OSCAR, SMAP via earthaccess
  ccmp.py              # ENTRY: CCMP via HTTP
  argo_fetch.py        # ENTRY: ARGO profiles via argopy
  run.py               # ENTRY: run several downloaders in sequence
  check_coverage.py    # ENTRY: missing-day report per product and stream map
```

---

## Credentials (checked by `common.check_credentials(product)`)

| Service | How the code authenticates | How the user sets it up on Lightning |
| --- | --- | --- |
| Copernicus Marine | `copernicusmarine` reads its stored credentials or env vars `COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD` (**Verify** names with `copernicusmarine login --help`) | `copernicusmarine login` once, or Lightning secrets → env vars |
| NASA Earthdata | `earthaccess.login(strategy="environment")` with `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`; fallback `strategy="netrc"` | Lightning secrets → env vars |
| RSS (CCMP) | none expected (**Verify**: if the server requires registration, **STOP** and ask the user) | — |
| Argo | none | — |

Never log, print or write credentials. Never put them in `config.yaml`.

---

## `common.py`

```python
def retry(max_tries: int = 6, base_delay_s: float = 10.0, max_delay_s: float = 600.0,
          exceptions: tuple = (Exception,))
    # decorator: exponential backoff with jitter; logs each failure with attempt number
def file_lock(path: Path) -> ContextManager
    # cross-process lock using os.open(path.with_suffix('.lock'), O_CREAT|O_EXCL); wait+retry up to 10 min
def update_manifest(paths: Paths, rel_path: str, record: dict) -> None   # under file_lock
def read_manifest(paths: Paths) -> dict
def sha256_file(path: Path, chunk: int = 1 << 20) -> str
def download_bbox(cfg) -> tuple[float, float, float, float]
    # (lon_min - buf, lat_min - buf, lon_max + buf, lat_max + buf) = (79, 4, 101, 24) by default
def crop_to_bbox(ds: xr.Dataset, bbox) -> xr.Dataset
    # after standardize_coords; inclusive selection with .sel(lat=slice, lon=slice); asserts non-empty
def concat_daily(files: list[Path], out: Path, product: str, stream: str) -> None
    # opens each cropped daily file, concatenates on time, sorts, drops duplicate times (keep last),
    # writes atomically with io.write_netcdf, then deletes inputs
def stream_date_ranges(cfg, product: str, probe: dict) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]
    # from results/probe/probe.json: for each stream (config order) the [start, end] it will be used for,
    # clipped to config dates; later streams only cover dates the earlier streams lack
def years_needed(start, end) -> list[int]
def check_credentials(product: str) -> None   # raises with a setup hint
```

Downloaders read `results/probe/probe.json`. If it is missing they exit with
"run `python -m src.download.probe` first".

---

## `probe.py` (run first, then **STOP**)

Goal: replace every `# VERIFY` in `config.yaml` with evidence. Downloads at most a few days per
stream to `results/probe/raw/` (git-ignored).

For each product in `config.products` and each stream:

1. **Metadata.**
   - copernicusmarine: `copernicusmarine.describe(dataset_id=...)` (or `--contains`); record the
     dataset version(s), time coverage start/end, variables + units, spatial resolution, depth
     levels. The return type differs between toolbox versions — wrap access in one helper and
     check the installed version's docs/`help()` at implementation time.
   - earthaccess: `earthaccess.search_datasets(short_name=...)` for collection temporal extent;
     `earthaccess.search_data(short_name=..., temporal=(first_day, first_day+2))` for granules.
   - CCMP: HTTP `HEAD` on the URL for `dates.start`, `dates.end`, and the first of every month
     in between; record which months exist.
   - argo: fetch 7 days in the box with `argopy` (`mode="expert"`); record variable names.
2. **Sample.** Download one day (one granule) at `dates.start` and one at `dates.end` (or the
   stream's last date if earlier). Open each with xarray and record: variable names, `units`
   attributes, dimension names and order per variable, coordinate names, lat/lon spacing and first
   values, lon convention (0–360 or ±180), lat order, time values (and their hour), depth values
   (GLORYS: the full list ≤ 1100 m), `_FillValue`, file size.
3. **Coverage plan.** Using the metadata, compute for each product which stream covers which date
   range of `dates` (first stream wins), any dates covered by no stream, and the **stream switch
   dates**.
4. **Extension check.** Compute the latest date covered by *every* product (through any stream).
   Report the period that would be possible if the study were extended (e.g. 2016–2024).

Outputs:

- `results/probe/probe.json` — machine-readable, everything above.
- `results/probe/probe_report.md` — human-readable tables:
  1. product × stream: dataset id, version, coverage, variables (units), grid, dims order;
  2. coverage plan with switch dates and the split each switch falls in (train/val/test);
  3. a list of every `# VERIFY` line in `config.yaml` with the observed value, marked
     `OK` (matches) or `CHANGE → <value>`;
  4. size estimates per product for the full period (sample size × days);
  5. extension option.

**STOP**: show the user `probe_report.md`. Agree (a) the corrected `config.yaml` values,
(b) whether to keep 2016–2022 or extend, (c) what to do if a stream switch falls inside the test
year (options: accept and document; move the period; or use only the interim stream for all
years if it spans them). Record the outcome in `docs/decisions_log.md`.

If OSCAR (or any product) does not cover the test year in any stream: report it; candidate
fixes are shortening the period or replacing the channel's source (e.g. Copernicus Marine
multi-observation surface currents). This is a **Decision**-level change → **STOP**.

---

## `copernicus.py` — GLORYS, OSTIA, DUACS

```python
def cm_subset(dataset_id: str, variables: list[str], bbox, t0: str, t1: str,
              out_file: Path, depth_range: tuple[float, float] | None = None) -> Path
    # single wrapper around copernicusmarine.subset(...); keep ALL toolbox-version-specific argument
    # names inside this function. Writes to <out_file>.part then renames.
def download_product(cfg, paths, product: str, years: list[int] | None, force: bool) -> None
def main() -> None   # CLI: --products glorys ostia duacs  --years 2016 2017 ...
```

Behaviour:

- For each stream (from `stream_date_ranges`), for each year intersecting that stream's range:
  target file `paths.raw_year_file(product, stream, year)`.
- Skip if the target is in the manifest with matching `time_start/time_end` and `file_is_complete`.
- **GLORYS: request one month at a time** into `raw/_tmp/glorys_<stream>_<YYYY>_<MM>.nc`
  (a year is several GB; monthly requests make retries cheap), then `concat_daily` into the year
  file. Depth range from `config.products.glorys.depth_range_m`.
- OSTIA and DUACS: one request per year.
- After writing: verify the number of daily time steps equals the expected days in the range;
  verify the bbox; update the manifest (record from `data/raw/IMPLEMENTATION.md`).
- Each request is wrapped in `@retry`.

## `earthdata.py` — OSCAR, SMAP

```python
def download_product(cfg, paths, product: str, years: list[int] | None, force: bool) -> None
def main() -> None
```

Behaviour per stream and year:

1. `earthaccess.search_data(short_name=..., temporal=(Y-01-01, Y-12-31), bounding_box=(lon0, lat0, lon1, lat1))`.
2. Download granules in batches of ~30 into `raw/_tmp/<product>/` (`earthaccess.download`).
3. Open each granule, `standardize_coords`, `crop_to_bbox`, keep only `config.products.<p>.variables`
   (plus their coordinates), write a cropped daily file, **delete the global granule immediately**.
4. Date of each granule is read from its `time` coordinate (floored to date), not the filename.
   Duplicate dates (re-processed granules) keep the one with the latest `date_created`/`history`
   attribute, else the latest file name.
5. `concat_daily` → `paths.raw_year_file(product, stream, year)`; update manifest.

OSCAR may store variables as `(time, longitude, latitude)` — after `standardize_coords`,
always `transpose("time", "lat", "lon")` by name.

## `ccmp.py` — CCMP winds

```python
def url_for(cfg, stream: dict, date: pd.Timestamp) -> str   # format url_template with yyyy, mm, dd
def download_day(url: str, tmp: Path) -> Path               # requests streaming, timeout=(30, 600), @retry
def download_product(cfg, paths, years, force, workers: int = 4) -> None
def main() -> None
```

- Each daily file is global (~50 MB, 4 times). Download → crop → keep `uwnd`, `vwnd` (and `nobs`
  if present) → write cropped daily → delete global. Use a `ThreadPoolExecutor(workers)` for
  downloads; cropping/writing in the main thread.
- Then `concat_daily` per year. Keep all 4 times per day (daily mean happens in preprocessing).
- 404 for a date → record as missing (do not crash); `check_coverage` will report it.

## `argo_fetch.py` — ARGO profiles

```python
def fetch_year(cfg, paths, year: int, force: bool) -> Path
def main() -> None
```

- `from argopy import DataFetcher`; `set_options(src=cfg.products.argo.argopy_src, mode="expert")`.
- Region `[lon_min, lon_max, lat_min, lat_max, 0, max_depth_m, start, end]` (the box itself, no
  buffer), **one request per month** (avoids server timeouts), `@retry`; on repeated failure
  switch `src` to `gdac` and log it.
- Keep the point collection with at least: `PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION, TIME,
  LATITUDE, LONGITUDE, PRES, TEMP, PSAL, PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED,
  PRES_QC, TEMP_QC, PSAL_QC, PRES_ADJUSTED_QC, TEMP_ADJUSTED_QC, PSAL_ADJUSTED_QC, DATA_MODE,
  POSITION_QC, TIME_QC`. Variable names differ between argopy versions/sources — map them in one
  dictionary at the top of the file after checking the probe output.
- Concatenate months, write `data/raw/argo/argo_<YYYY>.nc`, update manifest.
- No QC filtering here; it is done in `src/evaluation/argo.py`.

## `run.py`

`python -m src.download.run --config config.yaml --products all|glorys,ostia,... [--years ...]`
Runs downloaders in the order: duacs, ostia, oscar, smap, ccmp, glorys, argo (small first, so
problems surface early). Continues to the next product if one fails, and exits non-zero at the end
with a summary if any failed.

## `check_coverage.py`

For each product: open all raw year files (all streams), floor times to dates, and compare with
`daily_index(cfg)` (for CCMP: a date is covered only if all 4 times exist; for ARGO: skip).
Print and write `results/probe/coverage.md` + `coverage.csv` (`product, date, stream`) with:

- missing dates (ranges compressed: `2019-06-20 … 2019-07-22 (33 days)`),
- the stream used for each date range,
- stream switch dates and the split each falls in.

Exit code 1 if any non-SMAP product has a missing date. SMAP gaps are allowed (they are filled
in preprocessing) but are listed.

---

## Tests (`tests/test_download.py`, no network)

- `stream_date_ranges`: two streams with overlapping coverage → first wins, no gaps, no overlap.
- `crop_to_bbox` on a synthetic 0–360 global dataset → correct bbox, ascending lat.
- `concat_daily` with duplicate and unsorted days → sorted unique, inputs deleted.
- `retry`: function failing twice then succeeding is called 3 times (patch `time.sleep`).
- `update_manifest` from two threads → both records present.
- `url_for` formats zero-padded months/days.
- `check_coverage` core function on synthetic year files with a 3-day hole → reports it.

## Done when

- Tests pass on the authoring machine.
- On Lightning: `probe_report.md` reviewed with the user; `check_coverage` exits 0;
  `manifest.json` has one record per year file.
