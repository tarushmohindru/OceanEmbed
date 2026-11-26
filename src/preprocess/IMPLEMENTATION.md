# `src/preprocess/` — Harmonization: raw → interim → processed

Purpose: turn the raw downloads into the model-ready files in `data/processed/` (contract:
`data/processed/IMPLEMENTATION.md`; intermediate contract: `data/interim/IMPLEMENTATION.md`).
The deck flags this phase as the largest project risk; most bugs here are silent, so every step
has assertions and every rule below is tested on synthetic data.

Runs on: Lightning CPU machine (~16 GB RAM is enough). Unit tests run on the authoring machine.
No `torch` imports in this package.

## Files

```text
src/preprocess/
  __init__.py
  make_grid.py        # ENTRY  step 1: grid.nc
  readers.py          # per-product readers: raw year files -> daily, standardized, QC'd DataArrays
  regrid_core.py      # pure-numpy separable regridding for regular lat-lon grids (PRIMARY)
  regrid.py           # ENTRY  step 2: raw -> interim channel files + interim GLORYS (native levels)
  qc.py               # range checks
  vertical.py         # ENTRY  step 3: GLORYS native levels -> 15 target depths
  masks.py            # ENTRY  step 4: masks.nc
  climatology.py      # ENTRY  step 5: clim_inputs.nc, climatology_Y.nc (+ reusable functions)
  fill.py             # ENTRY  step 6: gap filling -> interim/filled/X_raw_<YYYY>.nc, fill_stats.csv
  normalize.py        # ENTRY  step 7: norm_stats.json
  build_tensors.py    # ENTRY  step 8: X.zarr, Y.zarr, Y_degC.zarr, dates.csv, provenance.csv
  check_interim.py    # ENTRY  validation of interim/
  sanity.py           # ENTRY  step 9: sanity checks + figures
  run.py              # ENTRY  runs steps 1–9 (or --steps subset) in order
```

Every ENTRY uses `src.common.cli.parse_and_load` and is idempotent (`--force` to redo).

---

## Step 1 — `make_grid.py`

`grid_from_config(cfg).to_dataset()` → write `processed/grid.nc` and `interim/grid.nc`.
Assert H, W are integers from the region/resolution. Log H, W, first/last centres.

## `readers.py`

```python
@dataclass
class DailyField:
    data: xr.Dataset            # dims (time, lat, lon) or (time, depth, lat, lon); standardized coords
    provenance: pd.DataFrame    # columns date, product, stream, dataset_id

def read_product_year(cfg, paths, product: str, year: int) -> DailyField
```

Rules, in this order, for each product and year:

1. Open the raw year file of every stream that exists for this year.
2. `standardize_coords` (rename to lat/lon, lon to ±180, sort ascending).
3. **Date labels:** `time` floored to dates (`timeutil.floor_to_date`).
4. **Stream selection per date:** for each date in the year ∩ `dates`, take the first stream in
   config order that has a finite field on that date. Record (date, stream, dataset_id).
5. **Daily mean:** CCMP has 4 times per date → mean over them; if fewer than 4 are present the
   date is NaN (logged). Other products already have one value per date; assert it.
6. **Units:** OSTIA `analysed_sst`: if `units` in {"K", "kelvin", "Kelvin"} subtract 273.15
   and set `units="degC"`. Assert expected units for every variable (table from the probe;
   hold the expected-units table in one dict at the top of `readers.py`).
7. **Rename** product variables to channel names using `config.channels`
   (e.g. `analysed_sst → sst`, `uwnd → u_wind`). GLORYS keeps `thetao`.
8. **QC:** `qc.apply_ranges(ds, cfg)` for input channels (before regridding, so outliers are
   not averaged into neighbours).
9. Transpose by name to `(time, lat, lon)` / `(time, depth, lat, lon)`; cast to float32.
10. Ensure the time axis has every date of the year ∩ `dates`; reindex with NaN for missing
    dates (SMAP gaps) — never drop dates.

GLORYS is large: `read_product_year` for GLORYS returns a lazily loaded (dask) dataset with
`chunks={"time": 31}`; `regrid.py` processes it month by month.

## `regrid_core.py` (primary regridder — pure numpy)

All source products are on **regular rectilinear lat-lon grids**, so conservative remapping is
separable into a latitude matrix and a longitude matrix. This avoids `xesmf` in the main path
(it remains in the environment only for a cross-check test on Lightning).

```python
def overlap_matrix_lon(src_edges: np.ndarray, tgt_edges: np.ndarray) -> np.ndarray
    # (W_t, W_s): length of overlap in degrees between target cell j and source cell l
def overlap_matrix_lat(src_edges: np.ndarray, tgt_edges: np.ndarray) -> np.ndarray
    # (H_t, H_s): overlap in sin(latitude) units  -> proportional to spherical area
def conservative(field: np.ndarray, A_lat, A_lon, na_thres: float) -> np.ndarray
    # field (..., H_s, W_s), NaN = missing. Let m = isfinite(field), v = nan_to_num(field).
    #   num   = A_lat @ (v*m) @ A_lon.T
    #   den   = A_lat @ m     @ A_lon.T
    #   cover = A_lat @ 1     @ A_lon.T          (source coverage of each target cell)
    #   out   = num / den  where den / cover >= na_thres, else NaN
    # Use np.matmul with broadcasting over leading dims; compute in float64, return float32.
def bilinear_matrix(src_centres, tgt_centres) -> np.ndarray     # (N_t, N_s), 2 non-zeros per row
def bilinear(field, B_lat, B_lon, na_thres: float) -> np.ndarray  # same NaN-normalization trick
def choose_method(src_lat, src_lon, grid: Grid) -> str
    # "select"       if source centres include all target centres (|diff| < 1e-6) -> exact .sel
    # "conservative" if source spacing <= target spacing (+1e-9)  (OSTIA, GLORYS, DUACS 0.125°,
    #                 and 0.25° grids offset from the target centres, e.g. OSCAR if centred on x.00)
    # "bilinear"     if the source is coarser than the target (not expected; log a warning)
class Regridder:
    def __init__(self, src_lat, src_lon, grid: Grid, na_thres: float, method: str | None = None)
    def __call__(self, field: np.ndarray) -> np.ndarray        # (..., H_s, W_s) -> (..., H, W)
    def save(self, path: Path) / @classmethod load(cls, path)   # cache matrices as .npz
```

The source must fully cover the target box (assert edges); this is why raw data has a 1° buffer.
Source edges come from `grid.regular_bounds(src_centres)`.

## Step 2 — `regrid.py`

```python
def regrid_inputs_year(cfg, paths, year: int, force: bool) -> None
def regrid_glorys_year(cfg, paths, year: int, force: bool) -> None
def main() -> None   # --products, --years
```

- Inputs: for each channel group (`sst`, `sss`, `sla`, `cur`, `wind`) read the product year,
  build/load the cached `Regridder` (`paths.weights_file(product, method)`), regrid every
  channel, write `interim/<group>/<group>_<YYYY>.nc` per the interim contract, and
  `interim/provenance/<product>_<YYYY>.csv`.
- GLORYS: per month, per depth level, regrid `thetao` → assemble `(time, depth, lat, lon)` →
  write `interim/glorys/glorys_<YYYY>.nc`. Note that conservative regridding with `na_thres=0.5`
  makes coastal and sea-floor cells valid only if at least half of the target cell is ocean at
  that level — this is the intended behaviour.
- After each file: assert coordinates equal the target grid, dtype float32, expected date count.
- Log, per channel and year: fraction of NaN over the ocean (approximate: cells with any valid
  value that year).

## `qc.py`

```python
def apply_ranges(ds: xr.Dataset, cfg) -> tuple[xr.Dataset, dict[str, int]]
    # for each channel present in ds: values outside preprocess.qc_ranges[channel] (inclusive bounds)
    # -> NaN. Returns counts of rejected values per channel (logged by caller).
```

## Step 3 — `vertical.py`

```python
def interp_to_targets(T: np.ndarray, z_src: np.ndarray, z_tgt: np.ndarray,
                      top_level_max_m: float) -> np.ndarray
    # T (..., K, H, W) on native depths z_src (ascending, metres); returns (..., D, H, W)
```

Rules:

1. Target depth 0: if `z_src[0] <= top_level_max_m` (GLORYS: 0.494 m), copy level 0 unchanged.
   Otherwise raise (never extrapolate to the surface).
2. Target depth d > 0: find k with `z_src[k] <= d <= z_src[k+1]`; linear interpolation in depth
   between levels k and k+1. If `d` equals a native level, use it directly.
3. If either bracketing value is NaN → NaN (no extrapolation below the sea floor, no partial
   columns).
4. If `d > z_src[-1]` → raise (download depth range too shallow).

Vectorize over (time, H, W) with `np.searchsorted` on `z_src` (computed once) — no Python loop
over cells. Write `interim/glorys_t15/glorys_t15_<YYYY>.nc`.

## Step 4 — `masks.py`

Implements the mask rules in `data/processed/IMPLEMENTATION.md` exactly. Inputs: all
`glorys_t15` years and all `sst` interim years (all years may be used for masks: masks are
geometry, not statistics). Output `processed/masks.nc`. Log: ocean cell count, cells removed by
the largest-component rule (with their lat/lon range), `depth_mask` count per depth.

## Step 5 — `climatology.py`

```python
def doy_climatology(values: np.ndarray, dates: pd.DatetimeIndex, use: np.ndarray,
                    window_days: int) -> np.ndarray
    # values (T, ...), use (T,) bool (training days). Returns (366, ...).
    # 1. per slot s = doy_index(date): S[s] = nansum over used days, N[s] = count of finite used values
    # 2. circular moving sums of S and N with a centred window of window_days (odd) over the 366 slots
    # 3. clim = S_smooth / N_smooth; NaN where N_smooth == 0
def build_clim_inputs(cfg, paths) -> None   # interim channel files (after QC, before fill), train years
def build_clim_y(cfg, paths) -> None        # glorys_t15 (°C), train years, masked by depth_mask
```

Training days = `timeutil.split_of(dates) == "train"`. Assert that no val/test day is used.

## Step 6 — `fill.py` (inputs only; targets are never filled)

Load all years of all 7 channels into one array `(T, C, H, W)` (~0.4 GB float32 for the
default box) and apply, per channel and ocean cell:

1. **Time interpolation:** interior gaps of length ≤ `max_interp_gap_days` (bounded by finite
   values on both sides) → linear interpolation in time. Longer gaps untouched by this step.
2. **Climatology:** remaining NaN at ocean cells → `clim_inputs[doy_index(date)]`.
3. **Spatial nearest:** remaining NaN at ocean cells (typically coastal cells where SMAP never
   has data) → value of the nearest finite ocean cell on the same day
   (`scipy.ndimage.distance_transform_edt(~valid, return_indices=True)`).
4. **Outside ocean:** set to NaN (land handled at tensor-building time).
5. Assert: no NaN inside `ocean` for any (day, channel).

Record per (date, channel) the fraction of ocean cells filled by each step (and rejected by QC,
from the regrid logs or by recounting) → `processed/fill_stats.csv`. Write
`interim/filled/X_raw_<YYYY>.nc`. Log the overall fill fraction per channel per split; if any
channel has > 20% filled in any split, log a WARNING.

## Step 7 — `normalize.py`

Statistics exactly as in the `norm_stats.json` schema (`data/processed/IMPLEMENTATION.md`):
inputs over `ocean` × training days from `filled/X_raw`; targets per depth over
`depth_mask[d]` × training days from `glorys_t15`. Use float64 accumulation. Assert std > 0.
Write `processed/norm_stats.json` including `years_used` from the config's training years.

## Step 8 — `build_tensors.py`

1. `X = normalize_x(X_raw)`; then `X[:, :, ~ocean] = 0`; assert no NaN.
2. `Y_degC` from `glorys_t15`, with `Y_degC[:, d, ~depth_mask[d]] = NaN`.
3. `Y = normalize_y(Y_degC)` (NaN preserved).
4. Write `X.zarr`, `Y.zarr`, `Y_degC.zarr` (chunks one day), `dates.csv`
   (`t_index, date, year, doy_index, split`) and `provenance.csv` (concatenate all interim
   provenance CSVs).
5. Write `_meta.json` in `processed/`.
6. Optional extras are **not** stored here; `src/dataset` builds them on the fly (land mask,
   day-of-year sin/cos, lat/lon) so the core tensors never change between ablations.

## `check_interim.py` and Step 9 `sanity.py`

`check_interim.py` validates `data/interim/` against its contract. `sanity.py` implements the
checklist in `data/processed/IMPLEMENTATION.md`, prints PASS/FAIL/WARNING per item, writes figures
to `results/figures/sanity/`, exits 1 on any FAIL.

## `run.py`

`python -m src.preprocess.run --config config.yaml [--steps grid,regrid,vertical,masks,clim,fill,normalize,tensors,sanity] [--years ...] [--force]`

---

## Tests (`tests/test_preprocess_*.py`; synthetic data only)

- `regrid_core`:
  - constant field → same constant everywhere valid;
  - conservation: area-weighted mean of a random field over the box is preserved within 1e-6
    when the source exactly tiles the target (e.g. 1/12° → 0.25°);
  - a 0.05° field equal to latitude → target ≈ target latitude (within 1e-3);
  - NaN handling: source cell block 60% NaN → target NaN with `na_thres=0.5`; 40% NaN → finite
    and equal to the mean of the valid part;
  - `choose_method` returns `select` / `conservative` / `bilinear` for crafted grids.
  - (marked `data`, Lightning only) compare against `xesmf` `conservative_normed` on a real OSTIA
    day: max abs difference < 0.01 °C over cells valid in both.
- `vertical`: linear profile T = 30 − 0.02 z on GLORYS-like levels → exact at targets; NaN below
  the floor; depth 0 copies the top level; target deeper than the deepest level raises.
- `qc`: boundary values kept, outside → NaN, counts correct.
- `climatology`: synthetic `sin(2π doy/365)` over 5 years → recovered within smoothing error;
  slot 59 defined; val/test days excluded (assert by injecting huge values into val days).
- `fill`: 2-day gap → linear; 10-day gap → climatology; coastal always-NaN cell → nearest
  neighbour value; land stays NaN; fill fractions sum correctly.
- `masks`: synthetic GLORYS with a shelf → `depth_mask` nested and shrinking; two disconnected
  ocean blobs → only the larger kept.
- `normalize` + `build_tensors`: on a 3-year synthetic dataset, train-only stats (inject an
  outlier year in test → stats unchanged), no NaN in X, Y NaN exactly outside masks.

## Done when

- `pytest tests -k preprocess` passes on the authoring machine.
- On Lightning: `python -m src.preprocess.run --config config.yaml` completes and `sanity`
  prints no FAIL.
