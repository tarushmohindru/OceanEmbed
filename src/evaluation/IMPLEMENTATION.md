# `src/evaluation/` — Metrics, GLORYS test scoring, ARGO validation, figures

Purpose: score every prediction store the same way, against (1) withheld GLORYS (dense) and
(2) real ARGO profiles (sparse), and produce every table and figure for the report.

Reads only: prediction stores (`results/predictions/<model_id>/<split>.zarr`),
`data/processed/` (`Y_degC.zarr`, `masks.nc`, `climatology_Y.nc`, `dates.csv`, `grid.nc`),
`data/raw/argo/`, and training run folders (for training curves). Never imports `models` or
`training`.

Runs on: Lightning CPU machine (minutes). `cartopy` and `gsw` are imported lazily.

## Files

```text
src/evaluation/
  __init__.py
  metrics.py        # pure numpy metric functions (no IO)
  bootstrap.py      # moving-block bootstrap over days
  evaluate.py       # ENTRY: GLORYS-based evaluation of model stores
  argo.py           # ENTRY: ARGO processing, matching and scoring
  tables.py         # ENTRY: assemble CSV + Markdown tables for the report
  plots.py          # ENTRY: all report figures
  style.py          # model colours/labels/line styles shared by every figure
```

---

## Metric definitions (`metrics.py`) — the single source of truth

Notation per depth d: `V_d` = all (day, cell) pairs of the split with `depth_mask[d] = 1` and
finite truth. `p` = prediction (°C), `t` = truth (°C), `c` = training climatology (°C) for the
day's `doy_index`.

| Metric | Definition | Notes |
| --- | --- | --- |
| RMSE_d | `sqrt(mean_{V_d} (p − t)²)` | pooled over days and cells |
| Bias_d | `mean_{V_d} (p − t)` | positive = model too warm |
| Corr_d | Pearson r of p and t over V_d | inflated by the static pattern — report, but rely on ACC |
| ACC_d | Pearson r of (p − c) and (t − c) over V_d | NaN for the climatology model (p − c ≡ 0) |
| Skill_d | `1 − RMSE_d / RMSE_d(climatology)` | > 0 beats climatology |
| Overall | mean of the per-depth values over the 15 depths | Skill overall = `1 − RMSE_overall/RMSE_overall(clim)` |
| Band | mean of the per-depth values over the band's depths (`config.depth_bands_m`) | |

Cells are not area-weighted (cell area varies by < 8 % across 5–23°N); state this in the
write-up. If a prediction is NaN anywhere in V_d, raise — stores must be complete inside the mask.

```python
def per_depth_sums(p, t, c, mask) -> dict[str, np.ndarray]
    # p, t, c: (N, D, H, W); mask: (D, H, W) bool or (N, D, H, W).
    # returns per-DAY, per-depth sufficient statistics, shape (N, D) each:
    #   n, se (sum sq err), e (sum err), sp, st, spp, stt, spt (for Pearson),
    #   sa, sb, saa, sbb, sab (anomalies p-c, t-c for ACC)
def metrics_from_sums(sums: dict, day_index: np.ndarray | None = None) -> pd.DataFrame
    # aggregates the (selected) days and returns one row per depth:
    # depth, n, rmse, bias, corr, acc   (float64 throughout)
def add_skill(df_model: pd.DataFrame, df_clim: pd.DataFrame) -> pd.DataFrame
def summarize(df: pd.DataFrame, bands: dict) -> pd.DataFrame
    # rows: overall + each band; columns rmse, bias, corr, acc, skill
def rmse_map(p, t, mask_d) -> np.ndarray   # (H, W) RMSE over days for one depth
def rmse_per_depth(p, t, mask) -> np.ndarray  # convenience wrapper (used by the training-metric test)
```

Working from per-day sufficient statistics makes seasons (select days), bootstrap (resample days)
and the overall numbers use exactly the same code.

## `bootstrap.py`

```python
def block_bootstrap_days(n_days: int, block: int, n_boot: int, seed: int) -> np.ndarray
    # (n_boot, n_days) day indices built from contiguous blocks with random starts (circular),
    # truncated to n_days. Blocks keep the strong day-to-day autocorrelation.
def bootstrap_rmse(sums: dict, idx: np.ndarray) -> np.ndarray          # (n_boot, D) and overall
def paired_difference_ci(sums_a, sums_b, idx, ci=0.95) -> pd.DataFrame
    # RMSE_a − RMSE_b per depth and overall, using the SAME resampled days for both models
```

Defaults from `config.eval.bootstrap` (1000 resamples, 30-day blocks, 95 % percentile CI).

## `evaluate.py` (ENTRY)

```bash
python -m src.evaluation.evaluate --config config.yaml --split test \
    [--models climatology linreg cnn_unet vit_cnn ...]   # default: every store found for the split, except "glorys"
```

1. Load truth `Y_degC` for the split's days, `depth_mask`, climatology `c` per day.
2. For each model: read the store; assert dates, depths and grid equal the truth's; compute
   per-day sums.
3. Write to `results/metrics/<split>/`:
   - `per_depth.csv` — model, depth, n, rmse, bias, corr, acc, skill
   - `summary.csv` — model, scope (overall / band name), rmse, bias, corr, acc, skill,
     rmse_ci_low, rmse_ci_high
   - `per_season.csv` — model, season, depth (+ "overall"), rmse, bias, acc
   - `paired_vit_vs_cnn.csv` — if both exist: per-depth and overall RMSE difference with CI
     (also `vit_cnn_ens` vs `cnn_unet` if the ensemble exists)
   - `rmse_maps.nc` — `rmse(model, depth, lat, lon)` for `eval.map_depths_m`, and skill maps
   - `daily_rmse.csv` — model, date, overall RMSE that day (used to choose the sample day)
4. Log a compact table to the console.

## `argo.py` (ENTRY)

```bash
python -m src.evaluation.argo --config config.yaml --split test \
    [--models climatology linreg cnn_unet vit_cnn glorys]
```

Step A — **profile processing** (cache to `results/metrics/argo/argo_profiles_<split>.nc`):

1. Load `data/raw/argo/argo_<YYYY>.nc` for the split's years; convert points to profiles
   (argopy `point2profile` or group by `PLATFORM_NUMBER, CYCLE_NUMBER, DIRECTION`).
2. Profile filters: `POSITION_QC` and `TIME_QC` ∈ `eval.argo.qc_ok`; position inside the box;
   date (UTC, floored) is one of the split's evaluation days (gap days excluded);
   `DIRECTION == "A"` (ascending); at least 5 good levels.
3. Variable choice per profile: `DATA_MODE ∈ {D, A}` → `*_ADJUSTED` values and `*_ADJUSTED_QC`;
   `R` → raw values and raw QC. Record the mode per profile.
4. Level filters: PRES, TEMP QC ∈ `qc_ok` and finite. PSAL QC ∈ `qc_ok` → use it; otherwise
   interpolate PSAL from neighbouring good levels; if none, use 35.0 and count it (the potential
   temperature correction depends only weakly on salinity).
5. Conversions with `gsw`: depth `z = −gsw.z_from_p(p, lat)`; absolute salinity
   `SA = gsw.SA_from_SP(SP, p, lon, lat)`; **potential temperature referenced to 0 dbar**
   `θ = gsw.pt0_from_t(SA, t, p)` (GLORYS `thetao` is potential temperature).
6. Sort by depth; drop duplicated depths (keep first); require strictly increasing depth.
7. Interpolate to `config.depths_m`:
   - 0 m: the shallowest good value if its depth ≤ `eval.argo.surface_max_depth_m`, else NaN;
   - d > 0: exact level within 1 m → use it; else bracketing levels `z_a ≤ d ≤ z_b` with
     `z_b − z_a ≤ max_bracket_gap_m(d)` (first rule whose `upto_m ≥ d`) → linear in depth; else NaN.
8. Store per profile: id (`<platform>_<cycle>`), date, lat, lon, data mode, θ at 15 depths, and
   a flag per depth explaining NaN (no data / gap too large / below profile).

Step B — **matching**: grid cell `(i, j) = grid.index_of(lat, lon)`; drop if outside the box or
not `ocean`; for each depth keep the value only if `depth_mask[d, i, j]`. Take each model's
prediction at (date, d, i, j). Multiple profiles in one cell-day are all kept.

Step C — **scoring**, over the **common set** (profile-depth pairs valid for every model in the
run, including `glorys`): per depth n, RMSE, bias, Pearson r; overall = mean over depths; also
bands. Report the number of profiles, floats, and matched values per depth.

Outputs in `results/metrics/argo/`: `argo_matches_<split>.csv` (one row per profile-depth with
obs and every model's prediction), `argo_per_depth_<split>.csv`, `argo_summary_<split>.csv`,
`argo_counts_<split>.csv`.

Interpretation notes to carry into the write-up (do not compute anything else from them):
- ARGO-vs-model error includes representativeness error (a point vs a 0.25° daily mean).
- GLORYS assimilates Argo, so GLORYS-vs-ARGO is an optimistic reference, not an independent bound.

## `tables.py` (ENTRY)

Builds `results/tables/` from the metric CSVs:

- `main_results_test.md/.csv` — rows: climatology, linreg, cnn_unet, vit_cnn (+ vit_cnn_ens if
  present); column groups: overall, 0–100, 100–300, 300–1000 m; each with RMSE (95 % CI), bias,
  ACC, skill. Same for val (`main_results_val.md`).
- `argo_results_test.md` — models + glorys: n, RMSE, bias, r per band and overall.
- `season_results_test.md` — overall RMSE per season × model.
- `ablations_test.md` — one row per ablation run vs the selected ViT-CNN (Δ RMSE with CI).
- Numbers with 3 decimals for °C, 2 for correlations/skill. Never edit these files by hand.

## `plots.py` (ENTRY) and `style.py`

`style.py`: one dict `MODEL_STYLE[model_id] = {"label", "color", "linestyle", "marker"}`
(colour-blind-safe palette, e.g. Okabe–Ito; climatology grey dashed; GLORYS black). Every figure
uses it. Figures: PNG, 200 dpi, `results/figures/`, sized for an A4 page or a 16:9 slide, axis
labels with units, depth axes inverted (0 m at the top).

| File | Content |
| --- | --- |
| `rmse_vs_depth_test.png` | RMSE (°C, x) vs depth (m, y, inverted), one line per model; shaded 95 % CI for cnn_unet and vit_cnn |
| `bias_vs_depth_test.png` | same layout for bias, vertical zero line |
| `maps_sample_day.png` | 3 rows (depths 0, 100, 500 m) × 4 columns: GLORYS, ViT-CNN, ViT-CNN − GLORYS, CNN − GLORYS. Shared colour limits per row for the two temperature panels; symmetric diverging limits for differences. Sample day = test day with the **median** ViT-CNN daily RMSE (from `daily_rmse.csv`; no cherry-picking); date in the title. Coastlines with cartopy if available, else the ocean-mask contour |
| `profiles_sample_day.png` | 4 panels (points: 15°N 88°E, 10°N 95°E, 20°N 89°E, 7°N 85°E; nearest ocean cell), temperature vs depth for GLORYS and every model on the sample day |
| `rmse_maps_test.png` | RMSE maps for vit_cnn at 0/100/500 m (row 1) and skill vs climatology (row 2, diverging, centred at 0) |
| `rmse_by_season_test.png` | grouped bars: season × model, overall RMSE |
| `argo_scatter_test.png` | two panels: vit_cnn vs ARGO and glorys vs ARGO; points coloured by depth (sequential colormap + colorbar), 1:1 line, n and RMSE in the title |
| `argo_rmse_vs_depth_test.png` | ARGO RMSE vs depth for every model and glorys |
| `training_curves.png` | train/val loss and val RMSE vs epoch for the selected cnn_unet and vit_cnn runs |
| `lambda_sweep.png` | val RMSE vs λ for both models |

## Tests (`tests/test_evaluation.py`, synthetic)

- Perfect prediction → RMSE 0, bias 0, corr 1; constant +0.5 °C offset → RMSE 0.5, bias 0.5.
- `metrics_from_sums` equals a direct numpy computation on random data with random masks (1e-10).
- ACC of climatology → NaN; Skill of climatology → 0.
- Season split: days partition exactly; sums over seasons recompose the overall sums.
- Bootstrap: indices in range, block structure, CI contains the point estimate for a smooth case;
  paired CI of a model with itself is exactly 0.
- ARGO: a synthetic profile with known θ(z) is interpolated exactly at targets; bracket-gap rule
  produces NaN where gaps are too large; surface rule; `pt0_from_t` conversion applied
  (`importorskip("gsw")`); matching drops land and off-mask depths.
- Store mismatch (wrong dates or grid) → clear error.

## Done when

- Tests pass on the authoring machine.
- On Lightning: `evaluate` (val and test), `argo`, `tables` and `plots` produce every file listed
  above for all models.
