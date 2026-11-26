# `data/processed/` — Model-ready data contract

Written by `src/preprocess/` (`make_grid.py`, `masks.py`, `climatology.py`, `normalize.py`,
`build_tensors.py`). Read by `src/dataset`, `src/baselines`, `src/training`, `src/evaluation`.

This is the most important contract in the project: every model and baseline reads only these
files. Shapes below are for the default Bay of Bengal box (H = 72, W = 80, T = 2557 days,
C = 7 inputs, D = 15 depths).

## Files

| File | Content | Shape / schema |
| --- | --- | --- |
| `grid.nc` | target grid | coords `lat` (H), `lon` (W); vars `lat_bnds` (H,2), `lon_bnds` (W,2), `cell_area_km2` (H,W) |
| `masks.nc` | static masks | `ocean` (H,W) uint8; `depth_mask` (D,H,W) uint8 with `depth` coord; attrs: rule used |
| `X.zarr` | normalized inputs | var `X` (T, C, H, W) float32; coords `time`, `channel`, `lat`, `lon`; chunks (1, C, H, W). **No NaN.** Land = 0 |
| `Y.zarr` | normalized targets | var `Y` (T, D, H, W) float32; coords `time`, `depth`, `lat`, `lon`; chunks (1, D, H, W). NaN outside `depth_mask` |
| `Y_degC.zarr` | targets in °C (for metrics and ARGO reference) | same as `Y`, physical units |
| `climatology_Y.nc` | target climatology, °C | var `clim` (366, D, H, W); coord `doy` 0…365; train years only, 31-day circular smoothing |
| `norm_stats.json` | normalization statistics | see schema below |
| `provenance.csv` | which stream provided each input/target on each date | columns `date, product, stream, dataset_id` |
| `fill_stats.csv` | gap-fill fractions | columns `date, channel, frac_qc_rejected, frac_time_interp, frac_clim, frac_spatial` (fractions of ocean cells) |
| `dates.csv` | the time index | columns `t_index, date, year, doy_index, split` (`split` ∈ train/val/test/gap) |
| `_meta.json` | provenance of this directory | root spec §5.7 |

### `norm_stats.json` schema

```json
{
  "years_used": [2016, 2017, 2018, 2019, 2020],
  "inputs": {"sst": {"mean": 0.0, "std": 0.0}, "...": {}},
  "targets": {"0": {"mean": 0.0, "std": 0.0}, "10": {}, "...": {}},
  "n_cells_per_depth": {"0": 0, "...": 0},
  "config_hash": "<sha256>",
  "created_utc": "..."
}
```

- Input statistics: over ocean cells (`ocean == 1`) and training days, computed on
  `filled/X_raw` (after fill).
- Target statistics: per depth, over `depth_mask[d] == 1` cells and training days.
- Depth keys are strings of the integer depth in metres.
- Normalization: `z = (v - mean) / std`. Denormalization: `v = z * std + mean`.
  Implemented once in `src/common/normalization.py`; nothing else re-implements it.

## Mask rules (implemented in `src/preprocess/masks.py`)

1. `glorys_valid0[h,w]` = fraction of days where GLORYS depth-0 is finite ≥ `ocean_valid_fraction`.
2. `sst_valid[h,w]` = fraction of days where regridded OSTIA is finite ≥ `ocean_valid_fraction`.
3. `ocean = glorys_valid0 & sst_valid`.
4. If `keep_largest_ocean_component`: label 4-connected components of `ocean`
   (`scipy.ndimage.label`) and keep only the largest. Log how many cells were removed and where.
5. `depth_mask[d] = ocean & (GLORYS at depth d finite on ≥ ocean_valid_fraction of days)`.
6. Consistency: `depth_mask[d+1] ⊆ depth_mask[d]` must hold; assert it.
7. Any cell inside `depth_mask[d]` that is NaN in Y on a particular day stays NaN in `Y`
   (the dataset turns it into mask 0 for that sample). Log the count; it should be ~0.

## Sanity checks (run `python -m src.preprocess.sanity --config config.yaml`)

The script prints PASS/FAIL per item and writes figures to `results/figures/sanity/`.

- [ ] `X` has no NaN; `Y` has no NaN inside `depth_mask` on more than 0.1% of (day, cell, depth)
- [ ] `norm_stats.json.years_used` equals the training years from `config.yaml`
- [ ] Normalized training-year `X` per channel over ocean: |mean| < 0.05, |std − 1| < 0.05
- [ ] Normalized training-year `Y` per depth over mask: |mean| < 0.05, |std − 1| < 0.05
- [ ] `depth_mask` sizes are non-increasing with depth; print the 15 counts
- [ ] Figure: one day of each of the 7 channels with the ocean mask contour — coastlines line up
- [ ] Figure: GLORYS profiles at 4 points (central Bay ~15°N 88°E, Andaman Sea ~10°N 95°E,
      northern Bay ~20°N 89°E, south ~7°N 85°E) — thermocline roughly 50–150 m
- [ ] Figure: SSS fill fraction over time (expect a ~1-month block in mid-2019 if the SMAP
      safe-mode outage is present — **Verify**)
- [ ] `provenance.csv`: print the stream switch dates per product and which split they fall in.
      **If any product switches stream inside the test year, print a WARNING** and record it in
      `docs/decisions_log.md`.

## Done when

All sanity checks print PASS (the stream-switch item may be WARNING if documented).
