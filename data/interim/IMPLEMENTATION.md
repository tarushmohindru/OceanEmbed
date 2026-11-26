# `data/interim/` — Regridded daily files contract

Written by `src/preprocess/regrid.py`, `src/preprocess/vertical.py`, `src/preprocess/fill.py`
and `src/preprocess/climatology.py`. Read by later preprocessing steps only.

## Layout

```text
data/interim/
  _weights/<product>_<method>.nc      # cached xESMF regridding weights
  grid.nc                              # copy of the target grid (also in processed/)
  sst/sst_<YYYY>.nc                    # one file per INPUT CHANNEL GROUP per year, see below
  sss/sss_<YYYY>.nc
  sla/sla_<YYYY>.nc
  cur/cur_<YYYY>.nc                    # u_cur, v_cur
  wind/wind_<YYYY>.nc                  # u_wind, v_wind
  glorys/glorys_<YYYY>.nc              # thetao on target grid, NATIVE depth levels
  glorys_t15/glorys_t15_<YYYY>.nc      # thetao on target grid, 15 TARGET depths
  provenance/<product>_<YYYY>.csv      # date,stream,dataset_id for every date
  clim_inputs.nc                       # day-of-year climatology of each input channel (train years, after QC, before fill)
  filled/X_raw_<YYYY>.nc               # all 7 channels after QC + fill, physical units, not normalized
```

## File schemas

### Input channel files (`sst/`, `sss/`, `sla/`, `cur/`, `wind/`)

- Dimensions: `time` (daily, every date of that year within `dates`), `lat` (H), `lon` (W).
- Coordinates exactly equal to the target grid in `grid.nc` (assert with `np.allclose(atol=1e-6)`).
- Variables named by **channel name** (`sst`, `sss`, `sla`, `u_cur`, `v_cur`, `u_wind`, `v_wind`),
  float32, physical units from the root spec §5.3, `units` attribute set.
- QC already applied: out-of-range values are NaN (`src/preprocess/qc.py`).
- Not gap-filled. NaN means missing or land.
- Global attribute `source_streams`: comma-separated stream names that contributed.

### `glorys/glorys_<YYYY>.nc`

- Dimensions `time, depth, lat, lon`; `depth` = native GLORYS levels within `depth_range_m`.
- Variable `thetao` (°C, potential temperature), float32. NaN below sea floor and on land.

### `glorys_t15/glorys_t15_<YYYY>.nc`

- Dimensions `time, depth, lat, lon`; `depth` = `config.depths_m` exactly (15 values).
- Variable `thetao` (°C). Depth 0 holds the shallowest native level (0.494 m) unchanged.
- NaN wherever either bracketing native level is NaN (never extrapolated).

### `clim_inputs.nc`

- Dimensions `doy` (366, month-day slots, root spec §5.2), `lat`, `lon`; one variable per channel.
- Computed from **training years only**, after QC, before fill; circular 31-day smoothing.
- Used only by `fill.py` (step "climatology fill").

### `filled/X_raw_<YYYY>.nc`

- Dimensions `time, channel, lat, lon`; `channel` coordinate = `config.inputs` order.
- Variable `X` float32, physical units. No NaN inside the ocean mask; NaN outside (land).

## Done when

- One file per year for every channel group and for both GLORYS stages.
- `python -m src.preprocess.check_interim --config config.yaml` passes (implemented in
  `src/preprocess`): grid equality, date completeness, dtype, units attributes.
