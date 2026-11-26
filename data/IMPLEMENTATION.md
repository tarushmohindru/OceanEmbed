# `data/` — Data layout and contracts

No code lives here. This directory holds every data file the pipeline reads or writes, in three
stages. Only the `IMPLEMENTATION.md` files are tracked by git; everything else is ignored.

```text
data/
  raw/        # downloads, cropped to region+buffer, otherwise untouched   -> data/raw/IMPLEMENTATION.md
  interim/    # per-product, per-year files on the target grid, daily      -> data/interim/IMPLEMENTATION.md
  processed/  # model-ready tensors, masks, statistics                       -> data/processed/IMPLEMENTATION.md
```

The location is `paths.data_root` in `config.yaml` (default `data`, relative to repo root).
All code obtains paths through `src.common.paths.Paths` (see `src/common/IMPLEMENTATION.md`);
**no module may hard-code a path under `data/`**.

## Stage flow

```text
raw/<product>/<stream>/...           src/download/*
      │
      ▼  regrid + daily mean + unit conversion + QC     (src/preprocess/regrid.py, qc.py)
interim/<product>/<product>_<YYYY>.nc
      │
      ▼  GLORYS vertical interpolation                  (src/preprocess/vertical.py)
interim/glorys_t15/glorys_t15_<YYYY>.nc
      │
      ▼  masks, fill, climatologies, normalization      (masks.py, fill.py, climatology.py, normalize.py)
      ▼  tensors                                         (build_tensors.py)
processed/X.zarr  Y.zarr  masks.nc  grid.nc  climatology_Y.nc  norm_stats.json  provenance.csv  fill_stats.csv
```

## Disk budget on Lightning (~200 GB available)

| Stage | Bay of Bengal, 7 years | Keep? |
| --- | --- | --- |
| raw | 25–50 GB (GLORYS ~20–25 GB dominates) | Yes. Allows re-preprocessing without re-downloading |
| interim | ~5 GB | Yes |
| processed | ~1.5 GB | Yes; copy to results archive at the end |
| predictions | ~130 MB per model per split | Yes (in `results/`) |

Global daily files (OSCAR, SMAP, CCMP) are **never kept**: downloaders crop each file to the
region+buffer immediately and delete the global file (see `src/download/IMPLEMENTATION.md`).
CCMP alone would otherwise need >100 GB.

Full-domain stretch goal (5–30°N, 45–105°E) is ~3–4× larger at the raw stage; process one
year at a time and delete the Bay of Bengal raw GLORYS first if space is short.

## Rules

1. `raw/` is written only by `src/download/`. Never modify a raw file after it is complete.
2. `interim/` and `processed/` are fully reproducible from `raw/` + `config.yaml`.
   Deleting them must never lose information.
3. Each stage directory gets a `_meta.json` (`src.common.io.write_meta`) when its producer
   finishes.
4. Files are written atomically: write to `<name>.part` (or a `<name>.tmp/` directory for zarr),
   then rename. A crash must never leave a file that looks complete.
