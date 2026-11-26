# `src/` — Package overview

`src` is a Python package (it has `__init__.py`). Every sub-directory is a sub-package with its
own `__init__.py` and `IMPLEMENTATION.md`. Code is run from the repo root as
`python -m src.<subpackage>.<module> --config config.yaml`.

## Sub-packages and dependency direction

```text
common  ◄──────────────────────────────────────────────── (everyone imports common)
download   → common
preprocess → common
dataset    → common
models     → common
baselines  → common, dataset
training   → common, dataset, models
evaluation → common, dataset (for splits only)
```

Rules:

- Imports only go in the arrow direction. `evaluation` must not import `training` or `models`;
  it reads **prediction stores** from disk. This keeps one evaluation path for every model.
- `download` and `preprocess` must not import `torch`.
- `models` must not import anything that touches disk.
- One allowed exception: `src/dataset/synthetic.py` (test and smoke-run support only) may
  import `src.preprocess.climatology` and `src.preprocess.normalize`, so synthetic data is built
  by the real functions. Nothing else in `dataset` imports `preprocess`.

## Files to create

```text
src/__init__.py                      # empty
src/common/__init__.py               # see src/common/IMPLEMENTATION.md
src/download/__init__.py
src/preprocess/__init__.py
src/dataset/__init__.py
src/baselines/__init__.py
src/models/__init__.py
src/training/__init__.py
src/evaluation/__init__.py
```

`__init__.py` files contain no logic; at most a module docstring and re-exports listed in the
sub-package spec.

## The prediction store (shared contract between producers and `evaluation`)

Every model — climatology, linear regression, CNN, ViT-CNN — and the GLORYS reference produce
predictions in this one format. Producers: `src/baselines/*`, `src/training/predict.py`.
Consumer: `src/evaluation/*`.

```text
results/predictions/<model_id>/<split>.zarr
  var  T_pred   (time, depth, lat, lon) float32, °C, NaN outside masks.nc:depth_mask
  coords time (dates of the split, gap days excluded), depth (config.depths_m), lat, lon
  attrs  model_id, split, created_utc, git_hash, config_hash,
         checkpoint (path or ""), run_dir (path or "")
results/predictions/<model_id>/_meta.json
```

- `<model_id>` examples: `climatology`, `linreg`, `cnn_unet`, `vit_cnn`,
  `vit_cnn_lambda0.05`, `vit_cnn_seed1`. Must match `^[a-z0-9_.\-]+$`.
- `<split>` ∈ `val`, `test`.
- Unpadded: H × W equal to the target grid (72 × 80 by default).
- Written with `src.common.io.write_prediction_store(...)` and read with
  `src.common.io.read_prediction_store(...)` — no other code touches these paths directly.
