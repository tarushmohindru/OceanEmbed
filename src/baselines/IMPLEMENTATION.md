# `src/baselines/` — Climatology, linear regression, GLORYS reference

Purpose: the non-neural baselines, built **before** the neural models so the evaluation pipeline
is exercised early. Each baseline writes prediction stores (format in `src/IMPLEMENTATION.md`)
for `val` and `test`; `src/evaluation/evaluate.py` scores them exactly like the neural models.

The plain CNN baseline lives in `src/models/cnn_unet.py` and is trained by `src/training`, because
it shares the neural training recipe.

Runs on: Lightning CPU machine (minutes). No torch imports here.

## Files

```text
src/baselines/
  __init__.py
  climatology.py         # ENTRY: model_id "climatology"
  linreg.py              # ENTRY: model_id "linreg" (per-cell ridge) or "linreg_pooled"
  glorys_reference.py    # ENTRY: model_id "glorys" (truth copy, used only as the ARGO reference)
```

---

## `climatology.py`

Prediction for date t, depth d, cell (i, j): `climatology_Y[doy_index(t), d, i, j]` in °C
(train years only, 31-day smoothing — already built by `src/preprocess/climatology.py`).
Ignores inputs entirely.

```python
def predict_climatology(cfg, paths, split: str) -> tuple[np.ndarray, pd.DatetimeIndex]
def main() -> None   # writes stores for val and test; --splits to override
```

Assert: finite everywhere inside `depth_mask` (if not, the climatology has holes → fail loudly).

## `linreg.py`

Per-cell ridge regression from the 7 normalized surface inputs at a cell to the 15 normalized
depths at the same cell. No spatial context (that is the point of this baseline).

```python
def fit_per_cell(X: np.ndarray, Y: np.ndarray, valid: np.ndarray, alpha: float
                 ) -> tuple[np.ndarray, np.ndarray]
    # X (N, C, H, W), Y (N, D, H, W) normalized, valid (N, D, H, W) bool
    # returns W (H, W, C, D) and b (H, W, D)
def predict_per_cell(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray   # (N, D, H, W) normalized
def select_alpha(cfg, paths, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100)) -> float
def main() -> None   # --variant per_cell|pooled
```

Fitting details (`per_cell`):

1. Training split only. Reshape to cells: `Xc (P, N, C)`, `Yc (P, N, D)` for the P ocean cells.
2. For each cell and depth, use only days where the target is valid. Because masks are static, a
   cell's valid depths are the same every day except rare NaN days; implement as: per cell, keep days
   where **all depths in that cell's depth_mask** are finite; log the number of dropped (cell, day).
3. Centre X and Y per cell over the kept days (the intercept is not penalized):
   `W = (XcᵀXc + α·n·I)⁻¹ XcᵀYc`, `b = ȳ − x̄ᵀW`. Batched with `np.einsum` + `np.linalg.solve`
   over cells, float64.
4. Depths outside a cell's depth_mask: prediction NaN.
5. `alpha` is selected on the **val** split by mean (over depths) RMSE in °C; then the model fit on
   train with that alpha predicts val and test. Do not refit on train+val (keeps the protocol the same
   as the neural models, which only use val for early stopping/selection).
6. Write `results/metrics/linreg_alpha_selection.csv` (`alpha, val_rmse_mean`).

`pooled` variant (optional, `model_id = "linreg_pooled"`): one ridge model over all ocean cells
with features `[7 inputs, lat_s, lon_s, lat_s², lon_s², lat_s·lon_s]` (lat_s, lon_s scaled to
[−1, 1]) — tests whether location-dependent mapping matters. Same alpha selection.

Denormalize predictions with `denormalize_y` before writing the store.

## `glorys_reference.py`

Copies `Y_degC` for the requested split into a store with `model_id = "glorys"`. Used only by
`src/evaluation/argo.py` to score GLORYS itself against ARGO. It must never appear in the GLORYS
test metrics table (it would trivially score 0).

## Tests (`tests/test_baselines.py`, synthetic)

- Climatology store values equal `climatology_Y[doy]` for a few hand-picked dates; leap day maps
  to slot 59.
- `fit_per_cell` recovers known per-cell linear maps (Y = A_cell X + c_cell + small noise) with
  α → 0 (max error < 1e-3).
- Masked depths produce NaN; land cells produce NaN.
- Alpha selection never touches the test split (monkeypatch the loader to raise on `split="test"`
  during selection).

## Done when

- Tests pass on the authoring machine.
- On Lightning: `results/predictions/{climatology,linreg}/{val,test}.zarr` exist and
  `python -m src.evaluation.evaluate --models climatology linreg --split test` produces metrics.
