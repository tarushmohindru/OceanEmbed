# `src/training/` — Loss, training loop, prediction, sweeps

Purpose: train `vit_cnn` and `cnn_unet` with one identical recipe, select on the validation
split, and write prediction stores for evaluation.

Runs on: Lightning GPU machine (smallest available: T4/L4/A10G). Also runs on CPU for the smoke
test. Depends on: `src/common`, `src/dataset`, `src/models`.

## Files

```text
src/training/
  __init__.py
  losses.py       # masked depth-weighted MSE + gradient term
  schedule.py     # warmup + cosine LR schedule
  precision.py    # AMP dtype resolution, autocast context, GradScaler
  checkpoint.py   # save/load/resume
  metrics_live.py # streaming per-depth RMSE in °C for validation during training
  train.py        # ENTRY: one training run
  predict.py      # ENTRY: run directory -> prediction stores (val/test)
  sweep.py        # ENTRY: λ sweep, ablations, seed ensemble (sequential runs)
```

---

## `losses.py`

All tensors are padded (B, D, Hp, Wp); `pred = out + base`, `y` normalized, `m` bool.

```python
def depth_weights(cfg, norm_stats: NormStats) -> torch.Tensor   # (D,), mean = 1
    # uniform:     ones
    # thermocline: thermocline_weight for depths 50 <= z <= 200 m, else 1; then divide by the mean
    # variance:    out_std**2 (σ_d² in °C²), divided by its mean  -> approximates MSE in °C
def masked_mse_per_depth(pred, y, m) -> torch.Tensor           # (D,)
    # for each depth: sum over (B, H, W) of m*(pred-y)^2 / max(sum m, 1)
def gradient_loss_per_depth(pred, y, m) -> torch.Tensor        # (D,)
    # forward differences along W (x, lon) and H (y, lat):
    #   dx = t[..., :, 1:] - t[..., :, :-1];  mx = m[..., :, 1:] & m[..., :, :-1]
    #   dy = t[..., 1:, :] - t[..., :-1, :];  my = m[..., 1:, :] & m[..., :-1, :]
    #   per depth: (sum mx*|dx_pred - dx_y| / max(sum mx,1)) + (sum my*|dy_pred - dy_y| / max(sum my,1))
    # Grid spacing differences with latitude (cos 5°–23° = 0.996–0.92) are ignored; note it in the docstring.
def total_loss(pred, y, m, w: torch.Tensor, lam: float) -> tuple[torch.Tensor, dict[str, float]]
    # L = sum_d w_d * mse_d + lam * sum_d grad_d   (matches the draft's formula)
    # returns loss and {"loss", "mse", "grad"} as python floats for logging
```

Compute the loss in float32 even under autocast (`pred.float()`). Masks guarantee that land,
padding and missing targets never contribute; the mask-pairs in the gradient term guarantee that
coastlines do not create false gradients.

## `schedule.py`

```python
def warmup_cosine(optimizer, steps_per_epoch: int, warmup_epochs: int, total_epochs: int,
                  base_lr: float, min_lr: float) -> torch.optim.lr_scheduler.LambdaLR
    # per-STEP schedule: linear from 0 to base_lr over warmup steps, then cosine to min_lr at the
    # last step of total_epochs. Early stopping may end training before the cosine finishes; that is fine.
```

## `precision.py`

```python
def resolve_amp(cfg, device) -> torch.dtype | None
    # "off" or CPU -> None; "bf16" -> torch.bfloat16 (error if unsupported);
    # "fp16" -> torch.float16; "auto" -> bf16 if torch.cuda.is_bf16_supported() else fp16
def autocast_ctx(dtype, device)  # torch.autocast(device_type="cuda", dtype=dtype) or nullcontext
def make_scaler(dtype)           # torch.amp.GradScaler("cuda") only for fp16, else a no-op object
```

T4 has no bf16 → fp16 + GradScaler. A100/L4/H100/H200 → bf16, no scaler.

## `checkpoint.py`

```python
def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, epoch: int,
                    best_val: float, cfg_raw: dict, extra: dict) -> None   # atomic
def load_checkpoint(path: Path, map_location) -> dict
```

Contents: `model`, `optimizer`, `scheduler`, `scaler` state dicts; `epoch`; `best_val`; `cfg`
(resolved dict); `git_hash`; `torch_rng`, `cuda_rng`, `numpy_rng`, `python_rng` states;
`model_name`; `in_channels`; `channel_names`; `grid_hw_padded`; `param_count`.

## `metrics_live.py`

```python
class StreamingDepthRMSE:
    def __init__(self, out_std: torch.Tensor)            # (D,) °C per normalized unit
    def update(self, pred, y, m) -> None                 # accumulates sum of squared error and counts per depth
    def compute(self) -> dict                            # {"rmse_degC": (D,) np.ndarray, "mean_rmse_degC": float}
```

RMSE per depth = sqrt(Σ m·(pred−y)²·σ_d² / Σ m) pooled over all samples and cells of the split;
mean RMSE = arithmetic mean over depths. This is **identical** to the definition used by
`src/evaluation/metrics.py` (test both on the same synthetic data to prove it).

## `train.py` (ENTRY)

```bash
python -m src.training.train --config config.yaml --model vit_cnn \
    [--run-name NAME] [--set train.grad_lambda=0.05 ...] [--resume results/runs/NAME] [--synthetic]
```

Procedure:

1. Parse config with overrides; `seed_everything(cfg.seed)`; device = cuda if available.
2. Run directory `results/runs/<run_name>/` (default run name
   `<model>_<YYYYmmdd-HHMMSS>`; error if it exists and `--resume` not given). Write
   `config.yaml` (resolved), `_meta.json` (incl. GPU name `torch.cuda.get_device_name()`,
   torch/CUDA versions), and start `train.log`.
3. Data: `build_split_tensors` for train and val (`smoke` from `train.smoke`); `BatchIterator`
   train (shuffle) and val (no shuffle). Assert the leakage check ran.
4. Model: `build_model`; log parameter count; `torch.compile` **not** used (simplicity, T4
   compatibility).
5. Optimizer: AdamW(lr, weight_decay=wd, betas=(0.9, 0.999)); **no weight decay** on biases,
   norm weights and the position embedding (two parameter groups).
6. Schedule: `warmup_cosine` per step. AMP via `precision.py`. Gradient clipping at `grad_clip`
   (unscale first when using GradScaler).
7. Epoch loop (1 … epochs):
   - train: `model.train()`, iterate batches, `pred = model(x) + base`, `total_loss`, backward,
     clip, step, scheduler step; accumulate mean loss/mse/grad.
   - validate: `model.eval()`, `torch.no_grad()`, autocast; `StreamingDepthRMSE` on val; also the
     val loss.
   - append a row to `metrics.csv`: `epoch, lr, train_loss, train_mse, train_grad, val_loss,
     val_rmse_mean, val_rmse_<depth> ×15, epoch_seconds`.
   - if `val_rmse_mean` improved by > 1e-4: save `best.pt`, reset patience counter.
   - always save `last.pt` (atomic; overwritten each epoch).
   - stop when the patience counter reaches `patience`; stop immediately if loss is NaN/inf
     (log the batch, save `nan_debug.pt` with the batch and state, exit non-zero).
8. At the end write `summary.json`: best epoch, best val RMSE (mean and per depth), epochs run,
   total seconds, parameter count, GPU name.

`--resume`: load `last.pt` (model, optimizer, scheduler, scaler, RNG states, epoch, best, patience
counter) and continue. A Lightning session ending mid-run must lose at most one epoch.

`--synthetic`: generate a synthetic `processed/` directory with
`src.dataset.synthetic.make_synthetic_processed(tmpdir, cfg)` and point `paths.data_root` to it.
Used for the smoke test on the authoring machine.

Smoke test definition: `python -m src.training.train --config config.yaml --model vit_cnn --synthetic --set train.smoke=true --set train.epochs=2`
finishes on CPU, writes all run files, and val RMSE is finite. Same for `cnn_unet`.

## `predict.py` (ENTRY)

```bash
python -m src.training.predict --config config.yaml --run-dir results/runs/NAME \
    --model-id vit_cnn [--splits val test] [--checkpoint best]
```

1. Load the run's `config.yaml` (not the current one — predictions must match the trained setup)
   and the checkpoint.
2. For each split: build tensors (no shuffle), predict in batches with `torch.no_grad()`,
   `pred = out + base`, **crop padding** to (H, W), denormalize with `denormalize_y`, set NaN outside
   `depth_mask`, write the store with `write_prediction_store` (attrs include run_dir, checkpoint).
3. Assert the output shape and that NaN pattern equals `depth_mask` broadcast.

## `sweep.py` (ENTRY)

Runs several `train.py` + `predict.py` jobs sequentially in-process (or via subprocess),
skipping finished runs (`summary.json` exists). Named experiment sets:

| Set | Runs | Selection |
| --- | --- | --- |
| `lambda` | for model ∈ {cnn_unet, vit_cnn}: grad_lambda ∈ {0, 0.05, 0.1, 0.3} | best val RMSE **per model**; writes `results/tables/lambda_sweep.csv` and `results/runs/selected.json` (`{"vit_cnn": run_name, "cnn_unet": run_name}`) |
| `core` | the two selected runs → `predict` for val and test with model_ids `vit_cnn`, `cnn_unet` | — |
| `seeds` | selected configs with seeds {1, 2, 3, 4} → model_ids `vit_cnn_seed<k>` | ensemble mean store `vit_cnn_ens` (mean of member predictions) |
| `ablations` | ViT-CNN selected config with one change each: `model.patch=4`; `model.use_skip_stem=false`; `model.depth=4`; `model.depth=8`; `train.target_mode=anomaly`; `train.depth_weights=variance`; `train.depth_weights=thermocline`; `extra_inputs.land_mask=true`; `train.drop_inputs=[sss]`; `train.drop_inputs=[u_cur,v_cur]` | reported, not selected |

```bash
python -m src.training.sweep --config config.yaml --set-name lambda
```

Protocol rules: λ and every other choice are selected on **val** only. Test predictions are
produced for the selected runs and for ablations, and reported as they are — never used to pick
anything.

`train.drop_inputs`: list of channel names removed from `x` in `build_split_tensors` (add this
key to `config.yaml` under `train`, default `[]`, and to the validator: names must be in
`inputs`).

## Expected runtime (estimate — replace with measured numbers in `docs/`)

~115 steps per epoch at batch 16; each step is a few GFLOPs per sample. Expect seconds to tens of
seconds per epoch on a T4 with GPU-resident data, i.e. minutes per run. Record the measured
`epoch_seconds` from `metrics.csv` in the write-up.

## Tests (`tests/test_training.py`, CPU)

- `masked_mse_per_depth`: hand-computed 2×2 example; masked cells with huge errors do not change
  the value; all-masked depth returns 0 without NaN.
- `gradient_loss_per_depth`: pred = y + constant → 0; a step edge next to a masked land cell → 0
  contribution from the pair that touches land.
- `depth_weights`: mean 1 for every mode; variance mode ∝ σ².
- `warmup_cosine`: lr at step 0 ≈ 0, at end of warmup = base_lr, at the final step = min_lr.
- `StreamingDepthRMSE` equals `src/evaluation/metrics.rmse_per_depth` on the same synthetic data
  (after denormalization) to 1e-5.
- Checkpoint save → load → identical model outputs; resume continues from the right epoch.
- Smoke runs for both models (`--synthetic`, 2 epochs) complete on CPU (mark `slow`).

## Done when

- Tests pass on the authoring machine (smoke runs included).
- On Lightning GPU: `sweep --set-name lambda` and `--set-name core` finish; `selected.json` and
  stores for `vit_cnn` and `cnn_unet` exist.
