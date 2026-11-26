# `tests/` — Test suite

Purpose: prove each stage's contract on small synthetic data, on the authoring machine (Windows,
CPU, no data, no network), and run a few data-dependent checks on Lightning.

Run: `pytest` (the default `addopts` in `pyproject.toml` deselects `data`, `gpu`, `network`).
On Lightning after preprocessing: `pytest -m "data"`; on a GPU machine: `pytest -m "gpu"`.

## Files

```text
tests/
  conftest.py                  # fixtures (below)
  test_common_config.py
  test_common_grid_time.py
  test_common_io_norm.py
  test_download.py
  test_preprocess_regrid.py
  test_preprocess_vertical_qc.py
  test_preprocess_clim_fill.py
  test_preprocess_masks_tensors.py
  test_dataset.py
  test_models.py
  test_baselines.py
  test_training.py
  test_evaluation.py
  test_integration_synthetic.py
  test_data_lightning.py       # all tests marked `data`
```

The specific cases for each file are listed in the "Tests" section of the matching
`src/<package>/IMPLEMENTATION.md`. This file defines shared fixtures, markers and the
cross-cutting tests.

## `conftest.py` fixtures

```python
@pytest.fixture(scope="session")
def cfg() -> Config                     # loads the repo's config.yaml (validates it)

@pytest.fixture
def tiny_cfg(cfg, tmp_path) -> Config   # cfg with paths.data_root/results_root inside tmp_path

@pytest.fixture(scope="session")
def synthetic_processed(tmp_path_factory, cfg) -> Config
    # calls src.dataset.synthetic.make_synthetic_processed once per session (3 years, 24×32 grid,
    # 15 depths) and returns the Config pointing at it

@pytest.fixture
def rng() -> np.random.Generator        # np.random.default_rng(0)
```

All tests must be deterministic (seeded) and leave nothing outside `tmp_path`.

## Markers

| Marker | Meaning | Where it runs |
| --- | --- | --- |
| (none) | pure unit test, < 30 s total per file | everywhere |
| `slow` | > 30 s on CPU (overfit checks, smoke training) | everywhere, may be skipped with `-m "not slow"` |
| `data` | needs the real `data/processed/` | Lightning after preprocessing |
| `gpu` | needs CUDA | Lightning GPU |
| `network` | needs internet + credentials | Lightning, manually |

Optional packages: use `pytest.importorskip("xesmf")`, `("gsw")`, `("cartopy")`, `("argopy")`
inside the tests that need them.

## Required cross-cutting tests (from the draft's list, all mandatory)

| Test | File | Assertion |
| --- | --- | --- |
| Grid alignment | `test_common_grid_time.py` | default box → 72 × 80, centres 5.125…22.875 / 80.125…99.875 |
| Mask shapes | `test_preprocess_masks_tensors.py` | `ocean` (H, W), `depth_mask` (D, H, W), nested with depth |
| No NaN inside masks | `test_preprocess_masks_tensors.py` | X no NaN anywhere; Y finite inside `depth_mask` |
| No date in two splits | `test_dataset.py` | `split_indices` disjoint + gap respected; deliberately broken dates fail |
| Model output shape | `test_models.py` | both models → (B, 15, Hp, Wp); after crop (B, 15, 72, 80) |
| Loss on a tiny batch | `test_training.py` | hand-computed value for a 1 × 2 × 3 × 3 example with mask, λ = 0.1 |
| Metric equivalence | `test_training.py` | training-time RMSE == evaluation RMSE |

`test_integration_synthetic.py` (marked `slow`): on `synthetic_processed`, run end to end —
climatology store → linreg store → 2-epoch ViT-CNN and CNN runs (`--synthetic` path) → predict →
`evaluate` → `tables`. Asserts all expected files exist and that linreg RMSE < climatology RMSE on
the synthetic data (the synthetic targets are a function of the inputs, so this must hold).

## `test_data_lightning.py` (marker `data`)

- `processed/` passes every check in `data/processed/IMPLEMENTATION.md` (reuse `sanity.py`
  functions; do not duplicate logic).
- Regridding cross-check vs `xesmf` on one OSTIA day (max abs diff < 0.01 °C).
- `split_indices` counts for the real config.
- One real batch through each model on CPU gives finite output.

## Done when

`pytest` passes on the authoring machine with zero failures (skips allowed only for the
markers/importorskips above). The count of tests and skips is recorded in `docs/decisions_log.md`
at each milestone.
