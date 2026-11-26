# `src/dataset/` — Splits, in-memory data, batching

Purpose: turn `data/processed/` into split-specific arrays/tensors for baselines (numpy) and
neural models (torch), with padding, masks, optional extra channels and anomaly bases. This is
the only package that reads `X.zarr`/`Y.zarr` for modelling.

Depends on: `src/common`. Used by: `src/baselines`, `src/training`, `src/evaluation` (splits only).

## Files

```text
src/dataset/
  __init__.py      # re-export: split_indices, load_split_numpy, SplitTensors, build_split_tensors, BatchIterator
  splits.py        # split logic + leakage assertions (no torch)
  arrays.py        # load processed data as numpy (no torch)
  tensors.py       # padding, masks, extras, anomaly base -> torch tensors
  iterator.py      # GPU-resident batch iterator
  synthetic.py     # synthetic processed/ directory for tests and smoke runs (no torch)
```

---

## `splits.py` (no torch)

```python
SPLITS = ("train", "val", "test")

def split_indices(cfg, paths) -> dict[str, np.ndarray]
    # Reads processed/dates.csv. Recomputes split labels with timeutil.split_of(dates, cfg)
    # and asserts they equal the 'split' column (guards against stale processed data built
    # with another config). Returns sorted int arrays of t_index for train/val/test
    # ('gap' and 'none' days are excluded from every split).
def assert_no_leakage(idx: dict[str, np.ndarray], dates: pd.DatetimeIndex, cfg) -> None
    # 1. pairwise-disjoint index sets
    # 2. every date's year belongs to its split's year range
    # 3. min(val date) - max(train date) > gap_days, same for test vs val
    # 4. none of the first gap_days days of the val/test years appear anywhere
```

`split_indices` must call `assert_no_leakage` before returning.

Expected counts for the default config: train 1827, val 350, test 350.

## `arrays.py` (no torch)

```python
@dataclass
class SplitArrays:
    split: str
    t_index: np.ndarray      # (N,)
    dates: pd.DatetimeIndex  # (N,)
    doy: np.ndarray          # (N,) doy_index
    X: np.ndarray            # (N, C, H, W) float32 normalized, no NaN, land 0
    Y: np.ndarray            # (N, D, H, W) float32 normalized, NaN outside valid
    Y_degC: np.ndarray       # (N, D, H, W) float32 °C, NaN outside valid
    ocean: np.ndarray        # (H, W) bool
    depth_mask: np.ndarray   # (D, H, W) bool
    clim_Y_norm: np.ndarray  # (N, D, H, W) float32: normalized climatology for each sample's doy
                             # (computed from climatology_Y.nc + norm_stats; NaN outside depth_mask)

def load_split_numpy(cfg, paths, split: str, load_degc: bool = True) -> SplitArrays
def load_static(cfg, paths) -> tuple[np.ndarray, np.ndarray, NormStats]   # ocean, depth_mask, stats
```

Read zarr with `xr.open_zarr(...).isel(time=t_index).values` (loads into RAM — fine at this size).
Assert shapes against `config.inputs`, `config.depths_m`, and `grid.nc`.

## `tensors.py`

```python
@dataclass
class SplitTensors:
    split: str
    x: torch.Tensor      # (N, C_in, Hp, Wp) float32
    y: torch.Tensor      # (N, D, Hp, Wp)   float32, normalized absolute target, 0 where invalid
    m: torch.Tensor      # (N, D, Hp, Wp)   bool, True where target valid (depth_mask & finite)
    base: torch.Tensor   # (N, D, Hp, Wp)   float32; zeros (absolute mode) or normalized clim (anomaly mode)
    t_index: torch.Tensor  # (N,) int64
    H: int; W: int       # unpadded size, for cropping
    channel_names: list[str]

def pad_multiple_for(cfg, model_name: str) -> int
    # vit_cnn: model.patch ; cnn_unet: 2 ** n_down from src/models/cnn_unet (default 8)
def pad_to_multiple(a: np.ndarray, multiple: int, value: float = 0.0) -> np.ndarray
    # pads the last two axes AT THE END (north rows, east columns) up to the next multiple
def extra_channels(cfg, arrays: SplitArrays) -> tuple[np.ndarray, list[str]]
    # builds optional planes per config.extra_inputs, each (N, 1 or 2, H, W):
    #   land_mask  -> ocean as 0/1 (name "ocean_mask")
    #   doy_sincos -> sin/cos(2π doy/366) broadcast over the plane ("doy_sin","doy_cos")
    #   latlon     -> lat and lon scaled to [-1, 1] over the box ("lat","lon")
def build_split_tensors(cfg, paths, split: str, model_name: str,
                        device: torch.device, smoke: bool = False) -> SplitTensors
```

`build_split_tensors` logic:

1. `arrays = load_split_numpy(cfg, paths, split, load_degc=False)`.
2. If `smoke`: keep only the first 31 samples of the split (and log it).
3. Drop channels listed in `train.drop_inputs` from `arrays.X` (by name, using `config.inputs`
   order), then `x = concat(arrays.X, extra_channels)` along the channel axis. Log the final
   `channel_names`.
4. `m = depth_mask[None] & isfinite(arrays.Y)`; `y = nan_to_num(arrays.Y, nan=0.0)`.
5. `base = nan_to_num(arrays.clim_Y_norm, 0)` if `train.target_mode == "anomaly"` else zeros.
6. Pad x, y, base with 0 and m with False to `pad_multiple_for(cfg, model_name)`.
7. Convert to torch; place on `device` if `train.gpu_resident` else keep on CPU
   (`pin_memory()` when CUDA is available).
8. Assert: no NaN in x, y, base; `m` False on all padding; shapes consistent.

Contract for models and losses: the network outputs `out` (B, D, Hp, Wp) and the prediction in
normalized absolute units is **`pred = out + base`**. The loss compares `pred` with `y` under `m`.
In anomaly mode this is exactly an anomaly loss; in absolute mode `base = 0`. One code path.

## `iterator.py`

```python
class BatchIterator:
    def __init__(self, t: SplitTensors, batch_size: int, shuffle: bool, seed: int,
                 device: torch.device, drop_last: bool = False)
    def __len__(self) -> int                         # number of batches
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]
        # yields {"x","y","m","base","t_index"} for each batch via index_select;
        # moves to device (non_blocking) if the tensors live on CPU
    def set_epoch(self, epoch: int) -> None          # reshuffle with seed + epoch (reproducible)
```

Shuffle uses a `torch.Generator` seeded with `seed + epoch`. Never shuffle val/test.

## `synthetic.py` (no torch)

```python
def make_synthetic_processed(root: Path, cfg, years=(2016, 2017, 2018), H: int = 24, W: int = 32,
                             seed: int = 0) -> Config
    # Writes a complete, contract-conforming processed/ directory under root/processed
    # (grid.nc, masks.nc, X.zarr, Y.zarr, Y_degC.zarr, climatology_Y.nc, norm_stats.json,
    #  dates.csv, provenance.csv, fill_stats.csv) and returns a Config copy whose
    # paths.data_root = root, dates/splits = the given years (one year per split), region resized
    # so the grid is H × W at grid_res.
```

Content must be learnable so smoke runs and overfit tests are meaningful:

- Land: a rectangular "continent" in one corner plus a 2-cell "shelf" ring where depths
  ≥ 200 m are masked (so `depth_mask` shrinks with depth).
- Inputs: smooth random fields (sum of a few 2-D sinusoids with random phases evolving slowly in
  time) plus a seasonal cycle for sst/sss.
- Targets (°C): `T_d = sst · exp(−z_d/150) + 0.5 · sla · g(z_d) + 4 + noise`, with `g` peaking at
  ~100 m; clamp to [2, 32]. Climatology, norm stats and fill stats computed by the **real**
  functions from `src/preprocess` on this synthetic data (this doubles as an integration test).

## Memory budget (default box, padded 80 × 80)

| Tensor (train) | Size |
| --- | --- |
| x 1827 × 7 × 80 × 80 float32 | ~330 MB |
| y, base 1827 × 15 × 80 × 80 float32 | ~700 MB each (base only allocated in anomaly mode; otherwise use a broadcast zero) |
| m bool | ~175 MB |

Fits in any ≥ 8 GB GPU. In absolute mode store `base` as a single zero tensor expanded with
`.expand(...)` to avoid allocating it.

## Tests (`tests/test_dataset.py`)

Build a tiny synthetic `processed/` (3 years 2016–2018 with splits 2016/2017/2018, gap 15, 12×16
grid, 3 depths) in a pytest fixture (`tests/conftest.py`, see `tests/IMPLEMENTATION.md`).

- `split_indices`: disjoint, counts (366, 365−15, 365−15), leakage assertion fails if gap_days
  is violated (construct a bad dates.csv).
- Stale data check: changing `splits` in the config without rebuilding → assertion error.
- `pad_to_multiple`: 12×16 with multiple 8 → 16×16, padding at the end, original block unchanged.
- `build_split_tensors`: shapes; `m` False on padding and land; y finite; anomaly mode base equals
  normalized climatology for the sample's doy; extra channel names/count.
- `BatchIterator`: covers each index exactly once per epoch; same seed+epoch → same order;
  different epoch → different order; val iterator unshuffled.

## Done when

`pytest tests -k dataset` passes on the authoring machine.
