# `src/common/` — Shared utilities

Implement this package first. Everything else depends on it. No module here imports `torch`
at top level except `seeding.py` (which imports it lazily) — `common` must be importable by
`download` and `preprocess`, which never use torch.

## Files

```text
src/common/
  __init__.py          # re-export: load_config, Paths, get_logger, Grid
  config.py            # load + validate + override config.yaml
  cli.py               # argparse helpers shared by all entry points
  paths.py             # every file path in the project
  grid.py              # target grid construction and helpers
  timeutil.py          # daily calendar, doy_index, seasons, split membership
  normalization.py     # load norm_stats.json, normalize / denormalize
  io.py                # atomic writes, meta files, NetCDF/zarr helpers, prediction store IO
  logging_utils.py     # logger factory
  seeding.py           # seed_everything
  gitinfo.py           # git hash, dirty flag
```

---

## `config.py`

```python
@dataclass(frozen=True)
class Config:
    raw: dict            # the full resolved YAML as nested dicts (read-only by convention)
    path: Path           # path of the YAML that was loaded
    def get(self, dotted: str, default=...) -> Any   # cfg.get("train.lr")
    def __getitem__(self, key: str) -> Any           # cfg["train"]["lr"]
    def hash(self) -> str                            # sha256 of yaml.safe_dump(raw, sort_keys=True)
    def dump(self, path: Path) -> None               # write resolved YAML

def load_config(path: str | Path, overrides: list[str] | None = None) -> Config
def apply_overrides(raw: dict, overrides: list[str]) -> dict
def validate(raw: dict) -> None                      # raises ConfigError with ALL problems listed
class ConfigError(ValueError): ...
```

`apply_overrides`: each override is `dotted.key=value`; `value` is parsed with
`yaml.safe_load` (so `3`, `0.1`, `true`, `[1,2]` become typed). Creating a new top-level key is
an error; creating a new nested key is allowed only under `extra_inputs` and `train`.

`validate` checks (collect all errors, then raise once):

- Required top-level keys: `seed, paths, region, grid_res, dates, splits, depths_m, inputs,
  channels, products, preprocess, model, train, eval`.
- `region`: `lat_min < lat_max`, `lon_min < lon_max`; `(lat_max-lat_min)/grid_res` and
  `(lon_max-lon_min)/grid_res` are integers (tolerance 1e-9).
- `dates.start <= dates.end`, both parse with `pd.Timestamp`.
- Split year ranges: inside the date range, non-overlapping, ordered train < val < test.
- `depths_m`: strictly increasing, length 15 (warn, don't fail, if not 15), first value 0.
- `depth_bands_m`: every listed depth is in `depths_m`; bands do not overlap.
- Every name in `inputs` has an entry in `channels`, whose `product` exists in `products`.
- `model.patch` is a power of two ≥ 2; `model.dim % model.heads == 0`.
- `train.lr`, `train.wd`, `train.grad_lambda` are floats (catches the PyYAML `3e-4` string trap).
- `train.depth_weights ∈ {uniform, thermocline, variance}`;
  `train.target_mode ∈ {absolute, anomaly}`; `train.amp ∈ {auto, bf16, fp16, off}`.
- `eval.seasons` months cover 1–12 exactly once.
- `train.drop_inputs` names are all in `inputs`; `train.patience <= train.epochs`.

## `cli.py`

```python
def base_parser(description: str) -> argparse.ArgumentParser
    # adds: --config (default "config.yaml"), --set (action="append", default []),
    #       --force (store_true), --log-level (default INFO)
def parse_and_load(parser) -> tuple[argparse.Namespace, Config]
    # parses args, calls load_config(args.config, args.set), configures logging
```

## `paths.py`

```python
class Paths:
    def __init__(self, cfg: Config, repo_root: Path | None = None)  # repo_root default: parent of src/
    # directories (created on access with mkdir(parents=True, exist_ok=True))
    raw_dir: Path                 # <data_root>/raw
    interim_dir: Path
    processed_dir: Path
    results_dir: Path
    def raw_product_dir(self, product: str, stream: str) -> Path
    def raw_year_file(self, product: str, stream: str, year: int) -> Path    # glorys/my/glorys_my_2016.nc
    def raw_tmp_dir(self) -> Path
    def manifest(self) -> Path
    def weights_file(self, product: str, method: str) -> Path
    def interim_channel_file(self, group: str, year: int) -> Path            # group in {sst,sss,sla,cur,wind}
    def interim_glorys(self, year: int) -> Path
    def interim_glorys_t15(self, year: int) -> Path
    def interim_provenance(self, product: str, year: int) -> Path
    def clim_inputs(self) -> Path
    def filled_x(self, year: int) -> Path
    # processed
    grid: Path; masks: Path; x_zarr: Path; y_zarr: Path; y_degc_zarr: Path
    clim_y: Path; norm_stats: Path; provenance: Path; fill_stats: Path; dates_csv: Path
    # results
    def run_dir(self, run_name: str) -> Path                                 # results/runs/<run_name>
    def prediction_store(self, model_id: str, split: str) -> Path
    metrics_dir: Path; figures_dir: Path; tables_dir: Path; probe_dir: Path; logs_dir: Path
```

Channel groups map: `sst→sst`, `sss→sss`, `sla→sla`, `u_cur,v_cur→cur`, `u_wind,v_wind→wind`.
Define it as a module constant `CHANNEL_GROUP: dict[str, str]` here.

## `grid.py`

```python
@dataclass(frozen=True)
class Grid:
    lat: np.ndarray        # (H,) ascending cell centres
    lon: np.ndarray        # (W,) ascending
    res: float
    @property H, W
    def lat_bounds(self) -> np.ndarray   # (H+1,) edges
    def lon_bounds(self) -> np.ndarray   # (W+1,)
    def cell_area_km2(self) -> np.ndarray  # (H, W) spherical area, R = 6371 km
    def index_of(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]
        # cell indices by floor((lat - lat_min)/res); returns -1 where outside the box
    def to_dataset(self) -> xr.Dataset   # coords lat/lon + lat_bnds/lon_bnds/cell_area_km2
    def padded_shape(self, multiple: int) -> tuple[int, int]   # ceil to multiple

def grid_from_config(cfg: Config) -> Grid
def regular_bounds(centres: np.ndarray) -> np.ndarray
    # edges for a regular 1-D grid: midpoints, extrapolated half a step at both ends;
    # raise if spacing is not uniform within 1e-6 relative tolerance
def standardize_coords(ds: xr.Dataset) -> xr.Dataset
    # rename latitude/longitude/nav_lat/nav_lon/LATITUDE/LONGITUDE -> lat/lon,
    # convert lon 0..360 -> -180..180 (then sortby lon), sortby lat ascending,
    # drop a length-1 'depth' dim if present on surface products
```

Use `np.round(..., 6)` when constructing centres to avoid 5.1250000001 artefacts.

## `timeutil.py`

```python
def daily_index(cfg: Config) -> pd.DatetimeIndex            # dates.start .. dates.end, freq D
def doy_index(dates: pd.DatetimeIndex | np.ndarray) -> np.ndarray   # 0..365 month-day slots (root §5.2)
def split_of(dates, cfg) -> np.ndarray[str]                  # 'train' | 'val' | 'test' | 'gap' | 'none'
    # 'gap' = first splits.gap_days days of val and of test
def season_of(dates, cfg) -> np.ndarray[str]                 # from eval.seasons
def years_of_split(cfg, split: str) -> list[int]
def floor_to_date(times) -> pd.DatetimeIndex                 # UTC, normalize to midnight
```

`doy_index` reference values (put them in the test): 2016-01-01→0, 2016-02-29→59,
2016-03-01→60, 2017-02-28→58, 2017-03-01→60, 2016-12-31→365, 2017-12-31→365.

## `normalization.py`

```python
@dataclass(frozen=True)
class NormStats:
    in_mean: np.ndarray   # (C,) in config.inputs order
    in_std: np.ndarray    # (C,)
    out_mean: np.ndarray  # (D,) in config.depths_m order
    out_std: np.ndarray   # (D,)
    years_used: list[int]
def load_norm_stats(path: Path, cfg: Config) -> NormStats   # validates channel & depth keys
def normalize_x(x, s: NormStats, channel_axis: int)          # works on numpy and torch
def denormalize_y(y, s: NormStats, depth_axis: int)          # works on numpy and torch
def normalize_y(y, s: NormStats, depth_axis: int)
```

Broadcasting must be done by reshaping the stat vectors to the given axis; test with both
numpy arrays and torch tensors (torch imported lazily inside the function only if the input is
a tensor).

## `io.py`

```python
def atomic_write_path(final: Path) -> ContextManager[Path]
    # yields <final>.part (or <final>.tmp dir for .zarr); renames on success, deletes on exception
def write_netcdf(ds: xr.Dataset, path: Path) -> None     # zlib=True, complevel=4 for float vars; atomic
def write_zarr(ds: xr.Dataset, path: Path, chunks: dict) -> None   # via ds.chunk(chunks).to_zarr, atomic
def write_json(obj, path: Path) -> None                  # indent=2, sort_keys=True, atomic
def write_meta(directory: Path, cfg: Config, extra: dict | None = None) -> None   # root §5.7
def write_prediction_store(T_pred: np.ndarray, dates, cfg, model_id: str, split: str,
                           paths: Paths, attrs: dict | None = None) -> Path
    # validates: shape (len(dates), D, H, W), float32, °C plausibility (finite values within [-5, 40]),
    # NaN exactly outside depth_mask (sets it if not), then writes
def read_prediction_store(paths: Paths, model_id: str, split: str) -> xr.Dataset
def file_is_complete(path: Path) -> bool                 # exists, no .part sibling, opens with xarray
```

## `logging_utils.py`

```python
def get_logger(name: str) -> logging.Logger
def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None
    # format: "%(asctime)s %(levelname)s %(name)s: %(message)s", UTC timestamps
```

## `seeding.py`

```python
def seed_everything(seed: int) -> None   # random, numpy, torch (if importable), PYTHONHASHSEED
```

## `gitinfo.py`

```python
def git_hash(repo_root: Path) -> str     # "nogit" if not a repo or git missing; append "-dirty" if dirty
```

Use `subprocess.run([...], capture_output=True, check=False)`; never raise.

## Tests (in `tests/test_common_*.py`)

- config: valid config loads; each validation rule triggers a `ConfigError`; overrides typed
  correctly; `"3e-4"` string for `train.lr` is rejected.
- grid: default box → H=72, W=80; first/last centres 5.125/22.875 and 80.125/99.875;
  `regular_bounds` round trip; `index_of` edges (lat 5.0 → row 0, 23.0 → -1); cell areas
  positive and decreasing with latitude.
- timeutil: `doy_index` reference values; split counts for default config:
  train 1827, val 365−15=350, test 365−15=350, gap 30.
- normalization: round trip numpy and torch, arbitrary axis.
- io: atomic write leaves no file on exception; prediction store round trip on a tiny grid.

## Done when

`pytest tests -k common` passes on the authoring machine.
