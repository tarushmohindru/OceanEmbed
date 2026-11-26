# OceanEmbed — Master Implementation Spec

This file is the entry point for the worker agent that writes the code. Read it completely
before opening any other `IMPLEMENTATION.md`. Every directory in this repository has its own
`IMPLEMENTATION.md` that specifies exactly what goes in that directory. This file holds the
rules that apply everywhere: conventions, the config schema, the environment, the build order
and the acceptance checks.

Background document: `OceanEmbed Implementation Guide (Draft v0.1).md` (repo root). This spec
**supersedes** the draft wherever they disagree; section 3 lists every intentional difference.

---

## 1. How to work through this repository

1. Read this file, then `src/IMPLEMENTATION.md`, then `src/common/IMPLEMENTATION.md`.
2. Implement directories in the build order of section 8. Do not skip ahead: later stages
   import earlier ones and depend on their file contracts.
3. For each directory: create every file listed in its `IMPLEMENTATION.md`, with the exact
   function signatures given, then the tests listed for it in `tests/IMPLEMENTATION.md`.
4. A stage is finished only when its **Done when** checklist passes. Unit tests must pass
   on the authoring machine (section 2). Data-dependent checks are run later on Lightning.
5. Keep the `IMPLEMENTATION.md` files. Do not delete or rewrite them; if you must deviate,
   record it in `docs/decisions_log.md` (format in `docs/IMPLEMENTATION.md`).
6. Never invent results. Any number in a README, write-up or slide must come from a file in
   `results/`. Use `TBD` placeholders until a real run produces the number.

### Tags used in these files

| Tag | Meaning |
| --- | --- |
| **Decision** | Fixed by the project deck. Do not change. |
| **Default** | Chosen for v1. Change only with a reason written to `docs/decisions_log.md`. |
| **Verify** | Must be confirmed by `src/download/probe.py` output or official docs before relying on it. Never hard-code a Verify value outside `config.yaml`. |
| **STOP** | Stop and ask the human user before continuing. |

---

## 2. Where code is written vs. where it runs

| Machine | OS | Used for |
| --- | --- | --- |
| Authoring machine (this one) | Windows 11 | Writing code, running unit tests on synthetic data. **No data download, no preprocessing, no training here.** |
| Lightning.ai Studio, CPU machine | Linux | Downloads, preprocessing, baselines (climatology, linear regression), evaluation, plots |
| Lightning.ai Studio, GPU machine | Linux | Training the CNN and ViT-CNN; generating their predictions |

Lightning Studio storage (~200 GB) persists when switching between CPU and GPU machines
(**Verify** in Lightning docs). Code moves from this machine to Lightning through git (push to a
private remote, pull on Lightning).

Consequences that every file must respect:

- **Line endings.** All `.sh` files must use LF. A `.gitattributes` file enforcing this exists
  at the repo root; do not remove it. A CRLF shell script fails on Linux with `$'\r': command not found`.
- **Paths.** Use `pathlib.Path` everywhere. Never build paths with string concatenation or
  backslashes. All paths in `config.yaml` are relative to the repo root unless absolute.
- **Heavy / Linux-only imports are lazy.** `xesmf`, `esmpy`, `copernicusmarine`, `earthaccess`,
  `argopy`, `cartopy` and `gsw` must be imported **inside the functions that use them**, never
  at module top level. `xesmf` cannot be installed on Windows; unit tests must still be able to
  import every module. Tests that need one of these packages call `pytest.importorskip(...)`.
- **No GPU assumptions.** Every script runs on CPU (slowly) if CUDA is absent.
- **Shell scripts are bash** and target Linux. Python entry points are the primary interface;
  shell scripts only chain them.

---

## 3. Differences from Draft v0.1 (all intentional)

| # | Draft v0.1 | This spec | Reason |
| --- | --- | --- | --- |
| 1 | Horizontal regridding with `xesmf` | Separable pure-numpy conservative regridding (`src/preprocess/regrid_core.py`); `xesmf` kept only for a cross-check test on Lightning | All sources are regular lat-lon grids, so conservative remapping is exact and separable; this removes the main install risk (`xesmf`/`esmpy` have no Windows builds) and makes regridding unit-testable on the authoring machine |
| 2 | Target depth "0 m" interpolated | 0 m = GLORYS shallowest level (0.494 m), used as-is | GLORYS has no 0 m level; interpolating to 0 m would be extrapolation, which the draft forbids |
| 3 | "No NaN in X after filling" via time interp + climatology | Adds spatial nearest-ocean fill for remaining ocean gaps, and land set to 0 after normalization | Time and climatology fills cannot fill land or permanently missing coastal SMAP cells |
| 4 | SSS QC range 25–38 PSU | 5–40 PSU | Northern Bay of Bengal river plume regularly drops below 25 PSU; those are real values |
| 5 | SST QC range 15–35 °C | 10–36 °C | Gross-error check only; Bay of Bengal never approaches 15 °C, so the old lower bound was not informative either way |
| 6 | Single dataset ID per product | Ordered list of "streams" per product (e.g. GLORYS `my` then `myint`); provenance recorded per day | Products switch from reprocessed to interim/NRT series; a switch inside the test year is a silent distribution shift and must be visible |
| 7 | Correlation over space and time only | Also anomaly correlation (ACC) against training climatology | Raw correlation is inflated by the static spatial pattern and seasonal cycle |
| 8 | ARGO: skip a depth if no measurement within 20 m | Depth-dependent bracketing-gap rule | Argo vertical sampling is ~50 m below ~400 m; a 20 m rule would drop most deep data |
| 9 | ARGO temperature compared directly | ARGO in-situ temperature converted to potential temperature with `gsw`; pressure converted to depth | GLORYS `thetao` is potential temperature; at 1000 m the difference is ~0.1 °C, comparable to model errors |
| 10 | GLORYS-vs-ARGO called "the practical limit" | Reported as an optimistic reference | GLORYS assimilates Argo profiles |
| 11 | Depth weights uniform only | Uniform default + `variance` and `thermocline` options | Per-depth normalization with uniform weights optimizes a different objective than the °C RMSE that is reported |
| 12 | Gulf of Thailand cells not mentioned | Ocean mask keeps only the largest connected ocean component | The 80–100°E box may clip disconnected basins at its eastern edge |
| 13 | `src/` flat with `datasets.py`, `train.py`, ... | `src/` split into sub-packages (`common`, `download`, `preprocess`, `dataset`, `baselines`, `models`, `training`, `evaluation`) | One `IMPLEMENTATION.md` per directory keeps each spec small and precise |
| 14 | Each model writes its own metrics | Every model writes a **prediction store** in one format; one `evaluate.py` scores all stores | Guarantees identical evaluation for all four models and the GLORYS reference |
| 15 | `torch` DataLoader | Whole split held on the GPU, batches by index | Data is ~1.3 GB; a DataLoader would be the bottleneck |
| 16 | Free Colab/Kaggle T4 | Lightning.ai: CPU machine for data, smallest GPU for training | User's platform; A100/H200 are unnecessary for a ~6M-parameter model |

---

## 4. Directory map

```text
ml_project/                       # repo root
  IMPLEMENTATION.md               # this file
  OceanEmbed Implementation Guide (Draft v0.1).md
  config.yaml                     # created by worker, exact content in section 6
  environment.yml                 # created by worker, section 7
  pyproject.toml                  # created by worker, section 7
  README.md                       # created by worker at the end (docs/IMPLEMENTATION.md)
  .gitignore  .gitattributes      # already present
  data/                           # data/IMPLEMENTATION.md
    raw/                          # data/raw/IMPLEMENTATION.md
    interim/                      # data/interim/IMPLEMENTATION.md
    processed/                    # data/processed/IMPLEMENTATION.md
  src/                            # src/IMPLEMENTATION.md
    common/                       # config, paths, grid, time, IO, logging, CLI helpers
    download/                     # probe + one downloader per product + coverage check
    preprocess/                   # grid, regrid, vertical, masks, QC, fill, normalize, tensors
    dataset/                      # splits, in-memory dataset, GPU batch iterator
    baselines/                    # climatology, linear regression
    models/                       # vit_cnn.py, cnn_unet.py, building blocks
    training/                     # losses, schedules, train loop, predict
    evaluation/                   # metrics, evaluate, ARGO, bootstrap, plots
  tests/                          # tests/IMPLEMENTATION.md
  scripts/                        # bash runners for Lightning
  notebooks/                      # exploration only
  results/                        # runs, predictions, metrics, figures, tables
  docs/                           # decisions log, write-up, viva notes
```

---

## 5. Global conventions (apply to every file)

### 5.1 Grid and array orientation

- Target grid: regular 0.25° lat-lon. Cell centres from `lat_min + res/2` to `lat_max - res/2`,
  same for longitude. Default box: lat 5.125 … 22.875 (H = 72), lon 80.125 … 99.875 (W = 80).
- Coordinate names everywhere: `lat`, `lon`, `time`, `depth`, `channel`. Every reader renames
  product-specific names (`latitude`, `nav_lat`, `longitude`, ...) to these immediately.
- Longitude range: −180 … 180. Convert 0 … 360 sources on read.
- **Latitude is ascending** (index 0 = southernmost row). Every reader sorts by `lat` and `lon`
  ascending. Plotting uses `origin="lower"` or explicit coordinates.
- Never rely on dimension order from a file. Always `transpose(...)` by name before converting
  to numpy (OSCAR, for one, may store `(time, lon, lat)` — **Verify**).
- Array dimension order in memory and on disk: `[time, channel, lat, lon]` for X,
  `[time, depth, lat, lon]` for Y.

### 5.2 Time

- Daily calendar from `dates.start` to `dates.end` inclusive: `pd.date_range(start, end, freq="D")`.
- Every daily value is labelled by its **date** (`datetime64[D]` / midnight), not by a source
  timestamp such as 12:00. Readers floor timestamps to the date.
- All times are UTC.
- Day-of-year for climatologies uses a **366-slot month-day calendar**: `doy_index(date)` returns
  0 … 365 where 1 Jan = 0, 29 Feb = 59, 1 Mar = 60 in **every** year (non-leap years skip slot 59).
  Implemented once in `src/common/timeutil.py`; never call `.dayofyear` directly.

### 5.3 Units (after reading, before anything else)

| Quantity | Unit | Notes |
| --- | --- | --- |
| SST, ocean temperature | °C | OSTIA `analysed_sst` is Kelvin: subtract 273.15 |
| SSS | PSU (practical salinity) | |
| SLA | m | |
| Currents, winds | m s⁻¹ | |
| Depth | m, positive down | Argo pressure (dbar) converted with `gsw.z_from_p` |

Every reader asserts the unit attribute it expects (from probe results) and converts.

### 5.4 Data types and missing values

- Floating point data on disk: `float32`. Masks: `uint8` (0/1) on disk, `bool` in memory.
- Missing value: `NaN` in interim and in `Y`. `X` contains no NaN (land = 0 after normalization).
- Never use fill values like −999 in arrays you write.

### 5.5 File formats

| Kind | Format | Writer |
| --- | --- | --- |
| Raw product subsets | NetCDF as delivered (cropped) | `src/download/*` |
| Interim per product per year | NetCDF4, zlib level 4 | `src/preprocess/*` |
| Final tensors | Zarr via `xarray.Dataset.to_zarr` only (never the `zarr` API directly, to avoid zarr v2/v3 API differences) | `src/preprocess/build_tensors.py` |
| Masks, grid, climatology | NetCDF | `src/preprocess/*` |
| Statistics, manifests, run metadata | JSON (indent 2, sorted keys) | various |
| Metrics, logs of epochs | CSV with header | `src/training`, `src/evaluation` |
| Figures | PNG at 200 dpi | `src/evaluation/plots.py` |

### 5.6 Entry points and CLI

- Every runnable module is executed from the repo root as
  `python -m src.<package>.<module> --config config.yaml [options]`.
- Every entry point uses `src.common.cli.base_parser()` (adds `--config` and repeatable
  `--set dotted.key=value` overrides) and `src.common.config.load_config(path, overrides)`.
- Every entry point is idempotent: re-running skips outputs that already exist and validate,
  unless `--force` is passed.
- Exit code 0 on success, non-zero with a clear message on failure. No bare `except:`.

### 5.7 Logging and provenance

- Use `src.common.logging_utils.get_logger(__name__)`; no `print` in library code (prints are
  allowed only in tests and notebooks).
- Every output directory created by a stage receives a `_meta.json` written by
  `src.common.io.write_meta(dir, extra)` containing: UTC timestamp, git commit hash (or `"nogit"`),
  config hash (SHA-256 of the resolved config YAML), hostname, Python version, and `extra`.

### 5.8 Randomness

- `src.common.seeding.seed_everything(seed)` seeds `random`, `numpy`, `torch` (CPU+CUDA).
- Bit-exact GPU determinism is not required (bilinear upsampling backward is non-deterministic);
  do not enable `torch.use_deterministic_algorithms`.

### 5.9 Code style

- Python 3.11, type hints on all public functions, docstrings stating shapes and units.
- Formatting: `ruff format`; lint: `ruff check` (config in `pyproject.toml`).
- Assertions about shapes in models and data code are required, not optional.
- No notebooks in the pipeline. No global mutable state except loggers.

---

## 6. `config.yaml` — create exactly this file

Values marked `# VERIFY` must be confirmed with `src/download/probe.py` (see
`src/download/IMPLEMENTATION.md`) and edited in this file only.

```yaml
# ---------------- project ----------------
project: oceanembed
seed: 42

paths:
  data_root: data            # raw/, interim/, processed/ live under this
  results_root: results

# ---------------- space & time ----------------
region:                      # Default (deck: "Bay of Bengal or Arabian Sea")
  name: bay_of_bengal
  lat_min: 5.0
  lat_max: 23.0
  lon_min: 80.0
  lon_max: 100.0
download_buffer_deg: 1.0     # extra margin downloaded around the box
grid_res: 0.25               # Decision

dates:                       # Default; quoted so YAML keeps them as strings
  start: "2016-01-01"
  end: "2022-12-31"

splits:                      # Default; by whole years. Inclusive year ranges.
  train: [2016, 2020]
  val: [2021, 2021]
  test: [2022, 2022]
  gap_days: 15               # first N days of val and test are dropped

depths_m: [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 400, 500, 700, 1000]  # Default
depth_bands_m:               # used for aggregated tables (inclusive membership by value)
  "0-100": [0, 10, 20, 30, 50, 75, 100]
  "100-300": [125, 150, 200, 300]
  "300-1000": [400, 500, 700, 1000]

# ---------------- inputs ----------------
inputs: [sst, sss, sla, u_cur, v_cur, u_wind, v_wind]   # Decision: order = channel order
channels:                    # channel -> (product, variable in raw file)
  sst:    {product: ostia, variable: analysed_sst}      # VERIFY variable name
  sss:    {product: smap,  variable: sss_smap}          # VERIFY variable name
  sla:    {product: duacs, variable: sla}               # VERIFY
  u_cur:  {product: oscar, variable: u}                 # VERIFY
  v_cur:  {product: oscar, variable: v}                 # VERIFY
  u_wind: {product: ccmp,  variable: uwnd}              # VERIFY
  v_wind: {product: ccmp,  variable: vwnd}              # VERIFY
extra_inputs:                # ablations only; all false in the core experiment
  land_mask: false
  doy_sincos: false
  latlon: false

# ---------------- products ----------------
# streams are tried in order; for each date the first stream that has it is used.
products:
  glorys:
    source: copernicusmarine
    streams:
      - {name: my,    dataset_id: cmems_mod_glo_phy_my_0.083deg_P1D-m}       # VERIFY
      - {name: myint, dataset_id: cmems_mod_glo_phy_myint_0.083deg_P1D-m}    # VERIFY
    variables: [thetao]
    depth_range_m: [0.0, 1100.0]
  ostia:
    source: copernicusmarine
    streams:
      - {name: rep, dataset_id: METOFFICE-GLO-SST-L4-REP-OBS-SST}            # VERIFY
      - {name: nrt, dataset_id: METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2}         # VERIFY
    variables: [analysed_sst]
  duacs:
    source: copernicusmarine
    streams:
      - {name: my,    dataset_id: cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D}     # VERIFY id and resolution
      - {name: myint, dataset_id: cmems_obs-sl_glo_phy-ssh_myint_allsat-l4-duacs-0.125deg_P1D}  # VERIFY
    variables: [sla]
  oscar:
    source: earthaccess
    streams:
      - {name: final,   short_name: OSCAR_L4_OC_FINAL_V2.0}                  # VERIFY coverage end
      - {name: interim, short_name: OSCAR_L4_OC_INTERIM_V2.0}                # VERIFY
    variables: [u, v]
  smap:
    source: earthaccess
    streams:
      - {name: rss, short_name: SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6}      # VERIFY; one product for the whole period
    variables: [sss_smap]
  ccmp:
    source: http
    streams:
      - {name: v3.1, url_template: "https://data.remss.com/ccmp/v03.1/Y{yyyy}/M{mm}/CCMP_Wind_Analysis_{yyyy}{mm}{dd}_V03.1_L4.nc"}  # VERIFY
    variables: [uwnd, vwnd]
  argo:
    source: argopy
    fetch_years: [2022]      # test years only
    argopy_src: erddap       # fallback: gdac
    max_depth_m: 1100

# ---------------- preprocessing ----------------
preprocess:
  qc_ranges:                 # values outside -> NaN, then gap-filled
    sst: [10.0, 36.0]
    sss: [5.0, 40.0]
    sla: [-1.5, 1.5]
    u_cur: [-3.0, 3.0]
    v_cur: [-3.0, 3.0]
    u_wind: [-40.0, 40.0]
    v_wind: [-40.0, 40.0]
  max_interp_gap_days: 3
  clim_window_days: 31       # centred, circular smoothing of day-of-year climatologies
  conservative_na_thres: 0.5 # min valid source-area fraction for a target cell to be valid
  ocean_valid_fraction: 0.99 # cell is ocean if valid on >= this fraction of days
  keep_largest_ocean_component: true
  glorys_top_level_max_m: 1.0  # shallowest GLORYS level (0.494 m) is used as "0 m" if shallower than this

# ---------------- model ----------------
model:
  name: vit_cnn              # vit_cnn | cnn_unet
  patch: 8                   # must be a power of two
  dim: 256
  depth: 6
  heads: 8
  mlp_ratio: 4
  dropout: 0.1
  stem_base_channels: 32     # skip stem channels = base * 2**i
  use_skip_stem: true        # ablation: false
cnn:
  base_channels: auto        # auto -> choose so params within 10% of vit_cnn

# ---------------- training ----------------
train:
  batch: 16
  lr: 3.0e-4                 # keep the ".0": PyYAML reads "3e-4" as a string
  wd: 0.05
  epochs: 100
  warmup_epochs: 5
  min_lr: 1.0e-6
  patience: 15
  grad_clip: 1.0
  grad_lambda: 0.1
  depth_weights: uniform     # uniform | thermocline | variance
  thermocline_weight: 2.0    # used when depth_weights == thermocline, for 50-200 m
  target_mode: absolute      # absolute | anomaly
  amp: auto                  # auto | bf16 | fp16 | off
  gpu_resident: true
  smoke: false               # true -> tiny subset, 2 epochs (see src/training)
  drop_inputs: []            # ablation: channel names removed from x (e.g. [sss])

# ---------------- evaluation ----------------
eval:
  seasons:                   # Default: Indian monsoon seasons (months)
    winter_monsoon: [12, 1, 2]
    pre_monsoon: [3, 4, 5]
    summer_monsoon: [6, 7, 8, 9]
    post_monsoon: [10, 11]
  map_depths_m: [0, 100, 500]
  bootstrap: {n: 1000, block_days: 30, ci: 0.95}
  argo:
    qc_ok: [1, 2]
    data_modes: [D, A, R]    # preference order
    surface_max_depth_m: 10.0
    max_bracket_gap_m:       # max distance between the two bracketing measurements
      - {upto_m: 100,  gap_m: 25}
      - {upto_m: 300,  gap_m: 50}
      - {upto_m: 700,  gap_m: 100}
      - {upto_m: 1000, gap_m: 200}
```

`src/common/config.py` validates this schema (see `src/common/IMPLEMENTATION.md`).

---

## 7. Environment and project files — create these

### 7.1 `environment.yml` (Linux / Lightning)

```yaml
name: oceanembed
channels: [conda-forge]
dependencies:
  - python=3.11
  - numpy
  - scipy
  - pandas
  - xarray
  - dask
  - netcdf4
  - h5netcdf
  - zarr
  - xesmf
  - esmpy
  - cf_xarray
  - cartopy
  - matplotlib
  - scikit-learn
  - gsw
  - pyyaml
  - tqdm
  - requests
  - pytest
  - ruff
  - pip
  - pip:
      - copernicusmarine
      - earthaccess
      - argopy
      # torch is installed separately, see below
```

Torch: on Lightning, a CUDA-enabled `torch` is normally preinstalled. If not, install the CUDA
wheel matching the machine with the selector command from pytorch.org. Never pin `torch` in
`environment.yml`. After setup, write `pip freeze > results/env_lock.txt` and
`conda env export > results/env_lock_conda.yml` (done by `scripts/setup_lightning.sh`).

### 7.2 Authoring machine (Windows) minimal environment

Only for unit tests: `python 3.11`, `numpy scipy pandas xarray netcdf4 zarr dask pyyaml torch
pytest ruff matplotlib scikit-learn`. Tests requiring `xesmf`, `gsw`, `argopy`, `cartopy` skip.

### 7.3 `pyproject.toml`

```toml
[project]
name = "oceanembed"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
  "data: needs data/processed (run on Lightning)",
  "gpu: needs CUDA",
  "network: needs internet and credentials",
  "slow: takes more than ~30 s on CPU",
]
addopts = "-m 'not data and not gpu and not network'"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "NPY"]
```

### 7.4 Already present

`.gitignore` and `.gitattributes` exist at the root. Keep them.

---

## 8. Build order and acceptance

| Step | Directory | Produces | Done when |
| --- | --- | --- | --- |
| 0 | root | `config.yaml`, `environment.yml`, `pyproject.toml`, `git init` | `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` works |
| 1 | `src/common` | config, paths, grid, time, IO, logging, CLI | its tests pass |
| 2 | `src/download` | `probe.py`, downloaders, `check_coverage.py` | tests pass; on Lightning: probe report reviewed (**STOP**: show the user `results/probe/probe_report.md` and agree the final period/streams before full download) |
| 3 | `src/preprocess` | all preprocessing modules | tests pass on synthetic data; on Lightning: sanity checks in `data/processed/IMPLEMENTATION.md` pass |
| 4 | `src/dataset` | splits, dataset, GPU iterator | leakage test and shape tests pass |
| 5 | `src/evaluation` (metrics + evaluate + plots core) | metrics on prediction stores | metric unit tests pass |
| 6 | `src/baselines` | climatology + linreg prediction stores | both scored by `evaluate.py` on val and test |
| 7 | `src/models` | `cnn_unet.py`, `vit_cnn.py` | shape tests and param-count printout pass |
| 8 | `src/training` | losses, train loop, predict | loss tests pass; smoke run (`train.smoke=true`) finishes on CPU in < 10 min with synthetic data |
| 9 | Lightning GPU | trained CNN and ViT-CNN, λ sweep | best checkpoints + prediction stores exist |
| 10 | `src/evaluation/argo.py` | ARGO matching for all models + GLORYS | `results/metrics/argo_*.csv` exist |
| 11 | `src/evaluation/plots.py`, `docs/` | figures, tables, README, write-up, viva notes | every checklist item in `docs/IMPLEMENTATION.md` ticked |

Model order on test RMSE expected by the draft: climatology > linreg > CNN ≥ ViT-CNN. If the
ViT-CNN does not beat the CNN, report it honestly; do not tune on the test set to change it.

---

## 9. Lightning runbook (summary; details in `scripts/IMPLEMENTATION.md`)

1. CPU Studio: `bash scripts/setup_lightning.sh` (env, credentials check, env lock).
2. CPU: `python -m src.download.probe` → **STOP** for user review of the probe report; after
   agreeing changes, edit `config.yaml` and `touch results/probe/APPROVED`.
3. CPU: `bash scripts/download_all.sh` (resumable; can take days; run under `nohup`/`tmux`).
4. CPU: `python -m src.download.check_coverage` → must report zero missing days.
5. CPU: `bash scripts/preprocess_all.sh`.
6. CPU: baselines + their evaluation (`bash scripts/run_baselines.sh`).
7. Switch to the smallest available GPU (T4/L4/A10G): `bash scripts/train_all.sh`.
8. CPU or GPU: `bash scripts/evaluate_all.sh` (GLORYS test + ARGO + figures).

Do not run downloads or preprocessing on a GPU machine (wasted credits).

---

## 10. Global "never" list

- Never compute statistics (normalization, climatology, ridge alpha, λ) using val or test years.
  Val may be used for model selection; test is touched once per final model.
- Never shuffle days across splits or split by pixel.
- Never fill or interpolate **targets** (Y). Missing targets stay NaN and are masked.
- Never use flips/rotations as augmentation.
- Never hard-code dataset IDs, variable names, depths, region or dates outside `config.yaml`.
- Never write into `data/raw/` from anything except `src/download/`.
- Never commit data, checkpoints or credentials (`.gitignore` handles this; do not weaken it).
