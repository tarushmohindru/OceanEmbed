# `scripts/` — Bash runners for Lightning.ai

Purpose: chain the Python entry points into the stages of the runbook, with logging and safe
re-runs. **No logic lives here** beyond ordering, logging and stopping on errors; all behaviour is
in `src/`.

Target: Linux bash on a Lightning Studio. Files must be LF (`.gitattributes` enforces it) and
executable (`git update-index --chmod=+x scripts/*.sh` from Windows, since `chmod` does not
record the bit there).

## Files

```text
scripts/
  common.sh            # sourced by the others
  setup_lightning.sh   # environment + credential checks + env lock
  download_all.sh      # probe check -> downloads -> coverage
  preprocess_all.sh    # src.preprocess.run + sanity
  run_baselines.sh     # climatology, linreg, glorys reference + their evaluation
  train_all.sh         # model summary, lambda sweep, core, seeds, ablations
  evaluate_all.sh      # evaluate val/test, argo, tables, plots
  run_all.sh           # everything after setup, in order (for a full rebuild)
  archive_results.sh   # pack deliverables for hand-in
```

## Conventions for every script

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
```

`common.sh` provides:

- `REPO_ROOT` (parent of `scripts/`), `cd "$REPO_ROOT"`.
- `CONFIG="${CONFIG:-config.yaml}"` (override with the environment variable).
- `PY="${PY:-python}"`.
- `log()` — timestamped echo to stdout and to `results/logs/<script>_<timestamp>.log`.
- `run()` — logs the command, runs it, logs duration; on failure logs the exit code and exits.
- `require_gpu()` — fails with a clear message if `nvidia-smi` is missing (used by `train_all.sh`).
- `warn_if_gpu()` — prints a warning when a CPU-only stage is started on a GPU machine
  (credits are wasted).
- `check_disk(min_gb)` — fails if free space under `REPO_ROOT` is below `min_gb`.

All scripts accept extra arguments and pass them through to the Python entry points where it
makes sense (e.g. `--force`).

## Script contents

### `setup_lightning.sh` (CPU machine, once per Studio)

1. Create/update the conda env from `environment.yml` (`conda env update -f environment.yml
   --prune`) or, if Lightning's preinstalled environment is used, `pip install` the pip-only
   packages and `conda install -c conda-forge` the rest. Print which path was taken.
2. Verify imports: `xarray, zarr, netCDF4, scipy, gsw, copernicusmarine, earthaccess, argopy,
   torch, cartopy, xesmf` (xesmf failure → warning only; it is used only for a cross-check).
3. Credential checks without printing secrets: `copernicusmarine` (a `describe` call on the GLORYS
   dataset id), Earthdata (`earthaccess.login(strategy="environment")`), HTTP HEAD to the CCMP URL
   for `dates.start`.
4. `pytest -q` (unit tests) must pass.
5. Write `results/env_lock.txt` (`pip freeze`) and `results/env_lock_conda.yml`.

### `download_all.sh` (CPU)

1. `check_disk 80`.
2. If `results/probe/probe.json` is missing: run `src.download.probe`, print the path of
   `probe_report.md`, and **exit 2 with the message "review the probe report with the user, update
   config.yaml, then re-run"**.
3. If `results/probe/APPROVED` does not exist: exit 2 with the same message. (The file is created
   by hand after the review: `touch results/probe/APPROVED`.)
4. `run $PY -m src.download.run --config "$CONFIG" --products all "$@"`.
5. `run $PY -m src.download.check_coverage --config "$CONFIG"`.

Recommend in the header comment: run inside `tmux` or with `nohup ... &` because downloads take
hours to days.

### `preprocess_all.sh` (CPU)

`check_disk 20`; `warn_if_gpu`; `run $PY -m src.preprocess.run --config "$CONFIG" "$@"`
(includes sanity). Then `pytest -m data -q`.

### `run_baselines.sh` (CPU)

```bash
run $PY -m src.baselines.climatology --config "$CONFIG"
run $PY -m src.baselines.linreg --config "$CONFIG" --variant per_cell
run $PY -m src.baselines.glorys_reference --config "$CONFIG"
run $PY -m src.evaluation.evaluate --config "$CONFIG" --split val
run $PY -m src.evaluation.evaluate --config "$CONFIG" --split test
```

### `train_all.sh` (GPU)

```bash
require_gpu
run $PY -m src.models.summary --config "$CONFIG"
run $PY -m src.training.sweep --config "$CONFIG" --set-name lambda
run $PY -m src.training.sweep --config "$CONFIG" --set-name core
# optional, controlled by env flags so a short session can stop after core:
[[ "${RUN_SEEDS:-1}" == 1 ]] && run $PY -m src.training.sweep --config "$CONFIG" --set-name seeds
[[ "${RUN_ABLATIONS:-1}" == 1 ]] && run $PY -m src.training.sweep --config "$CONFIG" --set-name ablations
```

Sweeps skip finished runs, so re-running after a session ends resumes the work.

### `evaluate_all.sh` (CPU)

`evaluate --split val`, `evaluate --split test`, `argo --split test`, `tables`, `plots`.

### `run_all.sh`

Calls `download_all.sh` (skips quickly if complete), `preprocess_all.sh`, `run_baselines.sh`,
then prints "switch to a GPU machine and run scripts/train_all.sh, then scripts/evaluate_all.sh"
and exits. It does not try to switch machines.

### `archive_results.sh`

Creates `deliverables/oceanembed_results_<date>.tar.gz` containing: `config.yaml`,
`data/processed/norm_stats.json`, `data/processed/masks.nc`, `results/tables/`, `results/figures/`,
`results/metrics/`, the `best.pt` + `config.yaml` + `metrics.csv` + `summary.json` of the selected
runs (from `results/runs/selected.json`), `results/env_lock*`, and `docs/`. Prints its size.

## Done when

`bash -n scripts/*.sh` (syntax check) passes on Lightning, and `file scripts/*.sh` reports no
CRLF line terminators.
