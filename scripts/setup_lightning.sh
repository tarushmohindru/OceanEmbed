#!/usr/bin/env bash
# One-time setup of a Lightning Studio (run on the CPU machine).
#
#   bash scripts/setup_lightning.sh                 # default: install into the active environment
#   SETUP_MODE=conda bash scripts/setup_lightning.sh  # build a separate conda env "oceanembed"
#
# Default mode keeps Lightning's preinstalled (CUDA-enabled) torch, which also works after you
# switch the Studio to a GPU machine. xesmf/esmpy are optional (cross-check test only).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODE="${SETUP_MODE:-inplace}"
log "setup mode: $MODE (python: $($PY --version 2>&1))"

if [[ "$MODE" == "conda" ]]; then
  command -v conda >/dev/null || { log "ERROR: conda not found"; exit 1; }
  run conda env update -n oceanembed -f environment.yml --prune
  log "Activate it with: conda activate oceanembed   (then install torch, see IMPLEMENTATION.md 7.1)"
  PY="$(conda run -n oceanembed which python)"
else
  # pip wheels exist for everything except esmpy/xesmf, which are tried separately below.
  run $PY -m pip install --upgrade pip
  run $PY -m pip install numpy scipy pandas xarray dask netcdf4 h5netcdf zarr cf_xarray \
      matplotlib scikit-learn gsw pyyaml tqdm requests pytest ruff cartopy \
      copernicusmarine earthaccess argopy
  if ! $PY -c "import torch" 2>/dev/null; then
    log "torch not found; installing the default Linux wheel (CUDA-enabled)"
    run $PY -m pip install torch
  fi
  if command -v conda >/dev/null; then
    log "trying optional conda install of esmpy/xesmf (failure is only a warning)"
    conda install -y -c conda-forge esmpy xesmf >>"$LOG_FILE" 2>&1 \
      || log "WARNING: esmpy/xesmf not installed; the xesmf cross-check test will be skipped"
  else
    log "WARNING: no conda; xesmf cross-check test will be skipped"
  fi
fi

log "verifying imports"
$PY - <<'EOF' 2>&1 | tee -a "$LOG_FILE"
import importlib, sys
required = ["xarray", "zarr", "netCDF4", "scipy", "gsw", "copernicusmarine", "earthaccess",
            "argopy", "torch", "cartopy"]
optional = ["xesmf"]
bad = []
for name in required + optional:
    try:
        m = importlib.import_module(name)
        print(f"ok   {name} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"{'FAIL' if name in required else 'warn'} {name}: {type(e).__name__}: {e}")
        if name in required:
            bad.append(name)
import torch
print("cuda available:", torch.cuda.is_available())
sys.exit(1 if bad else 0)
EOF

log "checking credentials (secrets are never printed)"
$PY - "$CONFIG" <<'EOF' 2>&1 | tee -a "$LOG_FILE" || true
import sys
import pandas as pd
from src.common.config import load_config
from src.download.common import check_credentials

cfg = load_config(sys.argv[1])
for product in ("glorys", "oscar", "ccmp"):
    try:
        check_credentials(product)
        print(f"ok   credentials for {product}")
    except RuntimeError as e:
        print(f"MISSING {e}")

try:
    import copernicusmarine
    ds_id = cfg["products"]["glorys"]["streams"][0]["dataset_id"]
    cat = copernicusmarine.describe(dataset_id=ds_id)
    n = sum(len(p.datasets) for p in cat.products)
    print(f"{'ok  ' if n else 'FAIL'} copernicusmarine describe({ds_id}) -> {n} dataset(s)")
except Exception as e:  # noqa: BLE001
    print(f"FAIL copernicusmarine describe: {type(e).__name__}: {e}")

try:
    import earthaccess
    auth = earthaccess.login(strategy="environment")
    print("ok   earthaccess login" if getattr(auth, "authenticated", False)
          else "FAIL earthaccess login (set EARTHDATA_USERNAME / EARTHDATA_PASSWORD)")
except Exception as e:  # noqa: BLE001
    print(f"FAIL earthaccess login: {type(e).__name__}: {e}")

try:
    import requests
    from src.download.ccmp import url_for
    stream = cfg["products"]["ccmp"]["streams"][0]
    url = url_for(cfg, stream, pd.Timestamp(cfg["dates"]["start"]))
    code = requests.head(url, allow_redirects=True, timeout=(30, 60)).status_code
    print(f"{'ok  ' if code == 200 else 'WARN'} CCMP HEAD {code} for {url}")
    if code in (401, 403):
        print("STOP: RSS requires registration; ask the user before continuing.")
except Exception as e:  # noqa: BLE001
    print(f"FAIL CCMP HEAD: {type(e).__name__}: {e}")
EOF

run $PY -m pytest -q

log "writing environment locks"
mkdir -p results
$PY -m pip freeze > results/env_lock.txt
if command -v conda >/dev/null; then
  conda env export > results/env_lock_conda.yml 2>/dev/null || log "WARNING: conda env export failed"
fi
log "setup finished. Next: $PY -m src.download.probe --config $CONFIG"
