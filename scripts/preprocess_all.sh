#!/usr/bin/env bash
# CPU stage: raw -> data/processed (all 9 steps; idempotent, pass --force via EXTRA).
set -euo pipefail
source "$(dirname "$0")/common.sh"
warn_if_gpu
check_disk 20
run "$PY" -m src.preprocess.run --config "$CONFIG" ${EXTRA:-}
