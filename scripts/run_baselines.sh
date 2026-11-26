#!/usr/bin/env bash
# CPU stage: climatology + per-cell ridge baselines (val and test stores).
set -euo pipefail
source "$(dirname "$0")/common.sh"
run "$PY" -m src.baselines.climatology --config "$CONFIG"
run "$PY" -m src.baselines.linreg --config "$CONFIG"
