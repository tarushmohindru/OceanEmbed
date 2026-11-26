#!/usr/bin/env bash
# CPU stage: score all stores on val and test.
set -euo pipefail
source "$(dirname "$0")/common.sh"
run "$PY" -m src.evaluation.evaluate --config "$CONFIG" --split val
run "$PY" -m src.evaluation.evaluate --config "$CONFIG" --split test
