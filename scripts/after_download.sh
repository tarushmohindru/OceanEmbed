#!/usr/bin/env bash
# Everything after the downloads: preprocess -> baselines -> train -> evaluate.
set -euo pipefail
DIR="$(dirname "$0")"
bash "$DIR/preprocess_all.sh"
bash "$DIR/run_baselines.sh"
bash "$DIR/train_all.sh"
bash "$DIR/evaluate_all.sh"
