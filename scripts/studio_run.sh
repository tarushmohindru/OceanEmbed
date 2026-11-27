#!/usr/bin/env bash
# Unattended end-to-end run on the Studio (resumable: every stage skips finished work).
# downloads (3 parallel streams) -> retry pass -> coverage -> preprocess -> baselines
# -> train -> evaluate.  Usage: nohup bash scripts/studio_run.sh > results/logs/studio_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
mkdir -p results/logs
stamp() { echo "$(date -u +%FT%TZ) $*"; }
stamp "downloads start"
python -m src.download.run --config config.yaml --products ccmp >> results/logs/dl_ccmp.log 2>&1 &
python -m src.download.run --config config.yaml --products glorys >> results/logs/dl_glorys.log 2>&1 &
python -m src.download.run --config config.yaml --products duacs ostia oscar smap argo >> results/logs/dl_sat.log 2>&1 &
wait
stamp "retry pass"
python -m src.download.run --config config.yaml --products all >> results/logs/dl_retry.log 2>&1 \
  || { stamp "DOWNLOADS FAILED - see results/logs/dl_retry.log"; exit 1; }
python -m src.download.check_coverage --config config.yaml || stamp "coverage gaps reported"
stamp "pipeline start"
bash scripts/after_download.sh && stamp "PIPELINE DONE"
