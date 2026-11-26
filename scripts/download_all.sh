#!/usr/bin/env bash
# Probe check -> downloads -> coverage report (CPU machine).
# Downloads take hours to days: run inside tmux, or  nohup bash scripts/download_all.sh &
# Extra arguments (e.g. --force, --years 2016 2017) are passed to src.download.run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

warn_if_gpu
check_disk 80

MSG="review the probe report with the user, update config.yaml, then re-run"
if [[ ! -f results/probe/probe.json ]]; then
  run $PY -m src.download.probe --config "$CONFIG"
  log "probe report: $REPO_ROOT/results/probe/probe_report.md"
  log "$MSG"
  exit 2
fi
if [[ ! -f results/probe/APPROVED ]]; then
  log "probe report: $REPO_ROOT/results/probe/probe_report.md"
  log "$MSG  (after review: touch results/probe/APPROVED)"
  exit 2
fi

run $PY -m src.download.run --config "$CONFIG" --products all "$@"
run $PY -m src.download.check_coverage --config "$CONFIG"
