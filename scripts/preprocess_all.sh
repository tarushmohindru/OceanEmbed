#!/usr/bin/env bash
# CPU stage: raw -> data/processed (all 9 steps; idempotent, pass --force via EXTRA).
# Step 2 (regrid) runs one process per year in parallel; the full run then skips done years.
set -euo pipefail
source "$(dirname "$0")/common.sh"
warn_if_gpu
check_disk 20
YEARS=$("$PY" -c "from src.common.config import load_config as l; c=l('$CONFIG'); \
print(*range(int(str(c['dates']['start'])[:4]), int(str(c['dates']['end'])[:4]) + 1))")
run "$PY" -m src.preprocess.run --config "$CONFIG" --steps grid ${EXTRA:-}
pids=()
for y in $YEARS; do
  "$PY" -m src.preprocess.regrid --config "$CONFIG" --years "$y" ${EXTRA:-} \
    > "$REPO_ROOT/results/logs/regrid_$y.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ $fail -ne 0 ]]; then
  log "ERROR: a parallel regrid failed - see results/logs/regrid_<year>.log"
  exit 1
fi
log "parallel regrid done for $YEARS"
run "$PY" -m src.preprocess.run --config "$CONFIG" ${EXTRA:-}
