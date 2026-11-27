#!/usr/bin/env bash
# Unattended end-to-end run on the Studio (resumable: every stage skips finished work).
# downloads (GLORYS one process per year + the rest) -> wait for ALL download processes
# -> retry pass -> coverage -> preprocess -> baselines -> train -> evaluate.
# Usage: nohup bash scripts/studio_run.sh > results/logs/studio_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source ./.env; set +a
mkdir -p results/logs
stamp() { echo "$(date -u +%FT%TZ) $*"; }
DL="python -m src.download.run --config config.yaml"
YEARS=$(python -c "from src.common.config import load_config as l; c=l('config.yaml'); \
print(*range(int(str(c['dates']['start'])[:4]), int(str(c['dates']['end'])[:4]) + 1))")

stamp "downloads start (GLORYS years: $YEARS)"
for y in $YEARS; do
  $DL --products glorys --years "$y" >> "results/logs/dl_glorys_$y.log" 2>&1 &
done
# OSCAR/SMAP are left to the retry pass if another process is already fetching them.
OTHERS="ccmp duacs ostia argo"
pgrep -f "products oscar" >/dev/null || OTHERS="$OTHERS oscar"
pgrep -f "products smap" >/dev/null || OTHERS="$OTHERS smap"
$DL --products $OTHERS >> results/logs/dl_sat.log 2>&1 &
wait
while pgrep -f "src.download.run" >/dev/null; do sleep 30; done

stamp "retry pass"
$DL --products all >> results/logs/dl_retry.log 2>&1 \
  || { stamp "DOWNLOADS FAILED - see results/logs/dl_retry.log"; exit 1; }
python -m src.download.check_coverage --config config.yaml || stamp "coverage gaps reported"
stamp "pipeline start"
bash scripts/after_download.sh && stamp "PIPELINE DONE"
