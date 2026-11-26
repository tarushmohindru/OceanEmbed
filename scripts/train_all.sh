#!/usr/bin/env bash
# GPU stage (core only, 1-day budget): train CNN U-Net and ViT-CNN at the config lambda,
# then write val/test prediction stores. Finished runs are skipped; interrupted ones resume.
set -euo pipefail
source "$(dirname "$0")/common.sh"
have_gpu || log "WARNING: no GPU - training on CPU will be very slow"
RUNS="$REPO_ROOT/results/runs"
for m in cnn_unet vit_cnn; do
  name="${m}_core"
  if [[ -f "$RUNS/$name/summary.json" ]]; then
    log "$name already finished"
  elif [[ -f "$RUNS/$name/last.pt" ]]; then
    run "$PY" -m src.training.train --config "$CONFIG" --model "$m" --resume "$RUNS/$name"
  else
    run "$PY" -m src.training.train --config "$CONFIG" --model "$m" --run-name "$name"
  fi
  run "$PY" -m src.training.predict --config "$CONFIG" --run-dir "$RUNS/$name" --model-id "$m"
done
printf "{\n  \"cnn_unet\": \"cnn_unet_core\",\n  \"vit_cnn\": \"vit_cnn_core\"\n}\n" > "$RUNS/selected.json"
