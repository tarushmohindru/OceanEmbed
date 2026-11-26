#!/usr/bin/env bash
# Sourced by the other scripts. Provides REPO_ROOT, CONFIG, PY, log, run, require_gpu,
# warn_if_gpu and check_disk. Contains no project logic.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG="${CONFIG:-config.yaml}"
PY="${PY:-python}"

_SCRIPT_NAME="$(basename "${BASH_SOURCE[1]:-script}" .sh)"
mkdir -p "$REPO_ROOT/results/logs"
LOG_FILE="$REPO_ROOT/results/logs/${_SCRIPT_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

log() {
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
  echo "$line" | tee -a "$LOG_FILE"
}

run() {
  log "RUN: $*"
  local start rc
  start=$(date +%s)
  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  set -e
  log "DONE (exit $rc, $(( $(date +%s) - start )) s): $*"
  if [[ $rc -ne 0 ]]; then
    log "FAILED with exit code $rc"
    exit "$rc"
  fi
}

have_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

require_gpu() {
  if ! have_gpu; then
    log "ERROR: no GPU found (nvidia-smi missing or failing). Switch the Studio to a GPU machine."
    exit 1
  fi
}

warn_if_gpu() {
  if have_gpu; then
    log "WARNING: a GPU machine is active but this stage is CPU-only; switch to a CPU machine to save credits."
  fi
}

check_disk() {
  local need_gb="$1" free_gb
  free_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
  if [[ "$free_gb" -lt "$need_gb" ]]; then
    log "ERROR: only ${free_gb} GB free under $REPO_ROOT, need ${need_gb} GB."
    exit 1
  fi
  log "disk ok: ${free_gb} GB free (need ${need_gb} GB)"
}
