#!/usr/bin/env bash
# Start the Streamlit viewer on port ${PORT:-8501} (expose it via the Studio "Port viewer").
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec "$PY" -m streamlit run app/streamlit_app.py --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 --server.headless true -- --config "$CONFIG"
