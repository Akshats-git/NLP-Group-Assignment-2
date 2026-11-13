#!/usr/bin/env bash
# Start the Q2 demo at http://localhost:8501. Works offline once the checkpoints are installed.
#   Question-2/demo/run_demo.sh                 light theme (best on a projector)
#   Question-2/demo/run_demo.sh --theme.base dark
set -euo pipefail
cd "$(dirname "$0")/.."
exec ../.venv/bin/streamlit run demo/app.py \
  --browser.gatherUsageStats false \
  --client.toolbarMode viewer \
  --server.runOnSave false \
  --server.fileWatcherType none \
  --theme.base light \
  "$@"
