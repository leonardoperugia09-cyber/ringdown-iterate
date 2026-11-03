#!/usr/bin/env bash
# Build and execute notebooks inside WSL (Ubuntu), from the project root:
#   bash run.sh 01_get_data            # one notebook
#   bash run.sh all                    # Steps 1-4, then the scan, then Step 5 plots
# Each notebooks/NN_name.py (jupytext "percent" format) becomes an executed
# notebooks/NN_name.ipynb with all outputs and plots saved inside it.
set -euo pipefail
VENV=${VENV:-/root/venvs/ringdown-iterate}
source "$VENV/bin/activate"
cd "$(dirname "$0")/notebooks"
export MPLBACKEND=Agg PYTHONWARNINGS=ignore

run_nb () {
  echo "=== $1 ($(date +%T))"
  jupytext --to ipynb --set-kernel python3 --execute "$1.py"
}

# one fresh process per start time (memory grows across fits in one process),
# then a final pass that writes the summary table
scan_loop () {
  while python 05a_run_scan.py next; do :; done
  python 05a_run_scan.py
}

if [ "${1:-all}" = all ]; then
  for nb in 01_get_data 02_noise 03_docs_example 04_compare_paper; do run_nb "$nb"; done
  scan_loop
  run_nb 05_start_time_scan
else
  for nb in "$@"; do
    if [ "$nb" = 05a_run_scan ]; then scan_loop; else run_nb "$nb"; fi
  done
fi
