#!/bin/sh
# Regenerate figures 1-3 from the repo root: sh experiments/run_figures_1_3.sh
# Resumable: rerunning skips runs already in figures/*.csv.
set -e
cd "$(dirname "$0")/.."
for f in fig3_stability fig1_cost_gap fig2_iterations; do
  echo "=== $f $(date '+%H:%M:%S')"
  "${PYTHON:-.venv/bin/python}" experiments/$f.py run plot
done
echo "=== done $(date '+%H:%M:%S')"
