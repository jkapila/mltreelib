#!/usr/bin/env bash
# Budgeted, resumable Elo field runner.
#   - loop order: DATASETS outer, algorithms inner (a big dataset that fails
#     many models completes before the next dataset starts)
#   - per-model budget: 300s hard cap (algorithm x dataset)
#   - skips already-scored (dataset, model) pairs - safe to re-run
#   - live progress log: tail -f benchmarks/elo_run.log
#
#   bash examples/elo_run_resume.sh                    # everything missing
#   ELO_DATASETS=ta_healthcare,ta_protein bash examples/elo_run_resume.sh
set -u
export R_LIBS_USER=$HOME/R/library
LOG=benchmarks/elo_run.log
PY_DEPS=(--with pandas --with numpy --with lightgbm --with catboost --with ctboost --with scikit-learn --with mars-earth --with forust --with perpetual)
RS_DEPS=(--with pandas --with numpy --with scikit-learn --with forust --with perpetual)

datasets=${ELO_DATASETS:-$(ls benchmarks/elo_data/*_train.csv | sed 's|.*/||;s|_train.csv||' | sort | tr '\n' ',' | sed 's|,$||')}
IFS=',' read -ra DS <<< "$datasets"

{
  echo "=== budgeted field run: 180s per model per dataset, skip-if-scored ==="
  ELO_DATASETS="$datasets" uv run "${PY_DEPS[@]}" python -u examples/elo_score_all.py
  echo "=== board ==="
  uv run python examples/elo_report.py | sed -n '/== Elo (/,$p' | tail -22
} 2>&1 | tee -a "$LOG"
