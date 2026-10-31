#!/usr/bin/env bash
# Budgeted, resumable Elo field runner for CLASSIFICATION.
# Runs all classification datasets under the 300s budget harness.
#
#   bash examples/elo_run_cla.sh
#   ELO_DATASETS=credit-g bash examples/elo_run_cla.sh
set -euo pipefail
export R_LIBS_USER=$HOME/R/library
LOG=benchmarks/elo_cla_run.log

PY_DEPS=(--with pandas --with pyarrow --with numpy --with lightgbm --with catboost --with ctboost --with chimeraboost --with interpret --with scikit-learn --with statsmodels --with mars-earth --with trust-free --with forust --with perpetual)

datasets=${ELO_DATASETS:-$(ls benchmarks/elo_cla_data/*_train.csv | sed 's|.*/||;s|_train.csv||' | sort | tr '\n' ',' | sed 's|,$||')}

{
  echo "=== budgeted classification field run: 300s per model per dataset ==="
  ELO_DATASETS="$datasets" uv run "${PY_DEPS[@]}" python -u examples/elo_score_all.py --task cla
  echo "=== board ==="
  uv run python examples/elo_report_cla.py | sed -n '/== Board (/,$p' | tail -25
} 2>&1 | tee -a "$LOG"
