#!/usr/bin/env bash
# Run the Elo field FROM SCRATCH in groups of 5 datasets (13 TabArena
# regression + 5 RuleQuest = 18). After each group, benchmarks/elo_scores.csv
# and `elo_report.py` are a complete, shareable intermediate result - rerun
# the report any time to see the board so far.
#
#   bash examples/elo_run_groups.sh
set -e
export R_LIBS_USER=$HOME/R/library

PY_DEPS=(--with pandas --with numpy --with lightgbm --with catboost --with ctboost --with scikit-learn --with mars-earth --with forust --with perpetual)
RS_DEPS=(--with pandas --with numpy --with scikit-learn --with forust --with perpetual)

echo "dataset,model,rmse,mae,fit_s" > benchmarks/elo_scores.csv

groups=(
  "abalone,boston,concrete,fc2008,ozone"
  "ta_airfoil,ta_concrete,ta_diamonds,ta_fiat500,ta_food_delivery"
  "ta_healthcare,ta_houses,ta_miami,ta_protein,ta_qsar_fish"
  "ta_qsar_tid11,ta_superconductivity,ta_wine"
)
for g in "${groups[@]}"; do
  echo "=== group: $g ==="
  ELO_DATASETS="$g" Rscript examples/elo_score_r.R
  ELO_DATASETS="$g" uv run "${PY_DEPS[@]}" python examples/elo_score_python.py
  ELO_DATASETS="$g" uv run "${RS_DEPS[@]}" python examples/elo_score_rust.py
  echo "--- board after this group ---"
  uv run python examples/elo_report.py | sed -n '/== Elo (/,$p'
done
echo "=== all groups done ==="
