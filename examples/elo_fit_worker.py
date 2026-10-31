"""Fit one (algorithm x dataset) as a secondary command-line process.

The orchestrator runs this under examples/elo_budget.run_with_budget so the
whole process group dies at the wall-clock budget.

Usage:
    python examples/elo_fit_worker.py <task> <dataset> <model>
    (task is 'reg' or 'cla'; if omitted, defaults to 'reg')

Outputs exactly one machine-parseable line:
    Regression:     RESULT <dataset>,<model>,<rmse>,<mae>,<fit_s>
    Classification: RESULT <dataset>,<model>,<logloss>,<accuracy>,<bal_acc>,<auc>,<macro_f1>,<fit_s>
"""

import sys
from pathlib import Path

# Ensure examples/ is importable
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from elo_common import read_split, regression_scores
from cla_scores import score_cla
import elo_models_reg
import elo_models_cla


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python elo_fit_worker.py [task] <dataset> <model>")
    
    if len(sys.argv) == 3:
        task, ds, name = "reg", sys.argv[1], sys.argv[2]
    else:
        task, ds, name = sys.argv[1], sys.argv[2], sys.argv[3]

    if task in ("reg", "regression"):
        data_dir = Path("benchmarks/elo_data")
        xtr, ytr, xte, yte, meta = read_split(data_dir, ds, is_classification=False)
        models = elo_models_reg.make_models(meta)
        if name not in models:
            raise SystemExit(f"Unknown regression model: {name}")
        
        pred, fit_s = models[name](xtr, ytr, xte)
        rmse, mae = regression_scores(pred, yte)
        print(f"RESULT {ds},{name},{rmse:.4f},{mae:.4f},{fit_s:.3f}", flush=True)

    elif task in ("cla", "classification"):
        data_dir = Path("benchmarks/elo_cla_data")
        xtr, ytr, xte, yte, meta = read_split(data_dir, ds, is_classification=True)
        models = elo_models_cla.make_models(meta)
        if name not in models:
            raise SystemExit(f"Unknown classification model: {name}")

        proba, fit_s = models[name](xtr, ytr, xte)
        sc = score_cla(proba, yte, meta["classes"])
        auc_str = f"{sc['auc']:.4f}" if sc.get("auc") is not None else ""
        print(
            f"RESULT {ds},{name},{sc['logloss']:.4f},{sc['accuracy']:.4f},"
            f"{sc['bal_acc']:.4f},{auc_str},{sc['macro_f1']:.4f},{fit_s:.3f}",
            flush=True,
        )
    else:
        raise SystemExit(f"Unknown task: {task} (expected 'reg' or 'cla')")


if __name__ == "__main__":
    main()
