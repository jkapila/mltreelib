"""Tune ModelTree hyperparameters on real data with the generic optimizer.

The script reads a local Parquet file from `datasets/`, splits it into a train
and a holdout part, and minimizes holdout RMSE over bounded tree
hyperparameters with `mltreelib.solvers.optimize`.

Random Search and Latin hypercube sampling receive the same evaluation budget,
so the two baselines compare on equal cost. The script prints one JSON payload
with the tuned parameters, the holdout RMSE, the baseline RMSE, and the
termination status of the run.

Run it from the repository root:

    uv run python examples/optimize_generic.py
    uv run python examples/optimize_generic.py --dataset airfoil_self_noise --budget 60
"""

from __future__ import annotations

import argparse
import json
from math import sqrt

from mltreelib import ModelTree
from mltreelib.datasets import load_dataset
from mltreelib.solvers import optimize

# Bounded search space: tree depth, minimum rows per leaf, and leaf L2 penalty.
BOUNDS = [(1.0, 6.0), (2.0, 60.0), (0.0, 1.0)]


def load_numeric_table(name: str, offline: bool):
    """Read one benchmark dataset into numeric features and one numeric target.

    Categorical columns are dropped. The returned names list holds the kept
    feature columns only. The dataset is downloaded on demand unless the run
    stays offline.
    """
    data = load_dataset(name, auto_download=not offline)
    columns = data["columns"]
    dtypes = data["dtypes"]
    target = list(columns)[-1]
    if dtypes[target] != "numeric":
        raise TypeError(f"target column {target} is not numeric")
    values = columns.pop(target)
    names = [name for name in columns if dtypes[name] == "numeric"]
    rows = []
    for index in range(len(values)):
        row = []
        for column in names:
            value = columns[column][index]
            if value is None:
                raise ValueError(f"column {column} has a missing value")
            row.append(float(value))
        rows.append(row)
    y = [float(value) for value in values]
    return names, rows, y


def rmse(model, x, y) -> float:
    """Root mean squared error of one model on one dataset."""
    predictions = model.predict(x)
    total = sum((prediction - target) ** 2 for prediction, target in zip(predictions, y))
    return sqrt(total / len(y))


def fit_tree(x, y, max_depth: int, min_rows: int, l2: float, seed: int) -> ModelTree:
    """Fit one regression tree with the given hyperparameters."""
    model = ModelTree(
        max_depth=max_depth,
        min_rows=min_rows,
        l2=l2,
        leaf="linear",
        leaf_solver="irls",
        seed=seed,
    )
    model.fit(x, y)
    return model


def run_method(method: str, x_train, y_train, x_test, y_test, budget: int, seed: int) -> dict:
    """Spend one evaluation budget on one method and report the outcome."""
    calls = {"count": 0}

    def objective(parameters):
        calls["count"] += 1
        max_depth = round(parameters[0])
        min_rows = round(parameters[1])
        l2 = parameters[2]
        model = fit_tree(x_train, y_train, max_depth, min_rows, l2, seed)
        return rmse(model, x_test, y_test)

    result = optimize(
        objective,
        BOUNDS,
        method=method,
        samples=budget,
        max_evaluations=budget,
        seed=seed,
    )
    return {
        "method": method,
        "status": result.status,
        "evaluations": result.evaluations,
        "objective_calls": calls["count"],
        "best_rmse": result.best_value,
        "best_parameters": {
            "max_depth": round(result.best_x[0]),
            "min_rows": round(result.best_x[1]),
            "l2": result.best_x[2],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="airfoil_self_noise")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never download; fail when the dataset is missing locally.",
    )
    args = parser.parse_args()

    names, x, y = load_numeric_table(args.dataset, args.offline)
    cut = int(len(y) * (1.0 - args.holdout))
    x_train, y_train = x[:cut], y[:cut]
    x_test, y_test = x[cut:], y[cut:]

    baseline = fit_tree(x_train, y_train, max_depth=3, min_rows=15, l2=0.001, seed=args.seed)
    payload = {
        "dataset": args.dataset,
        "rows": len(y),
        "features": names,
        "train_rows": len(y_train),
        "holdout_rows": len(y_test),
        "budget": args.budget,
        "seed": args.seed,
        "baseline_rmse": rmse(baseline, x_test, y_test),
        "methods": [
            run_method(
                method,
                x_train,
                y_train,
                x_test,
                y_test,
                args.budget,
                args.seed,
            )
            for method in ("random_search", "latin_hypercube")
        ],
    }
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
