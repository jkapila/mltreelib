"""Optimization Tracking Benchmark across TabArena datasets.

Tracks fit runtime, leaf counts, and prediction accuracy across incremental
tree growth optimizations on both regression and classification tasks,
benchmarked directly against official historic Elo scores.

Datasets:
  - Regression: 'houses' (20,640 rows, 8 features, target LnMedianHouseValue)
  - Classification (Large): 'bank-marketing' (45,211 rows, 13 features, target SubscribeTermDeposit)
  - Classification (Elo-tracked): 'credit-g' (1,000 rows, 20 features, target good_or_bad_customer)
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

from mltreelib import ModelTree
from mltreelib.datasets import load_dataset

HISTORIC_ELO_BASELINES = [
    # Regression: houses (from benchmarks/elo_scores.csv)
    {"tag": "Elo-ModelTree(all)", "dataset": "houses", "task": "regression", "rows": 20640, "cols": 8, "fit_ms": 24329.0, "leaves": 27, "splits": 26, "metric_name": "rmse", "metric_val": 0.2574},
    {"tag": "Elo-ModelTree(mean)", "dataset": "houses", "task": "regression", "rows": 20640, "cols": 8, "fit_ms": 467.0, "leaves": 25, "splits": 24, "metric_name": "rmse", "metric_val": 0.3585},
    {"tag": "Elo-Cubist", "dataset": "houses", "task": "regression", "rows": 20640, "cols": 8, "fit_ms": 2692.0, "leaves": 0, "splits": 0, "metric_name": "rmse", "metric_val": 0.2298},
    {"tag": "Elo-rpart", "dataset": "houses", "task": "regression", "rows": 20640, "cols": 8, "fit_ms": 411.0, "leaves": 0, "splits": 0, "metric_name": "rmse", "metric_val": 0.3779},
    {"tag": "Elo-LightGBM", "dataset": "houses", "task": "regression", "rows": 20640, "cols": 8, "fit_ms": 300.0, "leaves": 0, "splits": 0, "metric_name": "rmse", "metric_val": 0.2162},
    {"tag": "Elo-CatBoost", "dataset": "houses", "task": "regression", "rows": 20640, "cols": 8, "fit_ms": 605.0, "leaves": 0, "splits": 0, "metric_name": "rmse", "metric_val": 0.2193},

    # Classification: credit-g (from benchmarks/elo_cla_scores.csv)
    {"tag": "Elo-ctree", "dataset": "credit-g", "task": "classification", "rows": 1000, "cols": 20, "fit_ms": 1573.0, "leaves": 0, "splits": 0, "metric_name": "accuracy", "metric_val": 0.7350},
    {"tag": "Elo-glmtree", "dataset": "credit-g", "task": "classification", "rows": 1000, "cols": 20, "fit_ms": 1590.0, "leaves": 0, "splits": 0, "metric_name": "accuracy", "metric_val": 0.7350},
    {"tag": "Elo-rpart", "dataset": "credit-g", "task": "classification", "rows": 1000, "cols": 20, "fit_ms": 62.0, "leaves": 0, "splits": 0, "metric_name": "accuracy", "metric_val": 0.7250},
    {"tag": "Elo-CatBoost", "dataset": "credit-g", "task": "classification", "rows": 1000, "cols": 20, "fit_ms": 1192.0, "leaves": 0, "splits": 0, "metric_name": "accuracy", "metric_val": 0.7100},
    {"tag": "Elo-LightGBM", "dataset": "credit-g", "task": "classification", "rows": 1000, "cols": 20, "fit_ms": 237.0, "leaves": 0, "splits": 0, "metric_name": "accuracy", "metric_val": 0.7250},
]


def seed_historic_baselines(csv_path: Path):
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tag", "dataset", "task", "rows", "cols", "fit_ms", "leaves", "splits", "metric_name", "metric_val"
            ])
            writer.writeheader()
            writer.writerows(HISTORIC_ELO_BASELINES)
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        existing = f.read()

    to_add = [row for row in HISTORIC_ELO_BASELINES if row["tag"] not in existing]
    if to_add:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tag", "dataset", "task", "rows", "cols", "fit_ms", "leaves", "splits", "metric_name", "metric_val"
            ])
            writer.writerows(to_add)


def run_benchmark(tag: str = "current") -> list[dict]:
    csv_path = Path("benchmarks/optimization_tracker.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    seed_historic_baselines(csv_path)

    results = []

    # -------------------------------------------------------------
    # 1. Regression: houses (20,640 rows x 8 predictors)
    # -------------------------------------------------------------
    print("\n[1/3] Running regression benchmark: houses (20,640 rows)...")
    data_reg = load_dataset("houses")
    target_col = "LnMedianHouseValue"
    cols = [k for k in data_reg["columns"].keys() if k != target_col]
    n_rows = data_reg["n_rows"]

    split_idx = int(0.8 * n_rows)
    X_train = [[data_reg["columns"][c][i] for c in cols] for i in range(split_idx)]
    y_train = [float(data_reg["columns"][target_col][i]) for i in range(split_idx)]
    X_test = [[data_reg["columns"][c][i] for c in cols] for i in range(split_idx, n_rows)]
    y_test = [float(data_reg["columns"][target_col][i]) for i in range(split_idx, n_rows)]

    t0 = time.perf_counter()
    reg_tree = ModelTree(leaf="linear", max_depth=5, min_rows=20, seed=42)
    reg_tree.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    preds = reg_tree.predict(X_test)
    rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, y_test)) / len(y_test))
    leaves = reg_tree.n_leaves()
    splits = len(reg_tree.splits())

    print(f"  houses: fit={fit_s*1000:.1f}ms | leaves={leaves} | splits={splits} | test_rmse={rmse:.4f}")
    results.append({
        "tag": tag,
        "dataset": "houses",
        "task": "regression",
        "rows": n_rows,
        "cols": len(cols),
        "fit_ms": round(fit_s * 1000, 2),
        "leaves": leaves,
        "splits": splits,
        "metric_name": "rmse",
        "metric_val": round(rmse, 4),
    })

    # -------------------------------------------------------------
    # 2. Classification: bank-marketing (45,211 rows x 13 predictors)
    # -------------------------------------------------------------
    print("\n[2/3] Running classification benchmark: bank-marketing (45,211 rows)...")
    data_clf = load_dataset("bank-marketing")
    target_col = "SubscribeTermDeposit"
    cols_clf = [k for k in data_clf["columns"].keys() if k != target_col]
    n_rows_clf = data_clf["n_rows"]
    dtypes_clf = [data_clf["dtypes"][c] for c in cols_clf]

    split_idx_clf = int(0.8 * n_rows_clf)
    X_train_c = [[data_clf["columns"][c][i] for c in cols_clf] for i in range(split_idx_clf)]
    y_train_c = [1.0 if data_clf["columns"][target_col][i] == "yes" else 0.0 for i in range(split_idx_clf)]
    X_test_c = [[data_clf["columns"][c][i] for c in cols_clf] for i in range(split_idx_clf, n_rows_clf)]
    y_test_c = [1.0 if data_clf["columns"][target_col][i] == "yes" else 0.0 for i in range(split_idx_clf, n_rows_clf)]

    t0 = time.perf_counter()
    clf_tree = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=5, min_rows=20, seed=42)
    clf_tree.fit(X_train_c, y_train_c, dtypes=dtypes_clf)
    fit_s_clf = time.perf_counter() - t0

    preds_c = clf_tree.predict(X_test_c)
    acc = sum(1.0 for p, y in zip(preds_c, y_test_c) if (p >= 0.5) == (y >= 0.5)) / len(y_test_c)
    leaves_c = clf_tree.n_leaves()
    splits_c = len(clf_tree.splits())

    print(f"  bank-marketing: fit={fit_s_clf*1000:.1f}ms | leaves={leaves_c} | splits={splits_c} | test_acc={acc*100:.2f}%")
    results.append({
        "tag": tag,
        "dataset": "bank-marketing",
        "task": "classification",
        "rows": n_rows_clf,
        "cols": len(cols_clf),
        "fit_ms": round(fit_s_clf * 1000, 2),
        "leaves": leaves_c,
        "splits": splits_c,
        "metric_name": "accuracy",
        "metric_val": round(acc, 4),
    })

    # -------------------------------------------------------------
    # 3. Classification: credit-g (1,000 rows x 20 predictors)
    # -------------------------------------------------------------
    print("\n[3/3] Running classification benchmark: credit-g (1,000 rows)...")
    data_cg = load_dataset("credit-g")
    target_col_cg = "good_or_bad_customer"
    cols_cg = [k for k in data_cg["columns"].keys() if k != target_col_cg]
    n_rows_cg = data_cg["n_rows"]
    dtypes_cg = [data_cg["dtypes"][c] for c in cols_cg]

    split_idx_cg = int(0.8 * n_rows_cg)
    X_train_cg = [[data_cg["columns"][c][i] for c in cols_cg] for i in range(split_idx_cg)]
    y_train_cg = [1.0 if data_cg["columns"][target_col_cg][i] == "good" else 0.0 for i in range(split_idx_cg)]
    X_test_cg = [[data_cg["columns"][c][i] for c in cols_cg] for i in range(split_idx_cg, n_rows_cg)]
    y_test_cg = [1.0 if data_cg["columns"][target_col_cg][i] == "good" else 0.0 for i in range(split_idx_cg, n_rows_cg)]

    t0 = time.perf_counter()
    cg_tree = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=4, min_rows=20, seed=42)
    cg_tree.fit(X_train_cg, y_train_cg, dtypes=dtypes_cg)
    fit_s_cg = time.perf_counter() - t0

    preds_cg = cg_tree.predict(X_test_cg)
    acc_cg = sum(1.0 for p, y in zip(preds_cg, y_test_cg) if (p >= 0.5) == (y >= 0.5)) / len(y_test_cg)
    leaves_cg = cg_tree.n_leaves()
    splits_cg = len(cg_tree.splits())

    print(f"  credit-g: fit={fit_s_cg*1000:.1f}ms | leaves={leaves_cg} | splits={splits_cg} | test_acc={acc_cg*100:.2f}%")
    results.append({
        "tag": tag,
        "dataset": "credit-g",
        "task": "classification",
        "rows": n_rows_cg,
        "cols": len(cols_cg),
        "fit_ms": round(fit_s_cg * 1000, 2),
        "leaves": leaves_cg,
        "splits": splits_cg,
        "metric_name": "accuracy",
        "metric_val": round(acc_cg, 4),
    })

    # Append current run results
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag", "dataset", "task", "rows", "cols", "fit_ms", "leaves", "splits", "metric_name", "metric_val"
        ])
        writer.writerows(results)

    print(f"\nResults appended to {csv_path}")
    return results


if __name__ == "__main__":
    import sys
    tag_arg = sys.argv[1] if len(sys.argv) > 1 else "Phase1-Decoupled"
    run_benchmark(tag_arg)
