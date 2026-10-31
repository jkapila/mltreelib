"""Optimization Tracking Benchmark across TabArena datasets.

Tracks fit runtime, leaf counts, and prediction accuracy across incremental
tree growth optimizations on both regression and classification tasks.

Datasets:
  - Regression: 'houses' (20,640 rows, 8 features, target LnMedianHouseValue)
  - Classification: 'bank-marketing' (45,211 rows, 13 features, target SubscribeTermDeposit)
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

from mltreelib import ModelTree
from mltreelib.datasets import load_dataset


def run_benchmark(tag: str = "current") -> list[dict]:
    results = []

    # -------------------------------------------------------------
    # 1. Regression: houses (20,640 rows x 8 predictors)
    # -------------------------------------------------------------
    print("Loading regression dataset: houses...")
    data_reg = load_dataset("houses")
    target_col = "LnMedianHouseValue"
    cols = [k for k in data_reg["columns"].keys() if k != target_col]
    n_rows = data_reg["n_rows"]

    # 80/20 train/test split
    split_idx = int(0.8 * n_rows)
    X_train = [[data_reg["columns"][c][i] for c in cols] for i in range(split_idx)]
    y_train = [float(data_reg["columns"][target_col][i]) for i in range(split_idx)]
    X_test = [[data_reg["columns"][c][i] for c in cols] for i in range(split_idx, n_rows)]
    y_test = [float(data_reg["columns"][target_col][i]) for i in range(split_idx, n_rows)]

    print(f"Fitting ModelTree(leaf='linear') on houses ({len(X_train)} train rows, {len(cols)} cols)...")
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
    print("Loading classification dataset: bank-marketing...")
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

    print(f"Fitting ModelTree(leaf='mean', family='binomial') on bank-marketing ({len(X_train_c)} train rows, {len(cols_clf)} cols)...")
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

    # Record to CSV
    csv_path = Path("benchmarks/optimization_tracker.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag", "dataset", "task", "rows", "cols", "fit_ms", "leaves", "splits", "metric_name", "metric_val"
        ])
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults appended to {csv_path}")
    return results


if __name__ == "__main__":
    import sys
    tag_arg = sys.argv[1] if len(sys.argv) > 1 else "Phase1-Decoupled"
    run_benchmark(tag_arg)
