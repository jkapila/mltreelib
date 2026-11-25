"""Compare ModelTree parameter variations against baseline ModelTree(all) across all 56 TabArena datasets.

Evaluates:
- 18 Regression datasets (test RMSE + fit time)
- 38 Classification datasets (test Accuracy, LogLoss + fit time)

Variations tested:
1. Baseline: ModelTree(all) (default parameters: depth=6, min_rows=15, shrinkage=0.0)
2. Var 1: ModelTree(accurate) (preset="accurate": depth=8, min_rows=10, shrinkage=10.0, gain="twoing" for MC)
3. Var 2: ModelTree(subspaced_ridge) (depth=8, min_rows=12, shrinkage=8.0, l2=3.0, col_sample=0.85, gain="twoing" for MC)
4. Var 3: ModelTree(pruned_bic) (depth=8, min_rows=10, shrinkage=5.0, prune="bic", gain="twoing" for MC)
5. Var 4 (Reg only): ModelTree(spline_mars) (leaf="mars", max_depth=4, mars_degree=1, shrinkage=5.0)
"""

import itertools
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from elo_common import discover_datasets, read_split
from mltreelib import ModelTree

REG_DIR = Path("benchmarks/elo_data")
CLA_DIR = Path("benchmarks/elo_cla_data")


def compute_elo(models, datasets, scores, lower_is_better=True, k=32.0, start=1000.0):
    ratings = {m: start for m in models}
    for ds in datasets:
        here = {m: scores[(ds, m)] for m in models if (ds, m) in scores}
        for a, b in itertools.combinations(sorted(here), 2):
            va, vb = here[a], here[b]
            if lower_is_better:
                sa = 1.0 if va < vb else (0.0 if va > vb else 0.5)
            else:
                sa = 1.0 if va > vb else (0.0 if va < vb else 0.5)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += k * (sa - ea)
            ratings[b] += k * ((1.0 - sa) - (1.0 - ea))
    return ratings


# =============================================================================
# 1. REGRESSION SUITE
# =============================================================================
def run_regression_suite(limit=None):
    print("\n" + "=" * 90)
    print("EVALUATING MODEL VARIATIONS ON REGRESSION (18 TabArena datasets)")
    print("=" * 90)

    datasets = discover_datasets(REG_DIR)
    if limit:
        datasets = datasets[:limit]

    models = [
        "ModelTree(all)",
        "ModelTree(accurate)",
        "ModelTree(subspaced_ridge)",
        "ModelTree(pruned_bic)",
        "ModelTree(spline_mars)",
    ]

    rmse_scores = {}
    fit_times = {m: [] for m in models}

    for idx, ds in enumerate(datasets, 1):
        xtr, ytr, xte, yte, meta = read_split(REG_DIR, ds, is_classification=False)
        roles = list(meta["columns"].values()) if meta and "columns" in meta else None
        p = len(xtr[0])
        dtypes = roles if roles and len(roles) == p else [
            "categorical" if any(isinstance(r[j], str) for r in xtr) else "numeric"
            for j in range(p)
        ]
        ytr_flt = [float(y) for y in ytr]
        yte_flt = [float(y) for y in yte]

        eff_leaf = "mean" if p >= 100 else "linear"
        eff_cols = "path" if (50 <= p < 100) else "all"

        for m_name in models:
            t0 = time.perf_counter()
            try:
                if m_name == "ModelTree(all)":
                    m = ModelTree(leaf=eff_leaf, leaf_cols=eff_cols, seed=42)
                elif m_name == "ModelTree(accurate)":
                    m = ModelTree(preset="accurate", leaf=eff_leaf, leaf_cols=eff_cols, seed=42)
                elif m_name == "ModelTree(subspaced_ridge)":
                    m = ModelTree(max_depth=8, min_rows=12, shrinkage=8.0, l2=3.0, col_sample=0.85, leaf=eff_leaf, leaf_cols=eff_cols, seed=42)
                elif m_name == "ModelTree(pruned_bic)":
                    m = ModelTree(max_depth=8, min_rows=10, shrinkage=5.0, prune="bic", leaf=eff_leaf, leaf_cols=eff_cols, seed=42)
                elif m_name == "ModelTree(spline_mars)":
                    m = ModelTree(leaf="mars", max_depth=4, mars_degree=1, shrinkage=5.0, seed=42)
                
                m.fit(xtr, ytr_flt, dtypes=dtypes)
                preds = m.predict(xte)
                rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, yte_flt)) / len(yte_flt))
                fit_s = time.perf_counter() - t0
                rmse_scores[(ds, m_name)] = rmse
                fit_times[m_name].append(fit_s)
            except Exception as e:
                print(f"  [ERROR] {m_name} failed on {ds}: {e}")
                rmse_scores[(ds, m_name)] = float("inf")
                fit_times[m_name].append(0.0)

        best_m = min(models, key=lambda m: rmse_scores.get((ds, m), float("inf")))
        base_rmse = rmse_scores.get((ds, "ModelTree(all)"), 0.0)
        best_rmse = rmse_scores.get((ds, best_m), 0.0)
        diff = ((best_rmse - base_rmse) / base_rmse) * 100 if base_rmse > 0 else 0.0
        print(f"[{idx:2d}/{len(datasets):2d}] {ds:<22} | Base: {base_rmse:10.3f} | Best: {best_m:<25} ({best_rmse:10.3f}, {diff:+6.2f}%)", flush=True)

    # Elo & Summary
    ratings = compute_elo(models, datasets, rmse_scores, lower_is_better=True)
    base_rating = ratings["ModelTree(all)"]

    print("\n" + "-" * 90)
    print(f"{'MODEL':<28} | {'REG ELO':<8} | {'vs (all)':<9} | {'AVG FIT TIME':<12} | {'WIN/TIE/LOSS vs (all)'}")
    print("-" * 90)

    for m in sorted(models, key=lambda x: -ratings[x]):
        r = ratings[m]
        delta = r - base_rating
        avg_t = np.mean(fit_times[m]) if fit_times[m] else 0.0
        if m == "ModelTree(all)":
            record_str = "baseline"
        else:
            wins = sum(1 for ds in datasets if rmse_scores.get((ds, m), 9e9) < rmse_scores.get((ds, "ModelTree(all)"), 9e9))
            ties = sum(1 for ds in datasets if rmse_scores.get((ds, m), 9e9) == rmse_scores.get((ds, "ModelTree(all)"), 9e9))
            losses = sum(1 for ds in datasets if rmse_scores.get((ds, m), 9e9) > rmse_scores.get((ds, "ModelTree(all)"), 9e9))
            record_str = f"{wins}W - {ties}D - {losses}L"

        print(f"{m:<28} | {r:7.1f}  | {delta:+7.1f}   | {avg_t*1000:7.1f} ms    | {record_str}")

    return ratings, rmse_scores


# =============================================================================
# 2. CLASSIFICATION SUITE
# =============================================================================
def run_classification_suite(limit=None):
    print("\n" + "=" * 90)
    print("EVALUATING MODEL VARIATIONS ON CLASSIFICATION (38 TabArena datasets)")
    print("=" * 90)

    datasets = discover_datasets(CLA_DIR)
    if limit:
        datasets = datasets[:limit]

    models = [
        "ModelTree(all)",
        "ModelTree(accurate)",
        "ModelTree(subspaced_ridge)",
        "ModelTree(pruned_bic)",
    ]

    acc_scores = {}
    fit_times = {m: [] for m in models}

    for idx, ds in enumerate(datasets, 1):
        xtr, ytr, xte, yte, meta = read_split(CLA_DIR, ds, is_classification=True)
        roles = list(meta["columns"].values()) if meta and "columns" in meta else None
        classes = meta["classes"]
        n_classes = len(classes)
        is_binary = (n_classes == 2)
        p = len(xtr[0])
        dtypes = roles if roles and len(roles) == p else [
            "categorical" if any(isinstance(r[j], str) for r in xtr) else "numeric"
            for j in range(p)
        ]

        # Target formatting
        if is_binary:
            pos_label = str(classes[1])
            y_bin_tr = [1.0 if str(y) == pos_label else 0.0 for y in ytr]
            y_bin_te = [1.0 if str(y) == pos_label else 0.0 for y in yte]
        else:
            class_map = {str(c): float(i) for i, c in enumerate(classes)}
            y_multi_tr = [class_map[str(y)] for y in ytr]
            y_multi_te = [class_map[str(y)] for y in yte]

        eff_leaf = "mean" if p >= 100 else "linear"
        eff_cols = "path" if (50 <= p < 100) else "all"

        for m_name in models:
            t0 = time.perf_counter()
            try:
                if is_binary:
                    if m_name == "ModelTree(all)":
                        m = ModelTree(leaf=eff_leaf, leaf_family="binomial", gain="deviance", leaf_cols=eff_cols, seed=42)
                    elif m_name == "ModelTree(accurate)":
                        m = ModelTree(preset="accurate", leaf=eff_leaf, leaf_cols=eff_cols, leaf_family="binomial", seed=42)
                    elif m_name == "ModelTree(subspaced_ridge)":
                        m = ModelTree(max_depth=8, min_rows=12, shrinkage=8.0, l2=3.0, col_sample=0.85, leaf=eff_leaf, leaf_cols=eff_cols, leaf_family="binomial", seed=42)
                    elif m_name == "ModelTree(pruned_bic)":
                        m = ModelTree(max_depth=8, min_rows=10, shrinkage=5.0, prune="bic", leaf=eff_leaf, leaf_cols=eff_cols, leaf_family="binomial", seed=42)

                    m.fit(xtr, y_bin_tr, dtypes=dtypes)
                    probs = m.predict_proba(xte)
                    preds = [1.0 if prob >= 0.5 else 0.0 for prob in probs]
                    acc = sum(1.0 for pr, true in zip(preds, y_bin_te) if pr == true) / len(yte)

                else:
                    # Multiclass
                    if m_name == "ModelTree(all)":
                        m = ModelTree(leaf=eff_leaf, leaf_family="softmax", n_classes=n_classes, gain="entropy", leaf_cols=eff_cols, seed=42)
                    elif m_name == "ModelTree(accurate)":
                        m = ModelTree(preset="accurate", leaf=eff_leaf, leaf_cols=eff_cols, n_classes=n_classes, gain="twoing", seed=42)
                    elif m_name == "ModelTree(subspaced_ridge)":
                        m = ModelTree(max_depth=8, min_rows=12, shrinkage=8.0, l2=3.0, col_sample=0.85, leaf=eff_leaf, leaf_cols=eff_cols, n_classes=n_classes, gain="twoing", seed=42)
                    elif m_name == "ModelTree(pruned_bic)":
                        m = ModelTree(max_depth=8, min_rows=10, shrinkage=5.0, prune="bic", leaf=eff_leaf, leaf_cols=eff_cols, n_classes=n_classes, gain="twoing", seed=42)

                    m.fit(xtr, y_multi_tr, dtypes=dtypes)
                    preds = m.predict(xte)
                    acc = sum(1.0 for pr, true in zip(preds, y_multi_te) if round(pr) == round(true)) / len(yte)

                fit_s = time.perf_counter() - t0
                acc_scores[(ds, m_name)] = acc
                fit_times[m_name].append(fit_s)
            except Exception as e:
                print(f"  [ERROR] {m_name} failed on {ds}: {e}")
                acc_scores[(ds, m_name)] = 0.0
                fit_times[m_name].append(0.0)

        best_m = max(models, key=lambda m: acc_scores.get((ds, m), 0.0))
        base_acc = acc_scores.get((ds, "ModelTree(all)"), 0.0) * 100
        best_acc = acc_scores.get((ds, best_m), 0.0) * 100
        diff = best_acc - base_acc
        print(f"[{idx:2d}/{len(datasets):2d}] {ds:<32} | Base: {base_acc:6.2f}% | Best: {best_m:<25} ({best_acc:6.2f}%, {diff:+6.2f}%)", flush=True)

    # Elo & Summary
    ratings = compute_elo(models, datasets, acc_scores, lower_is_better=False)
    base_rating = ratings["ModelTree(all)"]

    print("\n" + "-" * 90)
    print(f"{'MODEL':<28} | {'CLA ELO':<8} | {'vs (all)':<9} | {'AVG FIT TIME':<12} | {'WIN/TIE/LOSS vs (all)'}")
    print("-" * 90)

    for m in sorted(models, key=lambda x: -ratings[x]):
        r = ratings[m]
        delta = r - base_rating
        avg_t = np.mean(fit_times[m]) if fit_times[m] else 0.0
        if m == "ModelTree(all)":
            record_str = "baseline"
        else:
            wins = sum(1 for ds in datasets if acc_scores.get((ds, m), 0.0) > acc_scores.get((ds, "ModelTree(all)"), 0.0))
            ties = sum(1 for ds in datasets if acc_scores.get((ds, m), 0.0) == acc_scores.get((ds, "ModelTree(all)"), 0.0))
            losses = sum(1 for ds in datasets if acc_scores.get((ds, m), 0.0) < acc_scores.get((ds, "ModelTree(all)"), 0.0))
            record_str = f"{wins}W - {ties}D - {losses}L"

        print(f"{m:<28} | {r:7.1f}  | {delta:+7.1f}   | {avg_t*1000:7.1f} ms    | {record_str}")

    return ratings, acc_scores


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_regression_suite(limit=limit)
    run_classification_suite(limit=limit)
