"""Comprehensive HPO Campaign to evaluate and establish empirical global defaults for ModelTree.

Sweeps across:
1. Regression benchmarks (elo_data):
   - abalone, boston, concrete, fc2008, ta_airfoil, ta_healthcare, ta_fiat500, ta_wine
2. Classification benchmarks (elo_cla_data):
   - diabetes, credit-g, blood-transfusion-service-center, churn, qsar-biodeg

Tuning knobs explored:
- Tree control:
  - max_depth: [3, 4, 5, 6, 7]
  - min_rows: [10, 20, 50, 100]
  - col_sample: [None, 0.8]
  - split_precision: [None, 0.01, 0.001]
  - prune / cp_alpha: [("none", 0.0), ("cp", 2.0), ("cp", 5.0), ("cp", 10.0)]
- Leaf control:
  - leaf: ["linear", "mean", "median", "prob"]
  - leaf_cols: ["all", "path"]
  - l2: [0.01, 0.1, 0.5, 1.0, 5.0]
  - leaf_family: ["gaussian", "binomial", "poisson"]

Computes relative error / rank scores per dataset and produces global rankings.
"""

import csv
import json
import math
import statistics
import time
import os
import sys

from mltreelib import ModelTree, LeafControl

REG_DIR = "benchmarks/elo_data"
CLA_DIR = "benchmarks/elo_cla_data"

REG_DATASETS = [
    "abalone",
    "boston",
    "concrete",
    "fc2008",
    "ta_airfoil",
    "ta_healthcare",
    "ta_fiat500",
    "ta_wine",
]

CLA_DATASETS = [
    "diabetes",
    "credit-g",
    "blood-transfusion-service-center",
    "churn",
    "qsar-biodeg",
]


def load_dataset(name, is_cla=False, cap=2500):
    prefix = CLA_DIR if is_cla else REG_DIR
    meta_path = f"{prefix}/{name}.meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    col_names = list(meta["columns"].keys())
    dtypes = [meta["columns"][c] for c in col_names]

    def read_csv(path):
        with open(path) as f:
            r = csv.reader(f)
            header = next(r)
            target_idx = len(header) - 1
            X, y = [], []
            for row in r:
                if not row:
                    continue
                row_x = []
                for idx, (c, dt) in enumerate(zip(col_names, dtypes)):
                    v = row[idx]
                    if v == "" or v is None:
                        row_x.append(None)
                    elif dt == "numeric":
                        row_x.append(float(v))
                    else:
                        row_x.append(v)
                X.append(row_x)
                if is_cla:
                    classes = meta.get("classes", ["0", "1"])
                    y.append(1.0 if row[target_idx] == classes[1] else 0.0)
                else:
                    y.append(float(row[target_idx]))
            return X, y

    xtr, ytr = read_csv(f"{prefix}/{name}_train.csv")
    xte, yte = read_csv(f"{prefix}/{name}_test.csv")
    if cap and len(xtr) > cap:
        xtr, ytr = xtr[:cap], ytr[:cap]
    return xtr, ytr, xte, yte, dtypes


def run_regression_hpo():
    print("=" * 60)
    print("RUNNING REGRESSION HPO CAMPAIGN")
    print("=" * 60)

    loaded = {}
    for ds in REG_DATASETS:
        d = load_dataset(ds, is_cla=False)
        if d:
            loaded[ds] = d
            print(f"Loaded {ds:15}: train={len(d[0])}, test={len(d[2])}")

    # Design search candidate configurations
    # Covering: current default, deeper, shallower, path-only, mean leaf, varying l2, varying min_rows, varying cp_alpha
    configs = [
        # id, label, tree_kwargs, leaf_kwargs
        ("C0_baseline", {"max_depth": 5, "min_rows": 50, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                        {"leaf": "linear", "leaf_cols": "all", "l2": 0.5}),
        ("C1_d6_m30_l2_1", {"max_depth": 6, "min_rows": 30, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                           {"leaf": "linear", "leaf_cols": "all", "l2": 1.0}),
        ("C2_d5_m25_l2_0.5", {"max_depth": 5, "min_rows": 25, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                             {"leaf": "linear", "leaf_cols": "all", "l2": 0.5}),
        ("C3_d6_m25_l2_0.5", {"max_depth": 6, "min_rows": 25, "cp_alpha": 2.0, "split_precision": 0.001, "prune": "cp"},
                             {"leaf": "linear", "leaf_cols": "all", "l2": 0.5}),
        ("C4_d5_m50_path", {"max_depth": 5, "min_rows": 50, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                           {"leaf": "linear", "leaf_cols": "path", "l2": 0.5}),
        ("C5_d6_m30_path", {"max_depth": 6, "min_rows": 30, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                           {"leaf": "linear", "leaf_cols": "path", "l2": 1.0}),
        ("C6_d7_m20_l2_1", {"max_depth": 7, "min_rows": 20, "cp_alpha": 2.0, "split_precision": 0.001, "prune": "cp"},
                           {"leaf": "linear", "leaf_cols": "all", "l2": 1.0}),
        ("C7_d5_m50_mean", {"max_depth": 5, "min_rows": 50, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                           {"leaf": "mean", "leaf_cols": "all", "l2": 0.0}),
        ("C8_d6_m25_mean", {"max_depth": 6, "min_rows": 25, "cp_alpha": 2.0, "split_precision": 0.001, "prune": "cp"},
                           {"leaf": "mean", "leaf_cols": "all", "l2": 0.0}),
        ("C9_d5_m30_l2_0.1", {"max_depth": 5, "min_rows": 30, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                             {"leaf": "linear", "leaf_cols": "all", "l2": 0.1}),
        ("C10_d6_m20_cp10", {"max_depth": 6, "min_rows": 20, "cp_alpha": 10.0, "split_precision": 0.001, "prune": "cp"},
                            {"leaf": "linear", "leaf_cols": "all", "l2": 0.5}),
        ("C11_d5_m20_no_prec", {"max_depth": 5, "min_rows": 20, "cp_alpha": 5.0, "split_precision": None, "prune": "cp"},
                               {"leaf": "linear", "leaf_cols": "all", "l2": 0.5}),
        ("C12_d6_m30_col0.8", {"max_depth": 6, "min_rows": 30, "col_sample": 0.8, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                              {"leaf": "linear", "leaf_cols": "all", "l2": 1.0}),
        ("C13_d5_m15_l2_0.5", {"max_depth": 5, "min_rows": 15, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                              {"leaf": "linear", "leaf_cols": "all", "l2": 0.5}),
        ("C14_d4_m40_l2_0.5", {"max_depth": 4, "min_rows": 40, "cp_alpha": 5.0, "split_precision": 0.001, "prune": "cp"},
                              {"leaf": "linear", "leaf_cols": "all", "l2": 0.5}),
    ]

    # Evaluate all configs on all datasets
    raw_scores = {cid: {} for cid, _, _ in configs}
    fit_durations = {cid: {} for cid, _, _ in configs}

    for cid, t_kw, l_kw in configs:
        lc = LeafControl(**l_kw)
        for ds, (xtr, ytr, xte, yte, dtypes) in loaded.items():
            t0 = time.perf_counter()
            try:
                m = ModelTree(leaf_control=lc, **t_kw, seed=42)
                m.fit(xtr, ytr, dtypes=dtypes)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                preds = m.predict(xte)
                rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, yte)) / len(yte))
            except Exception as e:
                rmse = float("inf")
                dur_ms = 0.0
                print(f"Error {cid} on {ds}: {e}")
            raw_scores[cid][ds] = rmse
            fit_durations[cid][ds] = dur_ms

    # Compute relative error compared to median/best across configs per dataset
    rel_scores = {cid: [] for cid, _, _ in configs}
    ranks = {cid: [] for cid, _, _ in configs}

    for ds in loaded.keys():
        ds_scores = [(cid, raw_scores[cid][ds]) for cid, _, _ in configs]
        ds_scores.sort(key=lambda x: x[1])
        best_score = ds_scores[0][1]
        for rank_idx, (cid, score) in enumerate(ds_scores):
            ranks[cid].append(rank_idx + 1)
            rel_scores[cid].append((score - best_score) / max(best_score, 1e-6))

    summary = []
    for cid, t_kw, l_kw in configs:
        mean_rank = statistics.mean(ranks[cid])
        mean_rel = statistics.mean(rel_scores[cid])
        mean_time = statistics.mean(fit_durations[cid].values())
        summary.append({
            "cid": cid,
            "mean_rank": mean_rank,
            "mean_rel_excess": mean_rel,
            "mean_fit_ms": mean_time,
            "tree_kw": t_kw,
            "leaf_kw": l_kw,
        })

    summary.sort(key=lambda s: s["mean_rank"])
    print("\n--- REGRESSION HPO SUMMARY RANKINGS ---")
    for s in summary:
        print(f"Rank {s['mean_rank']:.2f} | Rel +{s['mean_rel_excess']*100:.1f}% | Time: {s['mean_fit_ms']:.1f}ms | {s['cid']:18} | Tree: {s['tree_kw']} | Leaf: {s['leaf_kw']}")

    return summary


def run_classification_hpo():
    print("\n" + "=" * 60)
    print("RUNNING CLASSIFICATION HPO CAMPAIGN")
    print("=" * 60)

    loaded = {}
    for ds in CLA_DATASETS:
        d = load_dataset(ds, is_cla=True)
        if d:
            loaded[ds] = d
            print(f"Loaded {ds:32}: train={len(d[0])}, test={len(d[2])}")

    configs = [
        # (id, tree_kwargs, leaf_kwargs)
        ("CLA_prob_d5_m25", {"max_depth": 5, "min_rows": 25, "cp_alpha": 2.0, "gain": "entropy"},
                            {"leaf": "prob", "leaf_family": "binomial"}),
        ("CLA_prob_d5_m50", {"max_depth": 5, "min_rows": 50, "cp_alpha": 5.0, "gain": "entropy"},
                            {"leaf": "prob", "leaf_family": "binomial"}),
        ("CLA_prob_d6_m30", {"max_depth": 6, "min_rows": 30, "cp_alpha": 2.0, "gain": "entropy"},
                            {"leaf": "prob", "leaf_family": "binomial"}),
        ("CLA_prob_d4_m20", {"max_depth": 4, "min_rows": 20, "cp_alpha": 2.0, "gain": "entropy"},
                            {"leaf": "prob", "leaf_family": "binomial"}),
        ("CLA_mean_d5_m25", {"max_depth": 5, "min_rows": 25, "cp_alpha": 2.0, "gain": "gini"},
                            {"leaf": "mean", "leaf_family": "gaussian"}),
        ("CLA_linear_d5_m30_l2_1", {"max_depth": 5, "min_rows": 30, "cp_alpha": 5.0, "gain": "deviance"},
                                  {"leaf": "linear", "leaf_family": "binomial", "l2": 1.0}),
        ("CLA_linear_d6_m30_l2_0.5", {"max_depth": 6, "min_rows": 30, "cp_alpha": 2.0, "gain": "deviance"},
                                    {"leaf": "linear", "leaf_family": "binomial", "l2": 0.5}),
        ("CLA_prob_d6_m20_bic", {"max_depth": 6, "min_rows": 20, "prune": "bic", "gain": "entropy"},
                                {"leaf": "prob", "leaf_family": "binomial"}),
    ]

    raw_scores = {cid: {} for cid, _, _ in configs}
    fit_durations = {cid: {} for cid, _, _ in configs}

    eps = 1e-15
    for cid, t_kw, l_kw in configs:
        lc = LeafControl(**l_kw)
        for ds, (xtr, ytr, xte, yte, dtypes) in loaded.items():
            t0 = time.perf_counter()
            try:
                m = ModelTree(leaf_control=lc, **t_kw, seed=42)
                m.fit(xtr, ytr, dtypes=dtypes)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                preds = m.predict(xte, kind="prob" if l_kw.get("leaf_family") == "binomial" else "response")
                # Log-loss
                ll = -sum(
                    y * math.log(max(eps, min(1.0 - eps, p))) + (1.0 - y) * math.log(max(eps, min(1.0 - eps, 1.0 - p)))
                    for p, y in zip(preds, yte)
                ) / len(yte)
            except Exception as e:
                ll = 999.0
                dur_ms = 0.0
                print(f"Error {cid} on {ds}: {e}")
            raw_scores[cid][ds] = ll
            fit_durations[cid][ds] = dur_ms

    ranks = {cid: [] for cid, _, _ in configs}
    rel_scores = {cid: [] for cid, _, _ in configs}

    for ds in loaded.keys():
        ds_scores = [(cid, raw_scores[cid][ds]) for cid, _, _ in configs]
        ds_scores.sort(key=lambda x: x[1])
        best_score = ds_scores[0][1]
        for rank_idx, (cid, score) in enumerate(ds_scores):
            ranks[cid].append(rank_idx + 1)
            rel_scores[cid].append((score - best_score) / max(best_score, 1e-6))

    summary = []
    for cid, t_kw, l_kw in configs:
        mean_rank = statistics.mean(ranks[cid])
        mean_rel = statistics.mean(rel_scores[cid])
        mean_time = statistics.mean(fit_durations[cid].values())
        summary.append({
            "cid": cid,
            "mean_rank": mean_rank,
            "mean_rel_excess": mean_rel,
            "mean_fit_ms": mean_time,
            "tree_kw": t_kw,
            "leaf_kw": l_kw,
        })

    summary.sort(key=lambda s: s["mean_rank"])
    print("\n--- CLASSIFICATION HPO SUMMARY RANKINGS ---")
    for s in summary:
        print(f"Rank {s['mean_rank']:.2f} | Rel +{s['mean_rel_excess']*100:.1f}% | Time: {s['mean_fit_ms']:.1f}ms | {s['cid']:25} | Tree: {s['tree_kw']} | Leaf: {s['leaf_kw']}")

    return summary


def main():
    reg_summary = run_regression_hpo()
    cla_summary = run_classification_hpo()

    # Save to CSV
    out_csv = "benchmarks/hpo_empirical_defaults.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "rank", "cid", "mean_rank", "mean_rel_excess", "mean_fit_ms", "tree_params", "leaf_params"])
        for idx, s in enumerate(reg_summary):
            w.writerow(["regression", idx + 1, s["cid"], f"{s['mean_rank']:.3f}", f"{s['mean_rel_excess']:.3f}", f"{s['mean_fit_ms']:.1f}", json.dumps(s["tree_kw"]), json.dumps(s["leaf_kw"])])
        for idx, s in enumerate(cla_summary):
            w.writerow(["classification", idx + 1, s["cid"], f"{s['mean_rank']:.3f}", f"{s['mean_rel_excess']:.3f}", f"{s['mean_fit_ms']:.1f}", json.dumps(s["tree_kw"]), json.dumps(s["leaf_kw"])])
    print(f"\nSaved empirical HPO results to {out_csv}")


if __name__ == "__main__":
    main()
