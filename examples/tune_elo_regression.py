"""Tune ModelTree hyperparameters across the 18 Elo regression datasets.

Evaluates:
- Baseline ModelTree (Old default: d=5, min_rows=50, cp_alpha=5, shrinkage=0)
- New Default (d=6, min_rows=15, shrinkage=0)
- Preset 'accurate' (d=8, min_rows=10, shrinkage=10, prune="none")
- Candidate A (Balanced Shrinkage: d=6, min_rows=15, shrinkage=5.0, l2=0.5, cp_alpha=2.0)
- Candidate B (Deep + Regularized: d=8, min_rows=12, shrinkage=15.0, l2=1.0, prune="none")
- Candidate C (Subspace Shrinkage: d=7, min_rows=15, shrinkage=10.0, l2=0.5, col_sample=0.8)

Calculates pairwise Elo-style win rate and average relative RMSE against the baseline.
"""

import csv
import json
import math
import os
import time
import numpy as np
from mltreelib import ModelTree

DATA_DIR = "benchmarks/elo_data"

DATASETS = [
    "abalone",
    "boston",
    "concrete",
    "fc2008",
    "ozone",
    "ta_airfoil",
    "ta_concrete",
    "ta_diamonds",
    "ta_fiat500",
    "ta_food_delivery",
    "ta_healthcare",
    "ta_houses",
    "ta_miami",
    "ta_protein",
    "ta_qsar_fish",
    "ta_qsar_tid11",
    "ta_superconductivity",
    "ta_wine",
]

def load_ds(name, cap=4000):
    meta_path = f"{DATA_DIR}/{name}.meta.json"
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
                y.append(float(row[target_idx]))
            return X, y

    xtr, ytr = read_csv(f"{DATA_DIR}/{name}_train.csv")
    xte, yte = read_csv(f"{DATA_DIR}/{name}_test.csv")
    if cap and len(xtr) > cap:
        xtr, ytr = xtr[:cap], ytr[:cap]
    return xtr, ytr, xte, yte, dtypes

def run_tuning():
    print("=" * 95)
    print("MODELTREE ELO ARENA HYPERPARAMETER TUNING (18 Regression Datasets)")
    print("=" * 95)
    
    loaded = {}
    for ds in DATASETS:
        d = load_ds(ds)
        if d:
            loaded[ds] = d
    print(f"Loaded {len(loaded)} of {len(DATASETS)} datasets.\n")

    candidates = {
        "Old_Baseline": {
            "max_depth": 5, "min_rows": 50, "cp_alpha": 5.0, "shrinkage": 0.0, "l2": 0.5, "prune": "cp"
        },
        "New_Default": {
            "max_depth": 6, "min_rows": 15, "cp_alpha": 5.0, "shrinkage": 0.0, "l2": 0.5, "prune": "cp"
        },
        "Preset_Accurate": {
            "preset": "accurate"
        },
        "Candidate_A (Shrink_5)": {
            "max_depth": 6, "min_rows": 15, "shrinkage": 5.0, "l2": 0.5, "cp_alpha": 2.0, "prune": "cp"
        },
        "Candidate_B (Deep_Shrink_15)": {
            "max_depth": 8, "min_rows": 12, "shrinkage": 15.0, "l2": 1.0, "prune": "none"
        },
        "Candidate_C (Subspace_Shrink_10)": {
            "max_depth": 7, "min_rows": 15, "shrinkage": 10.0, "l2": 0.5, "col_sample": 0.8, "cp_alpha": 2.0, "prune": "cp"
        },
    }

    results = {c: {} for c in candidates}
    durations = {c: {} for c in candidates}

    for cname, params in candidates.items():
        for ds, (xtr, ytr, xte, yte, dtypes) in loaded.items():
            t0 = time.perf_counter()
            try:
                m = ModelTree(leaf="linear", leaf_cols="all", seed=42, **params)
                m.fit(xtr, ytr, dtypes=dtypes)
                preds = m.predict(xte)
                rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, yte)) / len(yte))
                results[cname][ds] = rmse
                durations[cname][ds] = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                print(f"[{cname}] failed on {ds}: {e}")
                results[cname][ds] = float("nan")

    # Print summary table
    print(f"{'Dataset':<20} | " + " | ".join(f"{c[:14]:<14}" for c in candidates))
    print("-" * 115)
    
    wins = {c: 0 for c in candidates}
    rel_rmses = {c: [] for c in candidates}

    for ds in loaded.keys():
        row_str = f"{ds:<20} | "
        ds_vals = {c: results[c].get(ds, float("nan")) for c in candidates}
        valid_vals = [v for v in ds_vals.values() if not math.isnan(v)]
        if not valid_vals:
            continue
        best_v = min(valid_vals)
        base_v = ds_vals["Old_Baseline"]
        
        for c in candidates:
            v = ds_vals[c]
            if not math.isnan(v):
                if v == best_v:
                    wins[c] += 1
                if not math.isnan(base_v) and base_v > 0:
                    rel_rmses[c].append(v / base_v)
                star = "*" if v == best_v else " "
                row_str += f"{v:10.4f}{star} | "
            else:
                row_str += f"{'FAIL':<14} | "
        print(row_str)

    print("\n" + "=" * 95)
    print("TUNING SUMMARY SCOREBOARD:")
    print("=" * 95)
    for c in candidates:
        avg_rel = np.mean(rel_rmses[c]) if rel_rmses[c] else 1.0
        geom_rel = math.exp(np.mean(np.log(rel_rmses[c]))) if rel_rmses[c] else 1.0
        avg_ms = np.mean(list(durations[c].values())) if durations[c] else 0.0
        win_count = wins[c]
        print(f"  {c:<30} | Wins: {win_count:2d}/18 | Mean Rel RMSE: {avg_rel:.4f} (Geom: {geom_rel:.4f}) | Avg Fit: {avg_ms:5.1f}ms")

if __name__ == "__main__":
    run_tuning()
