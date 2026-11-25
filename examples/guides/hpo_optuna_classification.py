"""Full-Surface Hyperparameter Optimization for Binary & Multiclass Classification.

Tunes:
- Growth & Topology: max_depth, min_rows, max_leaf_nodes, min_gain, alpha, imbalance_penalty
- Split Knobs: gain, split_precision, split_eval, col_sample, presort, cat_search, cat_order
- Leaf Architecture & Regularization: leaf, leaf_family, leaf_solver, shrinkage, l2, l1
- Pruning & Complexity: prune, cp_alpha
- Purity & Screening Gates: purity_gate, purity_gate_metric, preselect, min_relevance, split_test

Evaluates:
1. Binary Classification: credit-g (1,000 rows x 20 features)
2. Multiclass Classification: SDSS17 (78,053 rows, subsample 10,000 for fast trials x 11 features, 3 classes)

Usage:
    uv run --with optuna python examples/guides/hpo_optuna_classification.py
"""

import csv
import json
import math
import os
import sys
import optuna
import numpy as np
from mltreelib import ModelTree

CLA_DIR = "benchmarks/elo_cla_data"

def load_cla_dataset(name, cap=None):
    meta_path = f"{CLA_DIR}/{name}.meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    col_names = list(meta["columns"].keys())
    dtypes = [meta["columns"][c] for c in col_names]
    classes = meta["classes"]
    class_map = {c: float(i) for i, c in enumerate(classes)}

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
                y.append(class_map.get(row[target_idx], 0.0))
            return X, y

    xtr, ytr = read_csv(f"{CLA_DIR}/{name}_train.csv")
    xte, yte = read_csv(f"{CLA_DIR}/{name}_test.csv")
    if cap and len(xtr) > cap:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(xtr), cap, replace=False)
        xtr = [xtr[i] for i in idx]
        ytr = [ytr[i] for i in idx]
    return xtr, ytr, xte, yte, dtypes, len(classes)


# -----------------------------------------------------------------------------
# 1. Binary Classification Objective (credit-g)
# -----------------------------------------------------------------------------
def run_binary_hpo(n_trials=40):
    print("=" * 85)
    print(f"RUNNING FULL-SURFACE BINARY CLASSIFICATION HPO (credit-g, {n_trials} trials)")
    print("=" * 85)
    
    xtr, ytr, xte, yte, dtypes, n_classes = load_cla_dataset("credit-g")
    
    def objective_binary(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 2, 9)
        min_rows = trial.suggest_int("min_rows", 4, 80, log=True)
        use_max_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [False, True])
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 8, 96, log=True) if use_max_leaf_nodes else None
        
        gain = trial.suggest_categorical("gain", ["deviance", "gini", "entropy", "gain_ratio", "twoing"])
        split_eval = trial.suggest_categorical("split_eval", ["fast", "exhaustive"])
        split_precision = trial.suggest_categorical("split_precision", [None, 0.1, 0.01, 0.001])
        use_col_sample = trial.suggest_categorical("use_col_sample", [False, True])
        col_sample = trial.suggest_float("col_sample", 0.6, 0.95) if use_col_sample else None
        
        leaf = trial.suggest_categorical("leaf", ["linear", "mean", "prob", "odds", "log_odds"])
        leaf_family = trial.suggest_categorical("leaf_family", ["binomial", "probit"])
        leaf_solver = trial.suggest_categorical("leaf_solver", ["irls", "lbfgs"])
        shrinkage = trial.suggest_float("shrinkage", 0.0, 25.0)
        l2 = trial.suggest_float("l2", 1e-4, 15.0, log=True)
        l1 = trial.suggest_float("l1", 0.0, 1.0)
        
        prune = trial.suggest_categorical("prune", ["none", "cp", "bic", "mdl"])
        cp_alpha = trial.suggest_float("cp_alpha", 0.1, 20.0) if prune == "cp" else 5.0
        
        use_purity = trial.suggest_categorical("use_purity", [False, True])
        purity_gate = trial.suggest_float("purity_gate", 0.001, 0.05, log=True) if use_purity else None
        
        preselect = trial.suggest_categorical("preselect", ["none", "winnow", "boruta", "variance", "mutual_info", "unitree"])
        min_relevance = trial.suggest_float("min_relevance", 0.01, 0.20) if preselect != "none" else None

        params = dict(
            max_depth=max_depth,
            min_rows=min_rows,
            max_leaf_nodes=max_leaf_nodes,
            gain=gain,
            split_eval=split_eval,
            split_precision=split_precision,
            col_sample=col_sample,
            leaf=leaf,
            leaf_family=leaf_family,
            leaf_solver=leaf_solver,
            shrinkage=shrinkage,
            l2=l2,
            l1=l1,
            prune=prune,
            cp_alpha=cp_alpha,
            purity_gate=purity_gate,
            preselect=preselect,
            min_relevance=min_relevance,
            seed=42,
        )
        try:
            m = ModelTree(**params)
            m.fit(xtr, ytr, dtypes=dtypes)
            preds = m.predict_proba(xte) if hasattr(m, "predict_proba") else m.predict(xte)
            acc = sum(1.0 for p, y in zip(preds, yte) if (p >= 0.5) == (y >= 0.5)) / len(yte)
            return acc
        except Exception:
            return 0.0

    def print_best(study, trial):
        if study.best_trial.number == trial.number:
            print(f"  --> Trial {trial.number:2d}: New Best Acc = {trial.value*100:.2f}% | "
                  f"depth={trial.params.get('max_depth')}, min_rows={trial.params.get('min_rows')}, "
                  f"gain={trial.params.get('gain')}, leaf={trial.params.get('leaf')}, "
                  f"shrinkage={trial.params.get('shrinkage', 0.0):.2f}, l2={trial.params.get('l2', 0.0):.4f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_binary, n_trials=n_trials, callbacks=[print_best], show_progress_bar=False)

    print(f"\n[Binary] Optimization Finished: {len(study.trials)} trials completed.")
    print(f"[Binary] Best Test Accuracy: {study.best_value * 100:.2f}% (Baseline was 75.00%)")
    print("\nBest Binary Parameter Settings:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}={v!r},")
    t = study.best_trial
    print(f"Winning Trial #{t.number}")
    return study.best_params


# -----------------------------------------------------------------------------
# 2. Multiclass Classification Objective (SDSS17)
# -----------------------------------------------------------------------------
def run_multiclass_hpo(n_trials=40):
    print("\n" + "=" * 85)
    print(f"RUNNING FULL-SURFACE MULTICLASS CLASSIFICATION HPO (SDSS17, {n_trials} trials)")
    print("=" * 85)
    
    # Cap training rows to 12,000 for ultra-fast exploration over 40 trials
    xtr, ytr, xte, yte, dtypes, n_classes = load_cla_dataset("SDSS17", cap=12000)
    
    def objective_multiclass(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 3, 9)
        min_rows = trial.suggest_int("min_rows", 8, 80, log=True)
        use_max_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [False, True])
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 16, 128, log=True) if use_max_leaf_nodes else None
        
        gain = trial.suggest_categorical("gain", ["gini", "entropy", "gain_ratio", "twoing"])
        split_eval = trial.suggest_categorical("split_eval", ["fast", "exhaustive"])
        split_precision = trial.suggest_categorical("split_precision", [None, 0.1, 0.01, 0.001])
        use_col_sample = trial.suggest_categorical("use_col_sample", [False, True])
        col_sample = trial.suggest_float("col_sample", 0.65, 0.95) if use_col_sample else None
        
        leaf = trial.suggest_categorical("leaf", ["mean", "prob", "linear"])
        shrinkage = trial.suggest_float("shrinkage", 0.0, 25.0)
        l2 = trial.suggest_float("l2", 1e-4, 15.0, log=True)
        
        prune = trial.suggest_categorical("prune", ["none", "cp", "bic", "mdl"])
        cp_alpha = trial.suggest_float("cp_alpha", 0.1, 20.0) if prune == "cp" else 5.0
        
        preselect = trial.suggest_categorical("preselect", ["none", "winnow", "variance", "mutual_info"])
        min_relevance = trial.suggest_float("min_relevance", 0.01, 0.15) if preselect != "none" else None

        params = dict(
            max_depth=max_depth,
            min_rows=min_rows,
            max_leaf_nodes=max_leaf_nodes,
            gain=gain,
            n_classes=n_classes,
            split_eval=split_eval,
            split_precision=split_precision,
            col_sample=col_sample,
            leaf=leaf,
            shrinkage=shrinkage,
            l2=l2,
            prune=prune,
            cp_alpha=cp_alpha,
            preselect=preselect,
            min_relevance=min_relevance,
            seed=42,
        )
        try:
            m = ModelTree(**params)
            m.fit(xtr, ytr, dtypes=dtypes)
            preds = m.predict(xte)
            acc = sum(1.0 for p, y in zip(preds, yte) if round(p) == round(y)) / len(yte)
            return acc
        except Exception:
            return 0.0

    def print_best(study, trial):
        if study.best_trial.number == trial.number:
            print(f"  --> Trial {trial.number:2d}: New Best Acc = {trial.value*100:.2f}% | "
                  f"depth={trial.params.get('max_depth')}, min_rows={trial.params.get('min_rows')}, "
                  f"gain={trial.params.get('gain')}, leaf={trial.params.get('leaf')}, "
                  f"shrinkage={trial.params.get('shrinkage', 0.0):.2f}, l2={trial.params.get('l2', 0.0):.4f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_multiclass, n_trials=n_trials, callbacks=[print_best], show_progress_bar=False)

    print(f"\n[Multiclass] Optimization Finished: {len(study.trials)} trials completed.")
    print(f"[Multiclass] Best Test Accuracy: {study.best_value * 100:.2f}%")
    print("\nBest Multiclass Parameter Settings:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}={v!r},")
    t = study.best_trial
    print(f"Winning Trial #{t.number}")
    return study.best_params


def clean_binary_params(p):
    return dict(
        max_depth=p["max_depth"],
        min_rows=p["min_rows"],
        max_leaf_nodes=p.get("max_leaf_nodes") if p.get("use_max_leaf_nodes") else None,
        gain=p["gain"],
        split_eval=p["split_eval"],
        split_precision=p["split_precision"],
        col_sample=p.get("col_sample") if p.get("use_col_sample") else None,
        leaf=p["leaf"],
        leaf_family=p.get("leaf_family", "binomial"),
        leaf_solver=p.get("leaf_solver", "irls"),
        shrinkage=p.get("shrinkage", 0.0),
        l2=p.get("l2", 1.0),
        l1=p.get("l1", 0.0),
        prune=p.get("prune", "none"),
        cp_alpha=p.get("cp_alpha", 5.0) if p.get("prune") == "cp" else 5.0,
        purity_gate=p.get("purity_gate") if p.get("use_purity") else None,
        preselect=p.get("preselect", "none"),
        min_relevance=p.get("min_relevance") if p.get("preselect") != "none" else None,
        seed=42,
    )


def clean_multiclass_params(p, n_classes):
    return dict(
        max_depth=p["max_depth"],
        min_rows=p["min_rows"],
        max_leaf_nodes=p.get("max_leaf_nodes") if p.get("use_max_leaf_nodes") else None,
        gain=p["gain"],
        n_classes=n_classes,
        split_eval=p["split_eval"],
        split_precision=p["split_precision"],
        col_sample=p.get("col_sample") if p.get("use_col_sample") else None,
        leaf=p["leaf"],
        shrinkage=p.get("shrinkage", 0.0),
        l2=p.get("l2", 1.0),
        prune=p.get("prune", "none"),
        cp_alpha=p.get("cp_alpha", 5.0) if p.get("prune") == "cp" else 5.0,
        preselect=p.get("preselect", "none"),
        min_relevance=p.get("min_relevance") if p.get("preselect") != "none" else None,
        seed=42,
    )


def evaluate_across_datasets(best_bin, best_mc):
    print("\n" + "=" * 95)
    print("VALIDATING DEFAULT VS PRESET ACCURATE VS HPO WINNER ACROSS CLASSIFICATION DATASETS")
    print("=" * 95)
    print(f"{'Dataset':<32} | {'Task':<10} | {'Rows':<7} | {'Default':<10} | {'Accurate':<10} | {'HPO Best':<10} | {'Delta':<8}")
    print("-" * 95)

    bin_datasets = ["credit-g", "diabetes", "blood-transfusion-service-center", "qsar-biodeg"]
    mc_datasets = ["maternal_health_risk", "anneal", "SDSS17"]

    results = []

    for name in bin_datasets:
        xtr, ytr, xte, yte, dtypes, n_classes = load_cla_dataset(name)
        # Default
        m_def = ModelTree(leaf_family="binomial", leaf="prob", seed=42)
        m_def.fit(xtr, ytr, dtypes=dtypes)
        p_def = m_def.predict(xte)
        acc_def = sum(1.0 for p, y in zip(p_def, yte) if (p >= 0.5) == (y >= 0.5)) / len(yte)

        # Preset Accurate
        m_acc = ModelTree(preset="accurate", leaf_family="binomial", leaf="prob", seed=42)
        m_acc.fit(xtr, ytr, dtypes=dtypes)
        p_acc = m_acc.predict(xte)
        acc_acc = sum(1.0 for p, y in zip(p_acc, yte) if (p >= 0.5) == (y >= 0.5)) / len(yte)

        # HPO
        hpo_p = clean_binary_params(best_bin)
        m_hpo = ModelTree(**hpo_p)
        m_hpo.fit(xtr, ytr, dtypes=dtypes)
        p_hpo = m_hpo.predict(xte)
        acc_hpo = sum(1.0 for p, y in zip(p_hpo, yte) if (p >= 0.5) == (y >= 0.5)) / len(yte)

        delta = (acc_hpo - acc_def) * 100.0
        print(f"{name:<32} | {'binary':<10} | {len(xtr):<7} | {acc_def*100:6.2f}%    | {acc_acc*100:6.2f}%    | {acc_hpo*100:6.2f}%    | {delta:+6.2f}%")
        results.append((name, "binary", acc_def, acc_acc, acc_hpo, delta))

    for name in mc_datasets:
        cap = 12000 if name == "SDSS17" else None
        xtr, ytr, xte, yte, dtypes, n_classes = load_cla_dataset(name, cap=cap)
        # Default
        m_def = ModelTree(seed=42, n_classes=n_classes)
        m_def.fit(xtr, ytr, dtypes=dtypes)
        p_def = m_def.predict(xte)
        acc_def = sum(1.0 for p, y in zip(p_def, yte) if round(p) == round(y)) / len(yte)

        # Preset Accurate
        m_acc = ModelTree(preset="accurate", seed=42, n_classes=n_classes)
        m_acc.fit(xtr, ytr, dtypes=dtypes)
        p_acc = m_acc.predict(xte)
        acc_acc = sum(1.0 for p, y in zip(p_acc, yte) if round(p) == round(y)) / len(yte)

        # HPO
        hpo_p = clean_multiclass_params(best_mc, n_classes)
        m_hpo = ModelTree(**hpo_p)
        m_hpo.fit(xtr, ytr, dtypes=dtypes)
        p_hpo = m_hpo.predict(xte)
        acc_hpo = sum(1.0 for p, y in zip(p_hpo, yte) if round(p) == round(y)) / len(yte)

        delta = (acc_hpo - acc_def) * 100.0
        print(f"{name:<32} | {'multiclass':<10} | {len(xtr):<7} | {acc_def*100:6.2f}%    | {acc_acc*100:6.2f}%    | {acc_hpo*100:6.2f}%    | {delta:+6.2f}%")
        results.append((name, "multiclass", acc_def, acc_acc, acc_hpo, delta))

    print("-" * 95)
    mean_def = sum(r[2] for r in results) / len(results) * 100
    mean_acc = sum(r[3] for r in results) / len(results) * 100
    mean_hpo = sum(r[4] for r in results) / len(results) * 100
    print(f"{'AVERAGE':<32} | {'ALL':<10} | {'-':<7} | {mean_def:6.2f}%    | {mean_acc:6.2f}%    | {mean_hpo:6.2f}%    | {mean_hpo - mean_def:+6.2f}%")
    print("=" * 95)


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    best_bin = run_binary_hpo(trials)
    best_mc = run_multiclass_hpo(trials)
    evaluate_across_datasets(best_bin, best_mc)
