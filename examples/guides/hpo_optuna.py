"""Hyperparameter search over the FULL ModelTree parameter surface with Optuna.

Tunes:
1. Growth & Topology: max_depth, min_rows, max_leaf_nodes, min_gain, alpha, imbalance_penalty
2. Split Search: gain, split_precision, split_eval, col_sample, presort, cat_search, cat_order
3. Leaf Architecture: leaf, leaf_cols, shrinkage, l2, l1, leaf_solver, mars_degree/terms, knn_k/variant/metric
4. Regularization & Pruning: prune, cp_alpha
5. Purity & Stopping Gates: purity_gate, purity_gate_metric, tail_quantile
6. Screening & Statistical Gates: preselect, min_relevance, split_test, split_alpha, bonferroni
7. Honesty & Estimation: honesty, honesty_fraction

Usage:
    uv run --with optuna python examples/guides/hpo_optuna.py [n_trials]
"""

import sys
import math
import optuna
import pandas as pd
from mltreelib import MetricsReport, ModelTree

df = pd.read_parquet("datasets/airfoil_self_noise.parquet")
y = df["scaled-sound-pressure"].astype(float).tolist()
cols = [c for c in df.columns if c != "scaled-sound-pressure"]
X, dtypes = [], []
for col in cols:
    raw = [None if pd.isna(v) else v for v in df[col]]
    if str(df[col].dtype) == "category":
        dtypes.append("categorical")
        X.append([None if v is None else float(str(v)) for v in raw])
    else:
        dtypes.append("numeric")
        X.append([None if v is None else float(v) for v in raw])
X = [list(r) for r in zip(*X)]

n_tr = int(0.8 * len(X))
xtr, ytr, xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def objective(trial: optuna.Trial) -> float:
    # 1. Growth & Topology
    max_depth = trial.suggest_int("max_depth", 2, 10)
    min_rows = trial.suggest_int("min_rows", 4, 100, log=True)
    use_max_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [False, True])
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 8, 128, log=True) if use_max_leaf_nodes else None
    min_gain = trial.suggest_float("min_gain", 1e-9, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.0, 0.25)
    imbalance_penalty = trial.suggest_float("imbalance_penalty", 1e-4, 10.0, log=True)
    
    # 2. Split Search & Criteria
    gain = trial.suggest_categorical("gain", ["sse", "friedman_mse", "sdr", "huber", "absolute_error"])
    split_precision = trial.suggest_categorical("split_precision", [None, 0.1, 0.01, 0.001])
    split_eval = trial.suggest_categorical("split_eval", ["fast", "exhaustive"])
    use_col_sample = trial.suggest_categorical("use_col_sample", [False, True])
    col_sample = trial.suggest_float("col_sample", 0.5, 0.95) if use_col_sample else None
    presort = trial.suggest_categorical("presort", [False, True])
    cat_search = trial.suggest_categorical("cat_search", ["cumulative", "single_vs_rest", "both"])
    cat_order = trial.suggest_categorical("cat_order", ["frequency_desc", "target_mean"])
    
    # 3. Leaf Architecture & Solvers
    leaf = trial.suggest_categorical("leaf", ["linear", "mean", "huber", "mars", "knn"])
    leaf_cols = trial.suggest_categorical("leaf_cols", ["all", "path"])
    shrinkage = trial.suggest_float("shrinkage", 0.0, 30.0)
    l2 = trial.suggest_float("l2", 1e-4, 20.0, log=True)
    l1 = trial.suggest_float("l1", 0.0, 1.0)
    leaf_solver = trial.suggest_categorical("leaf_solver", ["irls", "ccd", "lbfgs"])
    
    # MARS specific
    mars_degree = trial.suggest_int("mars_degree", 1, 2) if leaf == "mars" else 1
    mars_terms = trial.suggest_int("mars_terms", 4, 16) if leaf == "mars" else 10
    mars_penalty = trial.suggest_float("mars_penalty", 1.0, 4.0) if leaf == "mars" else 2.0
    
    # KNN specific
    knn_k = trial.suggest_int("knn_k", 3, 15) if leaf == "knn" else 5
    knn_variant = trial.suggest_categorical("knn_variant", ["uniform", "distance", "exponential", "dual"]) if leaf == "knn" else "uniform"
    knn_metric = trial.suggest_categorical("knn_metric", ["euclidean", "manhattan", "chebyshev"]) if leaf == "knn" else "euclidean"

    # 4. Pruning
    prune = trial.suggest_categorical("prune", ["none", "cp", "bic", "mdl"])
    cp_alpha = trial.suggest_float("cp_alpha", 0.1, 25.0) if prune == "cp" else 5.0
    
    # 5. Purity Gate & Extrema
    use_purity_gate = trial.suggest_categorical("use_purity_gate", [False, True])
    purity_gate = trial.suggest_float("purity_gate", 0.001, 0.05, log=True) if use_purity_gate else None
    purity_gate_metric = trial.suggest_categorical("purity_gate_metric", ["sd", "cv"])
    use_tail_quantile = trial.suggest_categorical("use_tail_quantile", [False, True])
    tail_quantile = trial.suggest_float("tail_quantile", 0.01, 0.10) if use_tail_quantile else None
    
    # 6. Screening / Preselection
    preselect = trial.suggest_categorical("preselect", ["none", "winnow", "boruta", "variance", "mutual_info", "unitree"])
    min_relevance = trial.suggest_float("min_relevance", 0.01, 0.20) if preselect != "none" else None
    
    # 7. Statistical Gates (Chow test)
    split_test = trial.suggest_categorical("split_test", ["none", "chow"])
    split_alpha = trial.suggest_float("split_alpha", 0.01, 0.10) if split_test != "none" else 0.05
    bonferroni = trial.suggest_categorical("bonferroni", [False, True]) if split_test != "none" else False
    
    # 8. Honesty (grf style)
    honesty = trial.suggest_categorical("honesty", [False, True])
    honesty_fraction = trial.suggest_float("honesty_fraction", 0.5, 0.8) if honesty else 0.5

    params = dict(
        max_depth=max_depth,
        min_rows=min_rows,
        max_leaf_nodes=max_leaf_nodes,
        min_gain=min_gain,
        alpha=alpha,
        imbalance_penalty=imbalance_penalty,
        gain=gain,
        split_precision=split_precision,
        split_eval=split_eval,
        col_sample=col_sample,
        presort=presort,
        cat_search=cat_search,
        cat_order=cat_order,
        leaf=leaf,
        leaf_cols=leaf_cols,
        shrinkage=shrinkage,
        l2=l2,
        l1=l1,
        leaf_solver=leaf_solver,
        mars_degree=mars_degree,
        mars_terms=mars_terms,
        mars_penalty=mars_penalty,
        knn_k=knn_k,
        knn_variant=knn_variant,
        knn_metric=knn_metric,
        prune=prune,
        cp_alpha=cp_alpha,
        purity_gate=purity_gate,
        purity_gate_metric=purity_gate_metric,
        tail_quantile=tail_quantile,
        preselect=preselect,
        min_relevance=min_relevance,
        split_test=split_test,
        split_alpha=split_alpha,
        bonferroni=bonferroni,
        honesty=honesty,
        honesty_fraction=honesty_fraction,
        seed=42,
    )
    
    try:
        m = ModelTree(**params)
        m.fit(xtr, ytr, dtypes=dtypes)
        preds = m.predict(xte)
        rmse = MetricsReport("regression", y_true=yte, y_pred=preds)["rmse"]
        if math.isnan(rmse) or math.isinf(rmse):
            return 9999.0
        return rmse
    except Exception:
        return 9999.0


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    print(f"Starting Optuna search across FULL parameter surface ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nOptimization Finished: {len(study.trials)} trials completed.")
    print(f"Best test RMSE on airfoil_self_noise: {study.best_value:.4f}")
    print("\nBest Parameter Settings:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}={v!r},")
    t = study.best_trial
    print(f"\nWinning Trial #{t.number} (seeded TPE, airfoil 80/20)")


if __name__ == "__main__":
    main()
