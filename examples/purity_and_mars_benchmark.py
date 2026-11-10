"""Benchmark purity gate metrics (SD vs CV vs tail_quantile) and MARS splines on tabular datasets.

Zero scikit-learn dependency. Runs honest 80/20 train/test splits.
"""

import time
import numpy as np
import pyarrow.parquet as pq
from mltreelib import ModelTree, MarsRegressor

def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def compute_metrics(y_true, y_pred, tail_p=0.10):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Tail performance (extrema)
    q_low = np.quantile(y_true, tail_p)
    q_high = np.quantile(y_true, 1.0 - tail_p)
    tail_mask = (y_true <= q_low) | (y_true >= q_high)
    tail_rmse = float(np.sqrt(np.mean((y_true[tail_mask] - y_pred[tail_mask]) ** 2))) if np.any(tail_mask) else 0.0

    return rmse, r2, tail_rmse

def run_benchmark_on_dataset(name, path, target_col):
    print("=" * 80)
    print(f"DATASET: {name} ({path})")
    print("=" * 80)

    table = pq.read_table(path)
    df = table.to_pandas()
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}, Features: {X.shape[1]}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}], Mean: {y.mean():.2f}, Std: {y.std():.2f}, CV: {y.std() / abs(y.mean()):.4f}\n")

    # -------------------------------------------------------------
    # 1. ModelTree Purity Gate Comparisons
    # -------------------------------------------------------------
    print("--- 1. ModelTree Purity Gate Evaluation ---")
    purity_configs = [
        ("Baseline (purity=0.0)", dict(purity_gate=0.0)),
        ("Purity CV 0.15", dict(purity_gate=0.15, purity_gate_metric="cv")),
        ("Purity CV 0.25", dict(purity_gate=0.25, purity_gate_metric="cv")),
        ("Purity CV 0.35", dict(purity_gate=0.35, purity_gate_metric="cv")),
        ("Purity CV 0.35 + Tail Guard (0.05)", dict(purity_gate=0.35, purity_gate_metric="cv", tail_quantile=0.05)),
        ("Purity SD 8.0", dict(purity_gate=8.0, purity_gate_metric="sd")),
        ("Purity SD 8.0 + Tail Guard (0.05)", dict(purity_gate=8.0, purity_gate_metric="sd", tail_quantile=0.05)),
    ]

    print(f"{'Configuration':<38} | {'Leaves':<7} | {'Fit (ms)':<9} | {'Test RMSE':<10} | {'Test R²':<8} | {'Tail RMSE':<10}")
    print("-" * 93)

    for label, kwargs in purity_configs:
        t0 = time.perf_counter()
        tree = ModelTree(
            leaf="linear",
            max_depth=5,
            min_rows=10,
            **kwargs
        )
        tree.fit(X_train, y_train)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        preds = tree.predict(X_test)
        rmse, r2, tail_rmse = compute_metrics(y_test, preds)
        n_leaves = tree.n_leaves()

        print(f"{label:<38} | {n_leaves:<7} | {fit_ms:<9.2f} | {rmse:<10.3f} | {r2:<8.4f} | {tail_rmse:<10.3f}")

    # -------------------------------------------------------------
    # 2. MARS / Earth Spline Evaluation
    # -------------------------------------------------------------
    print("\n--- 2. MARS (Earth) Splines Evaluation ---")
    mars_configs = [
        ("MARS Additive (degree=1, terms=15)", dict(degree=1, max_terms=15, penalty=2.0)),
        ("MARS Interactions (degree=2, terms=20)", dict(degree=2, max_terms=20, penalty=2.0)),
    ]

    print(f"{'Configuration':<42} | {'Terms':<6} | {'Fit (ms)':<9} | {'Test RMSE':<10} | {'Test R²':<8} | {'GCV':<10}")
    print("-" * 94)

    for label, kwargs in mars_configs:
        t0 = time.perf_counter()
        mars = MarsRegressor(**kwargs)
        mars.fit(X_train, y_train)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        preds = mars.predict(X_test)
        rmse, r2, _ = compute_metrics(y_test, preds)
        n_terms = mars.n_terms
        gcv = mars.gcv

        print(f"{label:<42} | {n_terms:<6} | {fit_ms:<9.2f} | {rmse:<10.3f} | {r2:<8.4f} | {gcv:<10.3f}")

        # Show equation snippet
        eq = mars.equation(feature_cols)
        lines = eq.strip().split("\n")
        first_few = "\n   ".join(lines[:4])
        print(f"   Model equation:\n   {first_few}")
        if len(lines) > 4:
            print(f"   ... ({len(lines) - 4} more terms)")
        print()

    # -------------------------------------------------------------
    # 3. Tree-Structured MARS (Piecewise MARS Leaves)
    # -------------------------------------------------------------
    print("--- 3. Tree-Structured MARS (ModelTree with leaf='mars') ---")
    tree_mars_configs = [
        ("Tree-MARS depth=2 (degree=1, terms=8)", dict(leaf="mars", max_depth=2, min_rows=30, mars_degree=1, mars_terms=8)),
        ("Tree-MARS depth=3 (degree=1, terms=8)", dict(leaf="mars", max_depth=3, min_rows=25, mars_degree=1, mars_terms=8)),
        ("Tree-MARS depth=2 + Purity CV 0.25", dict(leaf="mars", max_depth=2, min_rows=30, mars_degree=1, mars_terms=8, purity_gate=0.25, purity_gate_metric="cv")),
        ("Tree-MARS depth=2 (degree=2, terms=10)", dict(leaf="mars", max_depth=2, min_rows=30, mars_degree=2, mars_terms=10)),
    ]

    print(f"{'Configuration':<42} | {'Leaves':<7} | {'Fit (ms)':<9} | {'Test RMSE':<10} | {'Test R²':<8} | {'Tail RMSE':<10}")
    print("-" * 97)

    for label, kwargs in tree_mars_configs:
        t0 = time.perf_counter()
        tree = ModelTree(**kwargs)
        tree.fit(X_train, y_train)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        preds = tree.predict(X_test)
        rmse, r2, tail_rmse = compute_metrics(y_test, preds)
        n_leaves = tree.n_leaves()

        print(f"{label:<42} | {n_leaves:<7} | {fit_ms:<9.2f} | {rmse:<10.3f} | {r2:<8.4f} | {tail_rmse:<10.3f}")
    print()

def main():
    datasets = [
        ("Concrete Compressive Strength", "datasets/concrete_compressive_strength.parquet", "ConcreteCompressiveStrength"),
        ("Airfoil Self-Noise", "datasets/airfoil_self_noise.parquet", "scaled-sound-pressure"),
    ]

    for name, path, target_col in datasets:
        run_benchmark_on_dataset(name, path, target_col)

if __name__ == "__main__":
    main()
