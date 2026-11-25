"""Benchmark cardinality-adaptive binned splits vs raw scan across tree runs (1, 10, 20, 50, 100)."""

import time
import numpy as np
from mltreelib import ModelTree


def run_benchmark():
    print("=" * 80)
    print("mltreelib Benchmark: Cardinality-Adaptive Binned Splits vs Raw Scan")
    print("=" * 80)

    np.random.seed(42)

    # N = 50,000 rows, P = 20 features
    n_rows = 50000
    n_cols = 20
    print(f"\nDataset Size: N = {n_rows:,} rows, P = {n_cols} features")

    # Construct mixed features:
    # - 10 continuous standard normal
    # - 5 low-cardinality discrete counts/ordinals (3-8 unique values)
    # - 5 binary flags (2 unique values)
    X = np.empty((n_rows, n_cols), dtype=np.float64)
    for j in range(10):
        X[:, j] = np.random.randn(n_rows)
    for j in range(10, 15):
        n_levels = j - 7  # 3, 4, 5, 6, 7
        X[:, j] = np.random.randint(0, n_levels, size=n_rows)
    for j in range(15, 20):
        X[:, j] = np.random.randint(0, 2, size=n_rows)

    # Complex response: linear + non-linear + interaction
    y = (
        X[:, 0] * 2.5
        + np.sin(X[:, 1]) * 3.0
        + (X[:, 2] > 0.0).astype(float) * 2.0
        + X[:, 10] * 1.5
        + (X[:, 11] == 2).astype(float) * 3.5
        + X[:, 15] * 2.0
        + np.random.randn(n_rows) * 0.5
    )

    # 80/20 train/test split
    n_train = 40000
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    tree_counts = [1, 10, 20, 50, 100]

    print("\nBenchmark Configuration: leaf='mean', gain='sse', max_depth=8, min_rows=15")
    print("-" * 80)
    print(f"{'Trees':>5} | {'Fast Scan (s)':>14} | {'Binned Hist (s)':>15} | {'Speedup':>10} | {'RMSE Fast':>10} | {'RMSE Hist':>10}")
    print("-" * 80)

    for n_trees in tree_counts:
        # 1. Fast raw scan
        t0 = time.perf_counter()
        preds_fast = np.zeros(len(y_test))
        for b in range(n_trees):
            # Deterministic bootstrap replicate
            np.random.seed(1000 + b)
            boot_idx = np.random.choice(n_train, n_train, replace=True)
            tree = ModelTree(
                split_eval="fast",
                gain="sse",
                leaf="mean",
                max_depth=8,
                min_rows=15,
                seed=42 + b,
            )
            tree.fit(X_train[boot_idx], y_train[boot_idx])
            preds_fast += np.asarray(tree.predict(X_test))
        t_fast = time.perf_counter() - t0
        preds_fast /= n_trees
        rmse_fast = float(np.sqrt(np.mean((preds_fast - y_test) ** 2)))

        # 2. Cardinality-adaptive histogram scan
        t0 = time.perf_counter()
        preds_hist = np.zeros(len(y_test))
        for b in range(n_trees):
            np.random.seed(1000 + b)
            boot_idx = np.random.choice(n_train, n_train, replace=True)
            tree = ModelTree(
                split_eval="histogram",
                max_bins=256,
                gain="sse",
                leaf="mean",
                max_depth=8,
                min_rows=15,
                seed=42 + b,
            )
            tree.fit(X_train[boot_idx], y_train[boot_idx])
            preds_hist += np.asarray(tree.predict(X_test))
        t_hist = time.perf_counter() - t0
        preds_hist /= n_trees
        rmse_hist = float(np.sqrt(np.mean((preds_hist - y_test) ** 2)))

        speedup = t_fast / t_hist if t_hist > 0 else 0.0

        print(
            f"{n_trees:5d} | {t_fast:14.2f} | {t_hist:15.2f} | {speedup:9.2f}x | {rmse_fast:10.4f} | {rmse_hist:10.4f}"
        )

    print("-" * 80)
    print("Benchmark complete.\n")


if __name__ == "__main__":
    run_benchmark()
