"""Benchmark real bootstrap tree training scaling across dataset sizes and tree counts."""

import time
import numpy as np
from mltreelib import ModelTree


def run_benchmark():
    print("=" * 70)
    print("mltreelib Bootstrap Scaling Benchmark")
    print("=" * 70)

    np.random.seed(42)

    # Scenarios: (N_rows, N_cols)
    datasets = [
        ("Medium (N=10,000, P=20)", 10000, 20),
        ("Large  (N=50,000, P=20)", 50000, 20),
        ("XL     (N=100,000, P=20)", 100000, 20),
    ]

    tree_counts = [1, 10, 20, 50]

    for d_name, n_rows, n_cols in datasets:
        print(f"\n--- Dataset: {d_name} ---")
        X = np.random.randn(n_rows, n_cols)
        # Linear + nonlinear response
        y = (
            X[:, 0] * 2.0
            + np.sin(X[:, 1]) * 3.0
            + (X[:, 2] > 0).astype(float) * 2.5
            + np.random.randn(n_rows) * 0.5
        )

        dtypes = ["numeric"] * n_cols

        # Test Configurations:
        # Config 1: leaf="mean", max_depth=6 (Fast standard RF tree)
        # Config 2: leaf="mean", max_depth=10 (Deeper RF tree)
        # Config 3: leaf="linear", max_depth=6 (Full ModelTree with ridge leaves)
        configs = [
            ("CART leaf='mean' depth=6", {"leaf": "mean", "max_depth": 6, "min_rows": 10}),
            ("CART leaf='mean' depth=10", {"leaf": "mean", "max_depth": 10, "min_rows": 10}),
            ("ModelTree leaf='linear' depth=6", {"leaf": "linear", "max_depth": 6, "min_rows": 20}),
        ]

        for cfg_name, tree_kwargs in configs:
            print(f"\n  Configuration: {cfg_name}")
            for n_trees in tree_counts:
                t0 = time.perf_counter()
                
                # Fit n_trees with bootstrap replicates
                for b in range(n_trees):
                    boot_idx = np.random.choice(n_rows, n_rows, replace=True)
                    X_b = X[boot_idx]
                    y_b = y[boot_idx]
                    tree = ModelTree(**tree_kwargs)
                    tree.fit(X_b, y_b, dtypes=dtypes)

                elapsed = time.perf_counter() - t0
                per_tree = elapsed / n_trees
                print(
                    f"    Trees: {n_trees:2d} | "
                    f"Total Time: {elapsed:6.2f}s | "
                    f"Per Tree: {per_tree * 1000:6.1f}ms | "
                    f"Est. 200 Trees: {(per_tree * 200) / 60:5.2f} min"
                )


if __name__ == "__main__":
    run_benchmark()
