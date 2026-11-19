"""Benchmark and comparison of all 7 numerical and evolutionary leaf solvers in mltreelib.

Compares:
1. Analytical / Exact: 'irls' (Cholesky/faer normal equations)
2. Quasi-Newton: 'lbfgs' (Limited-memory BFGS with Armijo backtracking)
3. Coordinate Descent: 'ccd' (Cyclic Coordinate Descent for Elastic Net L1+L2)
4. Evolutionary / Derivative-Free:
   - 'cmaes' (Covariance Matrix Adaptation Evolution Strategy)
   - 'shade' (Success-History based Adaptive Differential Evolution)
   - 'nes'   (Separable Natural Evolution Strategies)
   - 'ga'    (Real-Coded Genetic Algorithm with SBX crossover)

Evaluates on Yeh (1998) Concrete Compressive Strength benchmark (1,030 rows).
Produces: docs/assets/solver_benchmark_pareto.png
"""

import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mltreelib import ModelTree, Metrics, load_dataset


def run_solver_benchmarks():
    print("=" * 70)
    print("mltreelib Solver Benchmark: 7 Leaf Solvers on Real Tabular Data")
    print("=" * 70)

    # 1. Ingest benchmark dataset
    data = load_dataset("concrete_compressive_strength")
    target = "ConcreteCompressiveStrength"
    feature_names = [col for col in data["columns"].keys() if col != target]
    n_rows = data["n_rows"]

    X = [[data["columns"][c][i] for c in feature_names] for i in range(n_rows)]
    y = data["columns"][target]

    split_idx = int(n_rows * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    print(f"Dataset: Concrete Compressive Strength ({n_rows} rows, {len(feature_names)} features)")
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    metrics_engine = Metrics()

    # Solvers to evaluate
    solvers = [
        ("irls", "Exact Cholesky / IRLS", "#00897b"),
        ("lbfgs", "Quasi-Newton L-BFGS", "#004d40"),
        ("ccd", "Cyclic Coordinate Descent", "#26a69a"),
        ("shade", "Adaptive DE (SHADE)", "#f4511e"),
        ("cmaes", "Covariance Matrix (CMA-ES)", "#e65100"),
        ("ga", "Genetic Algorithm (GA)", "#ff7043"),
        ("nes", "Natural Gradient (NES)", "#d84315"),
    ]

    results = []

    print("\n--- Evaluating Standalone Global Solvers (max_depth=0) ---")
    print(f"{'Solver':<10} | {'Method Family':<28} | {'Latency (ms)':>12} | {'Test RMSE':>10} | {'Test R2':>10}")
    print("-" * 80)

    for solver_key, label, color in solvers:
        # Measure latency over multiple iterations (fewer for ccd)
        n_repeats = 3 if solver_key != "ccd" else 1
        durations = []
        model = None
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model = ModelTree(max_depth=0, leaf="linear", leaf_solver=solver_key, l2=0.01, seed=42)
            model.fit(X_train, y_train, col_names=feature_names)
            durations.append((time.perf_counter() - t0) * 1000.0)

        latency_ms = float(np.median(durations))
        preds = model.predict(X_test)
        m = metrics_engine.regression(y_test, preds)

        results.append({
            "key": solver_key,
            "label": label,
            "color": color,
            "latency_ms": latency_ms,
            "rmse": m["rmse"],
            "r2": m["r2"],
            "mae": m["mae"],
            "model": model,
        })

        print(f"{solver_key:<10} | {label:<28} | {latency_ms:>12.2f} | {m['rmse']:>10.3f} | {m['r2']:>10.3f}")

    # Also evaluate leaf solver in a tree (max_depth=2, 4 leaves)
    print("\n--- Tree-Decoupled Leaf Fitting (max_depth=2, 4 Leaves) ---")
    tree_results = []
    tree_solvers = [("irls", "IRLS"), ("lbfgs", "L-BFGS"), ("shade", "SHADE"), ("cmaes", "CMA-ES"), ("ga", "GA")]
    for s_key, s_name in tree_solvers:
        t0 = time.perf_counter()
        t_model = ModelTree(max_depth=2, min_rows=20, leaf="linear", leaf_solver=s_key, l2=0.01, seed=42)
        t_model.fit(X_train, y_train, col_names=feature_names)
        t_lat = (time.perf_counter() - t0) * 1000.0
        preds = t_model.predict(X_test)
        m = metrics_engine.regression(y_test, preds)
        tree_results.append((s_key, s_name, t_lat, m["rmse"], m["r2"]))
        print(f"Tree + {s_key:<8} | Latency: {t_lat:6.2f} ms | Test RMSE: {m['rmse']:6.3f} | R2: {m['r2']:6.3f}")

    # 2. Render Publication-Grade Diagnostic Figure
    os.makedirs("docs/assets", exist_ok=True)
    out_path = "docs/assets/solver_benchmark_pareto.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Latency vs RMSE Pareto Frontier
    for r in results:
        ax1.scatter(
            r["latency_ms"],
            r["rmse"],
            s=130,
            color=r["color"],
            alpha=0.9,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
        # Position annotations neatly
        offset_x = 1.15
        offset_y = 0.0
        if r["key"] == "lbfgs":
            offset_y = -0.3
        elif r["key"] == "irls":
            offset_y = 0.3
        elif r["key"] == "nes":
            offset_y = -0.3
        ax1.annotate(
            r["key"].upper(),
            (r["latency_ms"] * offset_x, r["rmse"] + offset_y),
            fontsize=9.5,
            fontweight="bold",
            color=r["color"],
        )

    ax1.set_xscale("log")
    ax1.set_title("A: Latency vs Holdout Accuracy Pareto Frontier", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Fit Latency (ms, log-scale)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Holdout Test RMSE (MPa)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, which="both")
    ax1.axhline(10.224, color="#00897b", linestyle=":", alpha=0.6, label="Exact OLS Floor (10.22 MPa)")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Panel B: Tree-Decoupled Speedup & Performance
    tree_names = [t[1] for t in tree_results]
    tree_rmses = [t[3] for t in tree_results]
    tree_lats = [t[2] for t in tree_results]

    x_pos = np.arange(len(tree_names))
    width = 0.38

    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x_pos - width/2, tree_rmses, width, label="Test RMSE (MPa)", color="#00897b", alpha=0.85, edgecolor="black")
    bars2 = ax2_twin.bar(x_pos + width/2, tree_lats, width, label="Fit Time (ms)", color="#f4511e", alpha=0.85, edgecolor="black")

    ax2.set_title("B: Solvers Inside ModelTree (Depth=2, 4 Leaves)", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tree_names, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Holdout Test RMSE (MPa, Lower is Better)", fontsize=10, fontweight="bold", color="#00897b")
    ax2_twin.set_ylabel("Fit Latency (ms)", fontsize=10, fontweight="bold", color="#f4511e")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Value labels
    for bar in bars1:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for bar in bars2:
        yval = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.1f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nSaved solver comparison figure to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_solver_benchmarks()
