"""Comprehensive Validation: Solvers, Multiclass Splits, and Leaf Shrinkage.

Measures:
1. Multiclass Split Scaling (streaming histogram subtraction O(N*C) vs baseline)
2. Solver Execution & Convergence (IRLS vs L-BFGS vs CCD vs Evolutionary)
3. Hierarchical Leaf Shrinkage & Presets (Variance control on deep trees)
"""

import time
import math
import numpy as np
from mltreelib import ModelTree, LeafControl

def run_multiclass_benchmark():
    print("=" * 80)
    print("1. MULTICLASS SPLIT OPTIMIZATION BENCHMARK (Streaming Histogram Subtraction)")
    print("=" * 80)
    
    n_samples = 30000
    n_features = 10
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    
    for n_classes in [3, 5, 10, 20]:
        # Latent class assignment
        weights = rng.randn(n_features, n_classes)
        logits = X @ weights
        y = np.argmax(logits, axis=1).astype(float)
        
        # Split train/test
        X_train, X_test = X[:25000].tolist(), X[25000:].tolist()
        y_train, y_test = y[:25000].tolist(), y[25000:].tolist()
        
        # Fast streaming histogram split
        t0 = time.perf_counter()
        tree_fast = ModelTree(
            gain="gini",
            n_classes=n_classes,
            max_depth=5,
            min_rows=20,
            split_eval="fast",
            seed=42
        )
        tree_fast.fit(X_train, y_train)
        fast_ms = (time.perf_counter() - t0) * 1000
        
        preds_fast = tree_fast.predict(X_test)
        acc_fast = sum(1.0 for p, yt in zip(preds_fast, y_test) if round(p) == round(yt)) / len(y_test)
        
        # Exhaustive comparison
        t0 = time.perf_counter()
        tree_ex = ModelTree(
            gain="gini",
            n_classes=n_classes,
            max_depth=5,
            min_rows=20,
            split_eval="exhaustive",
            seed=42
        )
        tree_ex.fit(X_train, y_train)
        ex_ms = (time.perf_counter() - t0) * 1000
        
        preds_ex = tree_ex.predict(X_test)
        acc_ex = sum(1.0 for p, yt in zip(preds_ex, y_test) if round(p) == round(yt)) / len(y_test)
        
        speedup = ex_ms / max(fast_ms, 0.001)
        leaves_fast = tree_fast.n_leaves()
        leaves_ex = tree_ex.n_leaves()
        
        print(f"Classes: {n_classes:2d} | Fast: {fast_ms:6.1f}ms (Acc={acc_fast*100:5.2f}%, Leaves={leaves_fast}) | "
              f"Exhaustive: {ex_ms:6.1f}ms (Acc={acc_ex*100:5.2f}%, Leaves={leaves_ex}) | "
              f"Speedup: {speedup:4.2f}x | Match: {acc_fast == acc_ex}")

def run_solver_benchmark():
    print("\n" + "=" * 80)
    print("2. SOLVER PERFORMANCE & CONVERGENCE BENCHMARK")
    print("=" * 80)
    
    rng = np.random.RandomState(42)
    n = 5000
    p = 8
    X = rng.randn(n, p)
    beta = np.array([2.5, -1.8, 0.0, 0.0, 1.2, -0.5, 0.0, 3.0])
    
    # 2.1 Linear / Sparse Regression (Gaussian)
    y_reg = X @ beta + 0.5 * rng.randn(n)
    X_list = X.tolist()
    y_list = y_reg.tolist()
    
    solvers = ["irls", "ccd", "lbfgs"]
    print("\n[A] Gaussian Ridge/Lasso Solvers (Linear Leaves, N=5000, P=8):")
    for s in solvers:
        t0 = time.perf_counter()
        tree = ModelTree(leaf="linear", leaf_solver=s, max_depth=3, min_rows=30, l1=0.01, l2=0.1, seed=42)
        tree.fit(X_list, y_list)
        ms = (time.perf_counter() - t0) * 1000
        preds = tree.predict(X_list)
        rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, y_list)) / len(y_list))
        print(f"  Solver: {s.upper():<6} | Fit time: {ms:6.2f}ms | Train RMSE: {rmse:.4f} | Leaves: {tree.n_leaves()}")
    
    # 2.2 Binomial Logistic Regression
    p_true = 1.0 / (1.0 + np.exp(-(X @ beta[:p])))
    y_bin = (rng.rand(n) < p_true).astype(float).tolist()
    
    print("\n[B] Binomial Logistic Solvers (Binomial Leaves, N=5000, P=8):")
    for s in ["irls", "lbfgs"]:
        t0 = time.perf_counter()
        tree = ModelTree(leaf="linear", leaf_family="binomial", leaf_solver=s, max_depth=3, min_rows=30, seed=42)
        tree.fit(X_list, y_bin)
        ms = (time.perf_counter() - t0) * 1000
        probs = tree.predict_proba(X_list)
        log_loss = -sum(y * math.log(max(pr, 1e-15)) + (1 - y) * math.log(max(1 - pr, 1e-15)) for y, pr in zip(y_bin, probs)) / len(y_bin)
        print(f"  Solver: {s.upper():<6} | Fit time: {ms:6.2f}ms | Log-Loss: {log_loss:.4f} | Leaves: {tree.n_leaves()}")

    # 2.3 Evolutionary Solvers (Global Optimization)
    print("\n[C] Evolutionary Solvers (Stochastic / Non-Convex Parameter Search):")
    evo_solvers = [("cmaes", "CMA-ES (Covariance Matrix Adaptation)"),
                   ("shade", "SHADE (Success-History Adaptive DE)"),
                   ("nes",   "sNES (Separable Natural Evolution Strategies)"),
                   ("ga",    "RCGA (Real-Coded Genetic Algorithm)")]
    for s_name, desc in evo_solvers:
        t0 = time.perf_counter()
        tree = ModelTree(leaf="linear", leaf_solver=s_name, max_depth=2, min_rows=100, seed=42)
        tree.fit(X_list[:1000], y_list[:1000])
        ms = (time.perf_counter() - t0) * 1000
        preds = tree.predict(X_list[:1000])
        rmse = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, y_list[:1000])) / 1000)
        print(f"  {desc:<45} | Fit: {ms:6.2f}ms | RMSE: {rmse:.4f}")

def run_shrinkage_benchmark():
    print("\n" + "=" * 80)
    print("3. HIERARCHICAL LEAF SHRINKAGE & PRESETS BENCHMARK (Variance Control)")
    print("=" * 80)
    
    # Non-linear synthetic regression with high noise
    rng = np.random.RandomState(42)
    n_train = 1000
    n_test = 1000
    X_all = rng.uniform(-3, 3, size=(n_train + n_test, 4))
    # True function: non-linear interactions
    y_true = np.sin(X_all[:, 0] * 2) + 0.5 * X_all[:, 1]**2 - 0.8 * X_all[:, 2]
    # Add significant noise to test overfitting in deep leaves
    y_all = y_true + 0.7 * rng.randn(n_train + n_test)
    
    X_train, y_train = X_all[:n_train].tolist(), y_all[:n_train].tolist()
    X_test, y_test = X_all[n_train:].tolist(), y_all[n_train:].tolist()
    y_test_true = y_true[n_train:].tolist()
    
    # Compare:
    # 1. Unshrunk deep tree (shrinkage=0.0, max_depth=7)
    # 2. Shrunk deep tree (shrinkage=10.0, max_depth=7)
    # 3. Preset "accurate" (max_depth=8, min_rows=10, shrinkage=10.0, prune="none")
    # 4. Preset "fast" (max_depth=5, min_rows=20, split_eval="fast")
    # 5. Preset "interpretable" (max_depth=3, min_rows=30)
    
    configs = [
        ("Deep Tree (Unshrunk, lambda=0.0)", ModelTree(leaf="linear", max_depth=7, min_rows=10, shrinkage=0.0, seed=42)),
        ("Deep Tree (Shrunk, lambda=10.0)",   ModelTree(leaf="linear", max_depth=7, min_rows=10, shrinkage=10.0, seed=42)),
        ("Preset 'accurate' (lambda=10.0)",  ModelTree(preset="accurate", seed=42)),
        ("Preset 'fast'",                     ModelTree(preset="fast", seed=42)),
        ("Preset 'interpretable'",           ModelTree(preset="interpretable", seed=42)),
    ]
    
    for name, tree in configs:
        t0 = time.perf_counter()
        tree.fit(X_train, y_train)
        fit_ms = (time.perf_counter() - t0) * 1000
        preds_test = tree.predict(X_test)
        # Test RMSE against true ground-truth signal
        rmse_noise = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds_test, y_test)) / len(y_test))
        rmse_true = math.sqrt(sum((p - yt) ** 2 for p, yt in zip(preds_test, y_test_true)) / len(y_test_true))
        leaves = tree.n_leaves()
        print(f"  {name:<36} | Fit: {fit_ms:5.1f}ms | Test RMSE: {rmse_noise:.4f} (True RMSE: {rmse_true:.4f}) | Leaves: {leaves}")

if __name__ == "__main__":
    run_multiclass_benchmark()
    run_solver_benchmark()
    run_shrinkage_benchmark()
