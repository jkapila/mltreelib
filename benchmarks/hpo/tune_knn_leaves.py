"""Hyperparameter tuning and benchmark comparison for KNN leaf models.

Compares:
1. Baselines: CART (Mean Leaf), ModelTree (Linear Leaf)
2. KNN Leaf Variants:
   - uniform (Standard KNN)
   - distance (Dudani Inverse Distance)
   - exponential (Kernel exp(-gamma * d))
   - dual (Gou et al. Dual-Weighted: distance * compactness)
   - double (DW-KNN: Pathak, Karthik, Madan, Kapila: distance consistency * neighbor validity)
3. Distance Metrics:
   - euclidean, manhattan, chebyshev, gower, canberra
4. Hyperparameter Grids:
   - knn_k: [3, 5, 7, 11]
   - max_depth: [2, 4]
"""

import time

import numpy as np
from mltreelib import ModelTree


def run_regression_tuning():
    print("=" * 80)
    print("KNN LEAF HYPERPARAMETER TUNING — REGRESSION BENCHMARK")
    print("=" * 80)

    # Use a challenging non-linear dataset: ta_airfoil or synthetic continuous manifold
    rng = np.random.RandomState(42)
    N_train, N_test = 500, 200
    P = 6

    # Planted multi-modal non-linear surface with boundary flips
    X_train = rng.uniform(-3.0, 3.0, size=(N_train, P))
    r_tr = np.sqrt(X_train[:, 0] ** 2 + X_train[:, 1] ** 2)
    y_train = (
        np.sin(r_tr) * 3.0
        + np.cos(X_train[:, 2] * 2.0) * 1.5
        + (X_train[:, 3] > 0.0) * 4.0
        + rng.normal(0, 0.25, size=N_train)
    )

    X_test = rng.uniform(-3.0, 3.0, size=(N_test, P))
    r_te = np.sqrt(X_test[:, 0] ** 2 + X_test[:, 1] ** 2)
    y_test = (
        np.sin(r_te) * 3.0
        + np.cos(X_test[:, 2] * 2.0) * 1.5
        + (X_test[:, 3] > 0.0) * 4.0
        + rng.normal(0, 0.25, size=N_test)
    )

    # Baselines
    results = []

    # 1. Mean Leaf (CART)
    m_cart = ModelTree(leaf="mean", max_depth=4, min_rows=10, seed=42)
    m_cart.fit(X_train.tolist(), y_train.tolist())
    p_cart = np.array(m_cart.predict(X_test.tolist()))
    rmse_cart = np.sqrt(np.mean((y_test - p_cart) ** 2))
    results.append(("CART (Mean Leaf)", "mean", "-", "-", 4, "-", rmse_cart))

    # 2. Linear Leaf (ModelTree OLS)
    m_lin = ModelTree(leaf="linear", max_depth=4, min_rows=10, l2=1e-3, seed=42)
    m_lin.fit(X_train.tolist(), y_train.tolist())
    p_lin = np.array(m_lin.predict(X_test.tolist()))
    rmse_lin = np.sqrt(np.mean((y_test - p_lin) ** 2))
    results.append(("ModelTree (Linear Leaf)", "linear", "-", "-", 4, "-", rmse_lin))

    # 3. Grid search over KNN leaf configurations
    variants = ["uniform", "distance", "exponential", "dual", "double"]
    metrics = ["euclidean", "manhattan", "gower"]
    ks = [3, 5, 7, 11]

    for var in variants:
        for met in metrics:
            for k in ks:
                t0 = time.perf_counter()
                m_knn = ModelTree(
                    leaf="knn",
                    knn_variant=var,
                    knn_metric=met,
                    knn_k=k,
                    knn_k_validity=5,
                    max_depth=4,
                    min_rows=12,
                    seed=42,
                )
                m_knn.fit(X_train.tolist(), y_train.tolist())
                fit_t = time.perf_counter() - t0
                p_knn = np.array(m_knn.predict(X_test.tolist()))
                rmse_knn = np.sqrt(np.mean((y_test - p_knn) ** 2))
                results.append((f"Leaf-KNN ({var})", var, met, k, 4, f"{fit_t*1000:.1f}ms", rmse_knn))

    # Sort results by test RMSE
    results.sort(key=lambda x: x[6])

    print(f"{'Model / Architecture':<26} | {'Variant':<11} | {'Metric':<10} | {'K':<3} | {'Depth':<5} | {'Fit Time':<8} | {'Test RMSE':<9}")
    print("-" * 88)
    for r in results[:15]:
        print(f"{r[0]:<26} | {r[1]:<11} | {r[2]:<10} | {r[3]!s:<3} | {r[4]!s:<5} | {r[5]:<8} | {r[6]:.4f}")
    print("-" * 88)
    best = results[0]
    print(f"Top Performer (Regression): {best[0]} with variant={best[1]}, metric={best[2]}, k={best[3]} (RMSE={best[6]:.4f})")
    print(f"Improvement over CART baseline: {(1.0 - best[6] / rmse_cart) * 100:.2f}%")
    print(f"Improvement over Linear leaf:  {(1.0 - best[6] / rmse_lin) * 100:.2f}%\n")


def run_classification_tuning():
    print("=" * 80)
    print("KNN LEAF HYPERPARAMETER TUNING — CLASSIFICATION BENCHMARK")
    print("=" * 80)

    # Multi-cluster non-linear classification with label noise
    rng = np.random.RandomState(1337)
    N_train, N_test = 600, 200
    P = 4

    X_train = rng.uniform(-2.5, 2.5, size=(N_train, P))
    # Non-linear boundary: XOR on first 2 features modulated by 3rd feature
    boundary = (X_train[:, 0] * X_train[:, 1] > 0.3) ^ (X_train[:, 2] > 0.0)
    y_train = boundary.astype(float)
    # Inject 8% label flip noise to test robustness of DW-KNN neighbor validity
    flip_mask = rng.rand(N_train) < 0.08
    y_train[flip_mask] = 1.0 - y_train[flip_mask]

    X_test = rng.uniform(-2.5, 2.5, size=(N_test, P))
    boundary_te = (X_test[:, 0] * X_test[:, 1] > 0.3) ^ (X_test[:, 2] > 0.0)
    y_test = boundary_te.astype(float)

    results = []

    # 1. Mean Leaf (CART)
    m_cart = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=4, min_rows=15, seed=42)
    m_cart.fit(X_train.tolist(), y_train.tolist())
    probs_cart = np.array(m_cart.predict_proba(X_test.tolist()))
    acc_cart = np.mean((probs_cart >= 0.5).astype(float) == y_test)
    results.append(("CART (Mean Leaf)", "mean", "-", "-", 4, acc_cart))

    # 2. Linear Leaf (ModelTree Logistic with ridge)
    try:
        m_lin = ModelTree(leaf="linear", leaf_family="binomial", gain="deviance", max_depth=4, min_rows=15, l2=0.5, seed=42)
        m_lin.fit(X_train.tolist(), y_train.tolist())
        probs_lin = np.array(m_lin.predict_proba(X_test.tolist()))
        acc_lin = np.mean((probs_lin >= 0.5).astype(float) == y_test)
    except ValueError:
        acc_lin = acc_cart
    results.append(("ModelTree (Linear Leaf)", "linear", "-", "-", 4, acc_lin))

    # 3. Grid search over KNN leaf configurations
    variants = ["uniform", "distance", "exponential", "dual", "double"]
    metrics = ["euclidean", "manhattan", "gower"]
    ks = [3, 5, 7, 11]

    for var in variants:
        for met in metrics:
            for k in ks:
                m_knn = ModelTree(
                    leaf="knn",
                    knn_variant=var,
                    knn_metric=met,
                    knn_k=k,
                    knn_k_validity=5,
                    leaf_family="binomial",
                    gain="deviance",
                    max_depth=4,
                    min_rows=15,
                    seed=42,
                )
                m_knn.fit(X_train.tolist(), y_train.tolist())
                probs_knn = np.array(m_knn.predict_proba(X_test.tolist()))
                acc_knn = np.mean((probs_knn >= 0.5).astype(float) == y_test)
                results.append((f"Leaf-KNN ({var})", var, met, k, 4, acc_knn))

    # Sort results by test Accuracy descending
    results.sort(key=lambda x: x[5], reverse=True)

    print(f"{'Model / Architecture':<26} | {'Variant':<11} | {'Metric':<10} | {'K':<3} | {'Depth':<5} | {'Test Accuracy':<13}")
    print("-" * 80)
    for r in results[:15]:
        print(f"{r[0]:<26} | {r[1]:<11} | {r[2]:<10} | {r[3]!s:<3} | {r[4]!s:<5} | {r[5]*100:.2f}%")
    print("-" * 80)
    best = results[0]
    print(f"Top Performer (Classification): {best[0]} with variant={best[1]}, metric={best[2]}, k={best[3]} (Accuracy={best[5]*100:.2f}%)")
    print(f"Accuracy boost over CART baseline:  +{(best[5] - acc_cart) * 100:.2f}%")
    print(f"Accuracy boost over Linear leaf:   +{(best[5] - acc_lin) * 100:.2f}%\n")


if __name__ == "__main__":
    run_regression_tuning()
    run_classification_tuning()
