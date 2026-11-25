"""Non-parametric KNN and Double-Weighted KNN (DW-KNN) leaf models showcase.

Demonstrates:
1. Non-parametric KNN leaf architecture in ModelTree (leaf="knn").
2. 5 KNN Weighting Variants:
   - uniform: Standard uniform k-nearest neighbors voting/averaging.
   - distance: Dudani inverse distance weighting (1 / (d + eps)^p).
   - exponential: Kernel distance weighting exp(-gamma * d).
   - dual: Gou et al. dual-weighting (distance * compactness).
   - double: DW-KNN (Pathak, Karthik, Madan, Kapila; arXiv:2512.08956)
     combining distance consistency with precomputed neighbor validity.
3. 10 Distance Metrics (euclidean, manhattan, chebyshev, gower, canberra, etc.).
4. Missing value and mixed-type resilience using Gower distance.
5. Decoupled leaf retraining using LeafControl.

Run:
    python examples/guides/knn_leaf_showcase.py
"""

import numpy as np
from mltreelib import LeafControl, ModelTree


def showcase_regression_variants():
    print("=" * 75)
    print("1. Leaf-KNN Regression & Weighting Variants Comparison")
    print("=" * 75)

    rng = np.random.RandomState(42)
    N = 250
    # 2D non-linear continuous ripple surface
    X = rng.uniform(-3.0, 3.0, size=(N, 2))
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = np.sin(r) * 2.5 + rng.normal(0, 0.1, size=N)

    X_test = rng.uniform(-3.0, 3.0, size=(60, 2))
    r_te = np.sqrt(X_test[:, 0] ** 2 + X_test[:, 1] ** 2)
    y_test = np.sin(r_te) * 2.5 + rng.normal(0, 0.1, size=60)

    # Baselines
    cart = ModelTree(leaf="mean", max_depth=3, min_rows=10, seed=42)
    cart.fit(X.tolist(), y.tolist())
    rmse_cart = np.sqrt(np.mean((y_test - np.array(cart.predict(X_test.tolist()))) ** 2))
    print(f"Baseline CART (Mean Leaf) RMSE        : {rmse_cart:.4f}")

    lin = ModelTree(leaf="linear", max_depth=3, min_rows=10, l2=1e-3, seed=42)
    lin.fit(X.tolist(), y.tolist())
    rmse_lin = np.sqrt(np.mean((y_test - np.array(lin.predict(X_test.tolist()))) ** 2))
    print(f"Baseline ModelTree (Linear Leaf) RMSE  : {rmse_lin:.4f}\n")

    print(f"{'KNN Variant':<18} | {'Metric':<12} | {'k':<3} | {'Test RMSE':<10} | {'vs CART':<10}")
    print("-" * 65)

    variants = ["uniform", "distance", "exponential", "dual", "double"]
    for var in variants:
        tree = ModelTree(
            leaf="knn",
            knn_variant=var,
            knn_metric="euclidean",
            knn_k=5,
            knn_k_validity=5,
            max_depth=3,
            min_rows=10,
            seed=42,
        )
        tree.fit(X.tolist(), y.tolist())
        preds = np.array(tree.predict(X_test.tolist()))
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        diff = (1.0 - rmse / rmse_cart) * 100
        print(f"Leaf-KNN ({var:<11}) | {'euclidean':<12} | 5   | {rmse:.4f}     | {diff:+.2f}%")


def showcase_distance_metrics():
    print("\n" + "=" * 75)
    print("2. Distance Proximity Metrics in Leaf-KNN")
    print("=" * 75)

    rng = np.random.RandomState(1337)
    N = 200
    X = rng.uniform(-2.0, 2.0, size=(N, 4))
    y = X[:, 0] * 2.0 - X[:, 1] * 1.5 + np.cos(X[:, 2]) * 2.0

    X_test = rng.uniform(-2.0, 2.0, size=(40, 4))
    y_test = X_test[:, 0] * 2.0 - X_test[:, 1] * 1.5 + np.cos(X_test[:, 2]) * 2.0

    metrics = [
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "cosine",
        "correlation",
        "canberra",
        "braycurtis",
        "gower",
        "mahalanobis",
    ]

    print(f"{'Distance Metric':<16} | {'Formula / Concept':<36} | {'Test RMSE':<10}")
    print("-" * 70)
    for met in metrics:
        tree = ModelTree(
            leaf="knn",
            knn_variant="double",
            knn_metric=met,
            knn_k=4,
            max_depth=2,
            min_rows=12,
            seed=42,
        )
        tree.fit(X.tolist(), y.tolist())
        preds = np.array(tree.predict(X_test.tolist()))
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        print(f"{met:<16} | {tree.knn_metric:<36} | {rmse:.4f}")


def showcase_dwknn_noise_immunity():
    print("\n" + "=" * 75)
    print("3. DW-KNN Noise Immunity: Precomputed Neighbor Validity")
    print("=" * 75)

    rng = np.random.RandomState(42)
    N_train, N_test = 400, 100
    X_train = rng.uniform(-2.0, 2.0, size=(N_train, 2))
    # True boundary: quadrant XOR
    true_label_tr = ((X_train[:, 0] > 0) ^ (X_train[:, 1] > 0)).astype(float)

    # Corrupt training set with 10% malicious label flip noise
    y_train = true_label_tr.copy()
    flips = rng.rand(N_train) < 0.10
    y_train[flips] = 1.0 - y_train[flips]

    X_test = rng.uniform(-2.0, 2.0, size=(N_test, 2))
    y_test = ((X_test[:, 0] > 0) ^ (X_test[:, 1] > 0)).astype(float)

    # 1. Standard CART
    cart = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=3, min_rows=12, seed=42)
    cart.fit(X_train.tolist(), y_train.tolist())
    p_cart = (np.array(cart.predict_proba(X_test.tolist())) >= 0.5).astype(float)
    acc_cart = np.mean(p_cart == y_test)

    # 2. Standard Uniform KNN Leaf
    knn_uni = ModelTree(leaf="knn", knn_variant="uniform", knn_k=7, leaf_family="binomial", gain="deviance", max_depth=3, min_rows=12, seed=42)
    knn_uni.fit(X_train.tolist(), y_train.tolist())
    p_uni = (np.array(knn_uni.predict_proba(X_test.tolist())) >= 0.5).astype(float)
    acc_uni = np.mean(p_uni == y_test)

    # 3. Double-Weighted KNN (DW-KNN) Leaf
    knn_dw = ModelTree(leaf="knn", knn_variant="double", knn_k=7, knn_k_validity=5, leaf_family="binomial", gain="deviance", max_depth=3, min_rows=12, seed=42)
    knn_dw.fit(X_train.tolist(), y_train.tolist())
    p_dw = (np.array(knn_dw.predict_proba(X_test.tolist())) >= 0.5).astype(float)
    acc_dw = np.mean(p_dw == y_test)

    print("Training Data Label Noise : 10.0% flipped")
    print(f"CART (Mean Leaf) Accuracy  : {acc_cart * 100:.2f}%")
    print(f"Uniform KNN Leaf Accuracy  : {acc_uni * 100:.2f}%")
    print(f"DW-KNN Leaf Accuracy       : {acc_dw * 100:.2f}% (boost: +{(acc_dw - acc_cart)*100:.2f}%)")
    print("-> DW-KNN's precomputed neighbor validity score v_i naturally attenuates noisy neighbors.")


def showcase_gower_missing_data():
    print("\n" + "=" * 75)
    print("4. Gower Distance with Missing (None) Values")
    print("=" * 75)

    X_train = [
        [1.0, 2.0, 3.0],
        [None, 2.1, 3.1],
        [1.1, None, 3.2],
        [5.0, 6.0, 7.0],
        [5.1, 6.1, None],
        [None, 6.2, 7.1],
    ]
    y_train = [10.0, 10.2, 10.5, 50.0, 50.2, 50.5]

    tree = ModelTree(leaf="knn", knn_variant="double", knn_metric="gower", knn_k=2, max_depth=1, min_rows=2, seed=42)
    tree.fit(X_train, y_train)

    query = [
        [1.05, None, 3.05],  # Near first cluster
        [None, 6.05, 7.05],  # Near second cluster
    ]
    preds = tree.predict(query)
    print(f"Query 1 (near cluster 1, missing col 1) -> Predicted: {preds[0]:.2f} (Expected: ~10)")
    print(f"Query 2 (near cluster 2, missing col 0) -> Predicted: {preds[1]:.2f} (Expected: ~50)")


def showcase_leaf_control_retrain():
    print("\n" + "=" * 75)
    print("5. Decoupled Retraining with LeafControl(leaf='knn')")
    print("=" * 75)

    rng = np.random.RandomState(42)
    X = rng.uniform(-2.0, 2.0, size=(100, 3)).tolist()
    y = [float(r[0] > 0) * 10.0 + r[1] * 2.0 for r in X]

    # Initial tree trained with fast constant mean leaves
    tree = ModelTree(leaf="mean", max_depth=2, min_rows=10, seed=42)
    tree.fit(X, y)
    print(f"Initial tree fit with leaf='mean', leaves={tree.n_leaves()}.")

    # Retrain leaves to DW-KNN without altering the tree structure:
    knn_ctrl = LeafControl(
        leaf="knn",
        leaf_family="gaussian",
        knn_variant="double",
        knn_metric="manhattan",
        knn_k=4,
    )
    tree.retrain(X, y, control=knn_ctrl)
    print("Retrained tree leaves to DW-KNN (metric='manhattan', k=4) successfully.")
    preds = tree.predict(X[:3])
    print(f"Predictions from retrained DW-KNN tree: {[round(p, 3) for p in preds]}")


if __name__ == "__main__":
    showcase_regression_variants()
    showcase_distance_metrics()
    showcase_dwknn_noise_immunity()
    showcase_gower_missing_data()
    showcase_leaf_control_retrain()
    print("\nAll KNN leaf showcases completed successfully.")
