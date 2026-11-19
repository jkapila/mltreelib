import numpy as np
import pytest
from mltreelib import ModelTree


def test_mmd_vector_targets_orthogonal_covariances():
    # Spec 12 requirement:
    # Two 10-dimensional Gaussian regimes with identical means (0.0) but orthogonal covariances
    # Standard variance reduction fails completely (mean difference = 0).
    # MMD tree isolates regimes along x0.
    rng = np.random.default_rng(42)
    n = 240
    d = 10
    x0 = np.linspace(-2, 2, n)
    x1_noise = rng.uniform(-1, 1, n)
    X = np.column_stack([x0, x1_noise])

    Y = np.zeros((n, d))
    for i in range(n):
        if x0[i] <= 0:
            # Regime 1: High variance on first 5 dimensions, small on last 5
            Y[i, :5] = rng.normal(0, 2.5, size=5)
            Y[i, 5:] = rng.normal(0, 0.1, size=5)
        else:
            # Regime 2: Small variance on first 5 dimensions, high on last 5
            Y[i, :5] = rng.normal(0, 0.1, size=5)
            Y[i, 5:] = rng.normal(0, 2.5, size=5)

    tree = ModelTree(
        leaf="mean",
        gain="mmd",
        mmd_kernel="rbf",
        max_depth=2,
        min_rows=15,
        prune="none",
    )
    tree.fit(X, Y)

    splits = tree.splits()
    assert len(splits) > 0, "MMD tree must split on distribution difference"
    root_split = tree.root_split()
    assert root_split is not None
    assert root_split["col"] == "x0", f"MMD root must split on x0, got {root_split['col']}"
    assert abs(root_split["threshold"] - 0.0) < 0.35

    # Check vector predictions
    test_x = [[-1.0, 0.0], [1.0, 0.0]]
    vec_preds = tree.predict_vector(test_x)
    assert len(vec_preds) == 2
    assert len(vec_preds[0]) == d
    assert len(vec_preds[1]) == d

    # Also test predict(kind="vector") and default predict
    preds_default = tree.predict(test_x)
    assert len(preds_default) == 2
    assert len(preds_default[0]) == d


def test_energy_distance_vector_targets():
    # Multi-output vector targets with Energy Distance
    rng = np.random.default_rng(123)
    n = 200
    d = 4
    x0 = np.linspace(-2, 2, n)
    x1_noise = rng.uniform(-1, 1, n)
    X = np.column_stack([x0, x1_noise])

    Y = np.zeros((n, d))
    for i in range(n):
        center = -5.0 if x0[i] <= 0 else 5.0
        Y[i] = center + rng.normal(0, 0.2, size=d)

    tree = ModelTree(
        leaf="mean",
        gain="energy_distance",
        max_depth=2,
        min_rows=15,
        prune="none",
    )
    tree.fit(X, Y)

    splits = tree.splits()
    assert len(splits) > 0
    assert splits[0]["col"] == "x0"

    test_x = [[-1.0, 0.5], [1.0, -0.5]]
    preds = tree.predict_vector(test_x)
    assert len(preds) == 2
    for val in preds[0]:
        assert abs(val - (-5.0)) < 0.5
    for val in preds[1]:
        assert abs(val - 5.0) < 0.5


def test_scalar_metric_splits():
    # 1D response with EnergyDistance and MMD
    rng = np.random.default_rng(777)
    n = 120
    x0 = np.linspace(-1, 1, n)
    y = np.where(x0 <= 0, -4.0, 4.0) + rng.normal(0, 0.1, n)
    X = x0.reshape(-1, 1)

    tree_energy = ModelTree(gain="energy", max_depth=1, min_rows=10, prune="none")
    tree_energy.fit(X, y)
    assert len(tree_energy.splits()) == 1

    tree_mmd = ModelTree(gain="mmd", mmd_kernel="rbf", mmd_bandwidth=1.0, max_depth=1, min_rows=10, prune="none")
    tree_mmd.fit(X, y)
    assert len(tree_mmd.splits()) == 1
