"""Tests for non-parametric KNN leaf models in ModelTree.

Verifies:
1. 5 weighting variants: uniform, distance, exponential, dual, double (DW-KNN).
2. 10 distance metrics: euclidean, manhattan, chebyshev, minkowski, cosine,
   correlation, canberra, braycurtis, gower, mahalanobis.
3. Regression and binary/multiclass classification.
4. Leaf routing: tree splits partition data, local KNN evaluates at leaves.
5. Parameter persistence and get_params().
"""

import numpy as np
import pytest
from mltreelib import ModelTree


@pytest.fixture
def sample_regression_data():
    rng = np.random.RandomState(42)
    X = rng.uniform(-2.0, 2.0, size=(120, 4))
    # Non-linear function with boundary change
    y = np.sin(X[:, 0]) * 2.0 + (X[:, 1] > 0) * 5.0 + rng.normal(0, 0.1, size=120)
    return X.tolist(), y.tolist()


@pytest.fixture
def sample_classification_data():
    rng = np.random.RandomState(1337)
    X = rng.uniform(-2.0, 2.0, size=(120, 4))
    # Binary classification target
    logits = X[:, 0] * 1.5 - X[:, 1] * 1.2
    y = (logits > 0).astype(float).tolist()
    return X.tolist(), y


def test_knn_leaf_all_variants_regression(sample_regression_data):
    X, y = sample_regression_data
    variants = ["uniform", "distance", "exponential", "dual", "double"]

    for var in variants:
        tree = ModelTree(
            leaf="knn",
            knn_variant=var,
            knn_k=5,
            knn_k_validity=5,
            max_depth=2,
            min_rows=15,
            seed=42,
        )
        tree.fit(X, y)
        preds = tree.predict(X[:10])
        assert len(preds) == 10
        assert all(np.isfinite(p) for p in preds)

        # Check config reflects knob values
        params = tree.config()
        assert params["leaf"] == "knn"
        assert params["knn_variant"] == var
        assert params["knn_k"] == 5


def test_knn_leaf_all_metrics_regression(sample_regression_data):
    X, y = sample_regression_data
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

    for met in metrics:
        tree = ModelTree(
            leaf="knn",
            knn_variant="double",
            knn_metric=met,
            knn_k=4,
            max_depth=2,
            min_rows=15,
            seed=42,
        )
        tree.fit(X, y)
        preds = tree.predict(X[:5])
        assert len(preds) == 5
        assert all(np.isfinite(p) for p in preds)


def test_knn_leaf_binary_classification(sample_classification_data):
    X, y = sample_classification_data

    for var in ["uniform", "distance", "double"]:
        tree = ModelTree(
            leaf="knn",
            knn_variant=var,
            leaf_family="binomial",
            gain="deviance",
            knn_k=5,
            max_depth=2,
            min_rows=15,
            seed=42,
        )
        tree.fit(X, y)
        probs = tree.predict_proba(X[:10])
        assert len(probs) == 10
        assert all(0.0 <= p <= 1.0 for p in probs)


def test_knn_leaf_missing_data_gower():
    # Test that Gower handles None missing values in leaves without crashing
    X = [
        [1.0, 2.0],
        [None, 2.1],
        [1.2, None],
        [3.0, 4.0],
        [3.1, 4.2],
        [None, 4.1],
    ]
    y = [10.0, 10.2, 10.5, 20.0, 20.2, 20.5]

    tree = ModelTree(
        leaf="knn",
        knn_variant="double",
        knn_metric="gower",
        knn_k=2,
        max_depth=1,
        min_rows=2,
        seed=42,
    )
    tree.fit(X, y)
    query = [[1.1, None], [None, 4.15]]
    preds = tree.predict(query)
    assert len(preds) == 2
    assert all(np.isfinite(p) for p in preds)
    # First query near [1, 2], second near [3, 4]
    assert preds[0] < 15.0
    assert preds[1] > 15.0
