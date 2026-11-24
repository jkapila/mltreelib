"""Tests for Spec 27 Phase 3: Huber Regression with Robust Scale."""

import pytest
from mltreelib import LeafControl, ModelTree, Metrics


def test_huber_leaf_control_attributes():
    """Verify LeafControl supports huber family and huber_delta parameter."""
    ctrl = LeafControl(
        leaf="linear",
        leaf_family="huber",
        huber_delta=1.5,
        l2=0.05,
    )
    assert ctrl.leaf == "linear"
    assert ctrl.leaf_family == "huber"
    assert ctrl.huber_delta == 1.5
    assert ctrl.l2 == 0.05


def test_huber_linear_model_tree_outlier_resistance():
    """Verify that Huber linear model trees resist severe response outliers compared to OLS."""
    # Data: y = 3.0 * x1 - 2.0 * x2 + 10.0
    import random
    rng = random.Random(42)

    n_train = 300
    X_train = []
    y_train = []
    for i in range(n_train):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        clean_y = 10.0 + 3.0 * x1 - 2.0 * x2 + rng.gauss(0.0, 0.1)
        # Inject 10% contamination with severe outliers (+/- 100.0)
        if i % 10 == 0:
            contaminated_y = clean_y + (100.0 if i % 20 == 0 else -100.0)
            y_train.append(contaminated_y)
        else:
            y_train.append(clean_y)
        X_train.append([x1, x2])

    # Clean held-out test data
    n_test = 100
    X_test = []
    y_test = []
    for _ in range(n_test):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        y_test.append(10.0 + 3.0 * x1 - 2.0 * x2)
        X_test.append([x1, x2])

    # Fit Huber Linear Model Tree
    huber_tree = ModelTree(
        leaf="linear",
        leaf_family="huber",
        huber_delta=1.345,
        max_depth=2,
        min_rows=20,
        l2=0.01,
    )
    huber_tree.fit(X_train, y_train)
    huber_preds = huber_tree.predict(X_test)

    # Fit standard Gaussian OLS Linear Model Tree
    ols_tree = ModelTree(
        leaf="linear",
        leaf_family="gaussian",
        max_depth=2,
        min_rows=20,
        l2=0.01,
    )
    ols_tree.fit(X_train, y_train)
    ols_preds = ols_tree.predict(X_test)

    metrics = Metrics()
    huber_mae = metrics.regression(y_test, huber_preds)["mae"]
    ols_mae = metrics.regression(y_test, ols_preds)["mae"]

    # Huber tree must significantly outperform OLS on outlier-contaminated data
    assert huber_mae < 1.0, f"Huber MAE should be small on clean test set, got {huber_mae}"
    assert huber_mae < ols_mae * 0.3, (
        f"Huber MAE ({huber_mae:.3f}) should be at least 3x lower than OLS MAE ({ols_mae:.3f})"
    )


def test_huber_delta_validation():
    """Verify invalid huber_delta values are rejected."""
    with pytest.raises(ValueError, match="huber_delta must be positive"):
        ModelTree(leaf="linear", leaf_family="huber", huber_delta=-0.5).fit(
            [[1.0, 2.0]], [1.0]
        )


def test_huber_leaf_family_rejection_for_unsupported_shapes():
    """Verify GLM non-Gaussian families reject leaf='huber'."""
    with pytest.raises(ValueError, match="non-gaussian families"):
        ModelTree(leaf="huber", leaf_family="binomial").fit(
            [[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0]
        )


def test_huber_0d_leaf_and_minimum_support_fallback():
    """Verify 0D huber leaf works and small leaves fall back gracefully to 0D location."""
    # 0D Huber constant leaf with 30 observations and 1 outlier
    X = [[float(i)] for i in range(30)]
    y = [10.0 + 0.1 * (i % 3) for i in range(29)] + [1000.0]
    mean_y = sum(y) / len(y)  # ~43.0

    tree_0d = ModelTree(leaf="huber", max_depth=0)
    tree_0d.fit(X, y)
    preds_0d = tree_0d.predict(X)
    assert len(preds_0d) == 30
    # 0D Huber location pulls down significantly from mean 43.0 towards 10.0
    assert preds_0d[0] < 25.0 < mean_y

    # Multivariate Huber linear leaf with robust scale estimates ~10.08
    tree_linear = ModelTree(leaf="linear", leaf_family="huber", max_depth=0)
    tree_linear.fit(X, y)
    preds_linear = tree_linear.predict(X)
    assert abs(preds_linear[0] - 10.0) < 0.5, f"Expected ~10.0, got {preds_linear[0]}"

    # Minimum support fallback: leaf rows < features + 1
    # 3 features => requires 4 rows for linear fit. min_rows=2 allows smaller leaf.
    X_small = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    y_small = [1.0, 2.0, 3.0]
    tree_small = ModelTree(
        leaf="linear", leaf_family="huber", max_depth=2, min_rows=2
    )
    tree_small.fit(X_small, y_small)
    preds_small = tree_small.predict(X_small)
    assert len(preds_small) == 3
    for p in preds_small:
        assert isinstance(p, float)
