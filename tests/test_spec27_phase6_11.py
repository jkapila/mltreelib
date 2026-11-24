"""Tests for Spec 27 Phase 6 (Group Lasso) and Phase 11 (Penalized B-Spline GAM)."""

import math
import numpy as np
import pytest

from mltreelib import GamRegressor, MarsRegressor, ModelTree


def test_gam_regressor_sine_fit_and_predict():
    """Verify GamRegressor fits 1D sine wave and estimates degrees of freedom."""
    np.random.seed(42)
    n = 200
    x = np.sort(np.random.uniform(0.0, 2.0 * math.pi, n))
    noise = np.random.normal(0.0, 0.1, n)
    y = np.sin(x) + noise

    gam = GamRegressor(n_splines=12, degree=3)
    gam.fit(x, y)

    y_hat = gam.predict(x)
    assert len(y_hat) == n
    mse = float(np.mean((y - y_hat) ** 2))
    assert mse < 0.05, f"GAM sine MSE too high: {mse}"

    # Verify diagnostic properties
    edf = gam.edf()
    assert 2.0 <= edf <= 12.0, f"Unexpected EDF: {edf}"
    assert gam.gcv() > 0.0
    assert gam.lambda_val() > 0.0
    assert gam.n_terms() == 13
    assert isinstance(gam.intercept(), float)
    assert len(gam.coefs()[0]) == 12


def test_gam_regressor_vs_linear_fit():
    """Verify GAM outperforms flat linear model on non-linear target."""
    np.random.seed(123)
    n = 150
    x = np.sort(np.random.uniform(-3.0, 3.0, n))
    y = np.cos(x) + np.random.normal(0.0, 0.05, n)

    gam = GamRegressor(n_splines=10, degree=3)
    gam.fit(x, y)
    y_gam = gam.predict(x)
    gam_mse = float(np.mean((y - y_gam) ** 2))

    # Baseline linear fit
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    y_lin = intercept + slope * x
    lin_mse = float(np.mean((y - y_lin) ** 2))

    assert gam_mse < lin_mse * 0.2, f"GAM MSE {gam_mse} should beat linear {lin_mse}"


def test_model_tree_with_group_lasso():
    """Verify ModelTree leaf linear solver accepts grouped features and group_l2 penalty."""
    np.random.seed(42)
    n = 150
    x1 = np.random.uniform(-1.0, 1.0, n)
    x2 = np.random.uniform(-1.0, 1.0, n)
    x3 = np.random.uniform(-1.0, 1.0, n)
    x4 = np.random.uniform(-1.0, 1.0, n)
    # y depends only on x1 and x2
    y = 2.0 * x1 - 1.5 * x2 + np.random.normal(0.0, 0.1, n)
    X = np.column_stack([x1, x2, x3, x4])

    groups = {"signal": [0, 1], "noise": [2, 3]}
    tree = ModelTree(
        leaf="linear",
        alpha=1.0,
        l1=0.0,
        group_l2=5.0,
        groups=groups,
        max_depth=2,
        min_rows=20,
    )
    tree.fit(X, y)

    y_pred = tree.predict(X)
    assert len(y_pred) == n
    assert np.all(np.isfinite(y_pred))
    mse = float(np.mean((y - y_pred) ** 2))
    assert mse < 0.2


def test_model_tree_hierarchical_shrinkage():
    """Verify hierarchical leaf shrinkage stabilizes predictions toward parent nodes."""
    np.random.seed(99)
    n = 120
    X = np.random.normal(0.0, 1.0, (n, 3))
    y = 3.0 * X[:, 0] + np.random.normal(0.0, 0.5, n)

    # Unshrunk model tree
    tree_raw = ModelTree(leaf="linear", shrinkage=0.0, max_depth=3, min_rows=15)
    tree_raw.fit(X, y)
    pred_raw = tree_raw.predict(X)

    # Shrunk model tree (shrinkage = 10.0)
    tree_shrunk = ModelTree(leaf="linear", shrinkage=10.0, max_depth=3, min_rows=15)
    tree_shrunk.fit(X, y)
    pred_shrunk = tree_shrunk.predict(X)

    assert len(pred_raw) == n
    assert len(pred_shrunk) == n
    assert np.all(np.isfinite(pred_raw))
    assert np.all(np.isfinite(pred_shrunk))
    # Predictions should be measurably shrunk toward the sample mean
    var_raw = float(np.var(pred_raw))
    var_shrunk = float(np.var(pred_shrunk))
    assert var_shrunk <= var_raw + 1e-6, "Shrinkage should dampen leaf prediction variance"


def test_mars_regressor_quantile_knots():
    """Verify MarsRegressor accepts quantile_knots and ridge_lambda options."""
    np.random.seed(77)
    n = 100
    X = np.random.exponential(1.0, (n, 2))
    y = 2.0 * np.sqrt(X[:, 0] + 0.1) + X[:, 1] ** 1.5

    mars = MarsRegressor(
        max_terms=12,
        degree=1,
        quantile_knots=True,
        ridge_lambda=1e-3,
    )
    mars.fit(X, y)
    y_pred = mars.predict(X)
    assert len(y_pred) == n
    assert np.all(np.isfinite(y_pred))
    mse = float(np.mean((y - y_pred) ** 2))
    assert mse < 1.0
