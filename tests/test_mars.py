import numpy as np
import pytest
from mltreelib import MarsRegressor

def test_mars_regressor_1d_hinge():
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 10.0, 51)
    # y = 3.0 + 2.0 * max(0, x - 5.0)
    y = 3.0 + 2.0 * np.maximum(0.0, x - 5.0)
    X = x.reshape(-1, 1)

    reg = MarsRegressor(degree=1, max_terms=10, penalty=2.0)
    reg.fit(X, y)

    assert reg.n_terms >= 2
    assert reg.rss < 1e-4
    assert reg.gcv < 1e-3

    preds = reg.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-2)

    eq = reg.equation(["x0"])
    assert "h(x0 - " in eq or "h(" in eq

def test_mars_regressor_2d_interaction():
    rng = np.random.default_rng(123)
    grid_1d = np.linspace(0.0, 10.0, 21)
    X0, X1 = np.meshgrid(grid_1d, grid_1d)
    x0 = X0.ravel()
    x1 = X1.ravel()
    y = 5.0 + 2.0 * np.maximum(0.0, x0 - 7.0) * np.maximum(0.0, x1 - 7.0)
    X = np.column_stack([x0, x1])

    reg = MarsRegressor(degree=2, max_terms=15, penalty=2.0)
    reg.fit(X, y)

    assert reg.n_terms >= 2
    preds = reg.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-2)

    eq = reg.equation(["feat_a", "feat_b"])
    assert "feat_a" in eq
    assert "feat_b" in eq

def test_mars_linpreds_and_sample_weights():
    # x0 is linear, x1 is hinged
    rng = np.random.default_rng(42)
    x0 = np.linspace(0.0, 5.0, 30)
    x1 = np.linspace(0.0, 5.0, 30)
    X0, X1 = np.meshgrid(x0, x1)
    X = np.column_stack([X0.ravel(), X1.ravel()])
    y = 1.0 + 2.5 * X[:, 0] + 3.0 * np.maximum(0.0, X[:, 1] - 2.5)

    # Outlier points with 0 weight
    X_bad = np.array([[2.5, 2.5], [1.0, 1.0]])
    y_bad = np.array([1000.0, 2000.0])
    X_all = np.vstack([X, X_bad])
    y_all = np.concatenate([y, y_bad])
    weights = np.ones(len(y_all))
    weights[-2:] = 0.0

    reg = MarsRegressor(degree=1, max_terms=10, linpreds=[0])
    reg.fit(X_all, y_all, sample_weights=weights)

    eq = reg.equation(["col0", "col1"])
    assert "col0" in eq
    assert "h(col0 - " not in eq  # linpreds forced pure linear

    preds = reg.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-2)

def test_mars_glm_binomial_classification():
    x = np.linspace(0.0, 10.0, 100)
    X = x.reshape(-1, 1)
    # y = 1 if x > 5.0 else 0
    y = (x > 5.0).astype(np.float64)

    clf = MarsRegressor(degree=1, max_terms=8, glm="binomial")
    clf.fit(X, y)

    probas = clf.predict_proba(X)
    assert probas[5] < 0.2
    assert probas[-5] > 0.8

    preds = clf.predict(X)
    assert preds[5] < 0.5
    assert preds[-5] > 0.5

    eq = clf.equation(["feat"])
    assert eq.startswith("logit(p) = ")
