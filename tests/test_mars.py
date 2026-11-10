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
    # Grid with integer points containing 7.0
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
