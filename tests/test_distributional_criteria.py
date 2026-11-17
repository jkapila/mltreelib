import pytest
from mltreelib import ModelTree


def test_tweedie_split():
    # 60 samples: first 30 zero-inflated with occasional small values, next 30 positive
    X = [[0.1 * i] if i < 30 else [10.0 + 0.1 * i] for i in range(60)]
    y = [1.5 if i % 5 == 0 else 0.0 for i in range(30)] + [12.0 + (i % 3) for i in range(30, 60)]
    
    tree = ModelTree(max_depth=1, min_rows=5, gain="tweedie", tweedie_p=1.5, prune="none")
    tree.fit(X, y)
    assert tree.n_leaves() >= 2
    preds = tree.predict([[1.0], [15.0]])
    assert preds[0] < preds[1]


def test_heteroscedastic_split():
    # Detects variance break where means are identical (0.0)
    X = [[0.1 * i] if i < 30 else [10.0 + 0.1 * i] for i in range(60)]
    y = [0.05 * (1 if i % 2 == 0 else -1) for i in range(30)] + [5.0 * (1 if i % 2 == 0 else -1) for i in range(30, 60)]
    
    tree = ModelTree(max_depth=1, min_rows=5, gain="heteroscedastic", prune="none")
    tree.fit(X, y)
    assert tree.n_leaves() >= 2


def test_tobit_split():
    # Left-censored continuous data at threshold 0.0
    X = [[0.1 * i] if i < 30 else [10.0 + 0.1 * i] for i in range(60)]
    y = [0.0 for _ in range(30)] + [2.0 + 0.2 * i for i in range(30, 60)]
    
    tree = ModelTree(max_depth=1, min_rows=5, gain="tobit", tobit_threshold=0.0, prune="none")
    tree.fit(X, y)
    assert tree.n_leaves() >= 2


def test_cv_split():
    # Coefficient of variation reduction
    X = [[0.1 * i] if i < 30 else [10.0 + 0.1 * i] for i in range(60)]
    y = [10.0 + 0.1 * (i % 2) for i in range(30)] + [1.0 + 3.0 * (i % 2) for i in range(30, 60)]
    
    tree = ModelTree(max_depth=1, min_rows=5, gain="cv", prune="none")
    tree.fit(X, y)
    assert tree.n_leaves() >= 2


def test_martingale_split():
    # Martingale residuals
    X = [[0.1 * i] if i < 30 else [10.0 + 0.1 * i] for i in range(60)]
    y = [-0.8 for _ in range(30)] + [0.8 for _ in range(30, 60)]
    
    tree = ModelTree(max_depth=1, min_rows=5, gain="martingale", prune="none")
    tree.fit(X, y)
    assert tree.n_leaves() >= 2
