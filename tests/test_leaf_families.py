"""Unit tests for newly added leaf families: Probit, Gamma, Tweedie, and Tobit."""

import math
import random
import pytest
from mltreelib import ModelTree, LeafControl


def test_probit_leaf_family_fit_and_retrain():
    """Probit binary classification with standard normal CDF link."""
    rng = random.Random(42)
    n = 200
    x = []
    y = []
    for _ in range(n):
        x0 = rng.uniform(-2.0, 2.0)
        x1 = rng.uniform(-1.0, 1.0)
        # Latent z
        z = 0.5 + 1.2 * x0 - 0.8 * x1
        # Probit probability: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        y_val = 1.0 if rng.random() < prob else 0.0
        x.append([x0, x1])
        y.append(y_val)

    tree = ModelTree(max_depth=1, min_rows=15, leaf="linear", leaf_family="probit", prune="none")
    tree.fit(x, y)

    assert tree.n_leaves() >= 1

    probs = tree.predict_proba(x)
    assert len(probs) == n
    assert all(0.0 <= p <= 1.0 for p in probs)

    labels = tree.predict_label(x, threshold=0.5)
    assert set(labels).issubset({0, 1})

    # Test retraining to probit with LeafControl and dictionary
    tree.retrain(x, y, control={"leaf": "linear", "leaf_family": "probit", "l2": 1e-3})
    leaves = tree.leaves()
    assert leaves[0]["intercept"] is not None


def test_gamma_leaf_family():
    """Gamma leaf family for strictly positive continuous targets (log link)."""
    rng = random.Random(123)
    n = 150
    x = []
    y = []
    for _ in range(n):
        x0 = rng.uniform(-1.0, 1.0)
        x1 = rng.uniform(0.0, 2.0)
        # Strictly positive target with exponential / gamma noise
        mu = math.exp(1.0 + 0.8 * x0 + 0.4 * x1)
        target = rng.gammavariate(alpha=4.0, beta=mu / 4.0)
        assert target > 0.0
        x.append([x0, x1])
        y.append(target)

    tree = ModelTree(max_depth=1, min_rows=10, leaf="linear", leaf_family="gamma", prune="none")
    tree.fit(x, y)

    preds = tree.predict(x)
    assert len(preds) == n
    assert all(p > 0.0 for p in preds)

    # Retrain with gamma dictionary
    retrained = tree.retrain(x, y, mode="external", control={"leaf": "linear", "family": "gamma"})
    preds_ret = retrained.predict(x)
    assert all(p > 0.0 for p in preds_ret)


def test_tweedie_leaf_family():
    """Tweedie leaf family for zero-inflated continuous positive targets."""
    rng = random.Random(456)
    n = 200
    x = []
    y = []
    for _ in range(n):
        x0 = rng.uniform(-1.5, 1.5)
        # 40% exact zeros, 60% positive values
        if rng.random() < 0.40:
            target = 0.0
        else:
            target = math.exp(0.5 + 0.9 * x0) + rng.uniform(0.1, 1.0)
        x.append([x0])
        y.append(target)

    tree = ModelTree(max_depth=1, min_rows=10, leaf="linear", leaf_family="tweedie", prune="none")
    tree.fit(x, y)

    preds = tree.predict(x)
    assert len(preds) == n
    assert all(p >= 0.0 for p in preds)

    # Test retraining to mean tweedie leaf
    tree.retrain(x, y, control={"leaf": "mean", "family": "tweedie"})
    preds_mean = tree.predict(x)
    assert all(p >= 0.0 for p in preds_mean)


def test_tobit_censored_leaf_family():
    """Tobit leaf family for left-censored continuous targets at zero."""
    rng = random.Random(789)
    n = 250
    x = []
    y = []
    for _ in range(n):
        x0 = rng.uniform(-2.0, 2.0)
        # Latent variable with noise
        latent = 0.8 * x0 + rng.gauss(0.0, 0.5)
        # Left-censored at 0
        observed = max(0.0, latent)
        x.append([x0])
        y.append(observed)

    tree = ModelTree(max_depth=1, min_rows=15, leaf="linear", leaf_family="tobit", prune="none")
    tree.fit(x, y)

    leaves = tree.leaves()
    assert len(leaves) >= 1
    assert leaves[0]["intercept"] is not None

    # Latent slope should be positive (recovering 0.8 latent direction)
    coef = leaves[0]["coef"][0]
    assert coef > 0.0

    preds = tree.predict([[1.5], [-1.5]])
    assert preds[0] > preds[1]
