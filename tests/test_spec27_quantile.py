"""Tests for Spec 27 Phase 2: Quantile regression, linear leaf, and crossing diagnostics."""

import pytest
from mltreelib import GrowthControl, LeafControl, ModelTree, detect_quantile_crossings


def test_leaf_control_accepts_quantile_family_and_tau():
    ctrl = LeafControl(leaf="linear", leaf_family="quantile", tau=0.85)
    assert ctrl.leaf == "linear"
    assert ctrl.leaf_family == "quantile"
    assert ctrl.tau == 0.85

    # Check flat kwargs on ModelTree
    tree = ModelTree(leaf="linear", leaf_family="quantile", tau=0.75, max_depth=2)
    assert tree.params["leaf"] == "linear"
    assert tree.params["leaf_family"] == "quantile"
    assert tree.params["tau"] == 0.75


def test_model_tree_quantile_linear_leaf_fit_and_monotonicity():
    # Synthetic linear dataset: y = 2.0 + 3.0 * x + noise
    X = [[float(i) / 20.0] for i in range(100)]
    y = [2.0 + 3.0 * row[0] + ((i % 5) - 2) * 0.5 for i, row in enumerate(X)]

    # Fit tau = 0.1, 0.5, 0.9
    tree_10 = ModelTree(leaf="linear", leaf_family="quantile", tau=0.1, max_depth=2, min_rows=15)
    tree_50 = ModelTree(leaf="linear", leaf_family="quantile", tau=0.5, max_depth=2, min_rows=15)
    tree_90 = ModelTree(leaf="linear", leaf_family="quantile", tau=0.9, max_depth=2, min_rows=15)

    tree_10.fit(X, y)
    tree_50.fit(X, y)
    tree_90.fit(X, y)

    eval_X = [[0.5], [1.5], [2.5], [3.5], [4.5]]
    preds_10 = tree_10.predict(eval_X)
    preds_50 = tree_50.predict(eval_X)
    preds_90 = tree_90.predict(eval_X)

    assert len(preds_10) == 5
    assert len(preds_50) == 5
    assert len(preds_90) == 5

    # Quantile ordering: Q_0.1 <= Q_0.5 <= Q_0.9 on typical points
    for p10, p50, p90 in zip(preds_10, preds_50, preds_90):
        assert p10 <= p50 + 0.1, f"Expected Q_0.1 ({p10}) <= Q_0.5 ({p50})"
        assert p50 <= p90 + 0.1, f"Expected Q_0.5 ({p50}) <= Q_0.9 ({p90})"


def test_quantile_minimum_support_fallback():
    # Small dataset where leaf rows < P + 1
    X = [[float(i), float(i * 2), float(i * 3)] for i in range(10)]
    y = [float(i) for i in range(10)]

    tree = ModelTree(leaf="linear", leaf_family="quantile", tau=0.5, max_depth=3, min_rows=2)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert len(preds) == 10
    assert all(isinstance(p, float) for p in preds)


def test_detect_quantile_crossings_python_binding():
    taus = [0.1, 0.5, 0.9]

    # Monotone predictions (no crossings)
    clean_preds = [
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [2.0, 3.0, 4.0],
    ]
    rep_clean = detect_quantile_crossings(clean_preds, taus)
    assert rep_clean["n_crossings"] == 0
    assert rep_clean["n_violating_rows"] == 0
    assert rep_clean["crossing_rate"] == 0.0
    assert rep_clean["max_violation"] == 0.0
    assert rep_clean["crossings"] == []

    # Crossing at row 1
    crossed_preds = [
        [1.0, 2.0, 3.0],
        [0.5, 2.8, 2.0],  # Q_0.5 (2.8) > Q_0.9 (2.0)
        [2.0, 3.0, 4.0],
    ]
    rep_crossed = detect_quantile_crossings(crossed_preds, taus)
    assert rep_crossed["n_crossings"] == 1
    assert rep_crossed["n_violating_rows"] == 1
    assert rep_crossed["crossing_rate"] > 0.0
    assert abs(rep_crossed["max_violation"] - 0.8) < 1e-6
    assert rep_crossed["crossings"] == [(1, 1, 2)]
