"""Tests for Equalized Odds fairness penalty, ExtraTrees randomized splits, and EBM FAST interaction screening."""

import pytest
import random
from mltreelib import ModelTree, fast_interaction_screen


def test_equalized_odds_fairness_penalty():
    # Setup dataset:
    # Col 0: Protected attribute S in {0, 1}
    # Col 1: High correlation with S (proxy feature: X1 = S + small noise) -> unfair
    # Col 2: Fair predictor X2 in [-1, 1], completely independent of S, also predicts Y
    rng = random.Random(42)
    n = 300
    X = []
    y = []

    for _ in range(n):
        s = 1.0 if rng.random() < 0.5 else 0.0
        x1_proxy = s + (rng.random() - 0.5) * 0.1
        x2_fair = rng.random() * 2.0 - 1.0

        target = (6.0 if x2_fair > 0.0 else 0.0) + 10.0 * x1_proxy
        X.append([s, x1_proxy, x2_fair])
        y.append(target)

    # 1. Unconstrained tree (fairness_penalty=0.0)
    tree_uncon = ModelTree(max_depth=1, min_rows=30, fairness_penalty=0.0, protected_col=0)
    tree_uncon.fit(X, y, col_names=["protected", "proxy", "fair"])
    splits_uncon = tree_uncon.splits()
    assert len(splits_uncon) == 1
    # Raw tree splits on proxy or protected
    assert splits_uncon[0]["col"] in ("proxy", "protected", 0, 1)

    # 2. Equalized Odds constrained tree
    tree_eo = ModelTree(
        max_depth=1,
        min_rows=30,
        fairness_penalty=0.85,
        fairness_metric="equalized_odds",
        protected_col=0,
    )
    tree_eo.fit(X, y, col_names=["protected", "proxy", "fair"])
    splits_eo = tree_eo.splits()
    assert len(splits_eo) == 1
    # Equalized odds forces tree to choose fair feature x2
    assert splits_eo[0]["col"] in ("fair", 2)


def test_fairness_metric_invalid_value():
    with pytest.raises(ValueError, match="Unknown fairness_metric"):
        ModelTree(fairness_penalty=0.5, fairness_metric="invalid_metric")


def test_extratrees_random_cuts():
    rng = random.Random(123)
    X = [[rng.uniform(0.0, 100.0), rng.uniform(-50.0, 50.0)] for _ in range(150)]
    y = [1.0 if row[0] > 50.0 else 0.0 for row in X]

    # Standard CART tree picks optimal cut
    cart_tree = ModelTree(leaf="mean", max_depth=2, min_rows=10, extratrees=False, seed=42)
    cart_tree.fit(X, y, col_names=["x0", "x1"])
    cart_cuts = [s["threshold"] for s in cart_tree.splits() if "threshold" in s]

    # ExtraTrees with seed 42
    et_tree_1 = ModelTree(leaf="mean", max_depth=2, min_rows=10, extratrees=True, seed=42)
    et_tree_1.fit(X, y, col_names=["x0", "x1"])
    et_cuts_1 = [s["threshold"] for s in et_tree_1.splits() if "threshold" in s]

    # ExtraTrees with seed 999
    et_tree_2 = ModelTree(leaf="mean", max_depth=2, min_rows=10, extratrees=True, seed=999)
    et_tree_2.fit(X, y, col_names=["x0", "x1"])
    et_cuts_2 = [s["threshold"] for s in et_tree_2.splits() if "threshold" in s]

    # ExtraTrees cuts must be deterministic for identical seeds
    et_tree_1_repeat = ModelTree(leaf="mean", max_depth=2, min_rows=10, extratrees=True, seed=42)
    et_tree_1_repeat.fit(X, y, col_names=["x0", "x1"])
    et_cuts_1_repeat = [s["threshold"] for s in et_tree_1_repeat.splits() if "threshold" in s]
    assert et_cuts_1 == et_cuts_1_repeat

    # ExtraTrees cuts should differ from CART cuts and differ between seeds
    assert len(et_cuts_1) > 0
    assert et_cuts_1 != cart_cuts
    assert et_cuts_1 != et_cuts_2

    # Predictions must run cleanly
    preds = et_tree_1.predict(X[:10])
    assert len(preds) == 10
    assert all(0.0 <= p <= 1.0 for p in preds)


def test_fast_interaction_screening():
    rng = random.Random(42)
    n = 300
    X = []
    y = []

    # Strong non-linear interaction between x0 and x1: y = 2*x0 + 2*x1 + 8*(x0*x1)
    # x2, x3 are pure noise
    for _ in range(n):
        x0 = rng.uniform(-1.0, 1.0)
        x1 = rng.uniform(-1.0, 1.0)
        x2 = rng.uniform(-1.0, 1.0)
        x3 = rng.uniform(-1.0, 1.0)
        y_val = 2.0 * x0 + 2.0 * x1 + 8.0 * (x0 * x1)
        X.append([x0, x1, x2, x3])
        y.append(y_val)

    # 1. Test standalone fast_interaction_screen
    col_names = ["feat0", "feat1", "noise2", "noise3"]
    screened = fast_interaction_screen(X, y, col_names=col_names, top_k=5, min_samples=10)
    assert len(screened) > 0
    top_pair = screened[0]
    pair_names = {top_pair["col1_name"], top_pair["col2_name"]}
    assert pair_names == {"feat0", "feat1"}, f"Top interaction pair should be feat0 & feat1, got {pair_names}"
    assert top_pair["score"] > 0.05

    # 2. Test ModelTree(ebm_fast=True)
    tree = ModelTree(max_depth=3, min_rows=10, ebm_fast=True)
    tree.fit(X, y, col_names=col_names)
    tree_interactions = tree.fast_interactions()
    assert len(tree_interactions) > 0
    top_tree_pair = tree_interactions[0]
    assert {top_tree_pair["col1_name"], top_tree_pair["col2_name"]} == {"feat0", "feat1"}
