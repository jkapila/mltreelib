"""Tests for Wasserstein-2 Optimal Transport and Fairness-Constrained Splitting (Spec 12 Phase 12D)."""

import math
import random
import pytest

from mltreelib import ModelTree, distributions_to_quantiles, histograms_to_quantiles


def test_histograms_to_quantiles_conversion():
    # Two simple 4-bin probability histograms
    # Bin 0: [0, 1), Bin 1: [1, 2), Bin 2: [2, 3), Bin 3: [3, 4)
    # Hist 1: mass concentrated in bin 0
    # Hist 2: mass concentrated in bin 3
    hists = [
        [0.85, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.85],
    ]
    edges = [0.0, 1.0, 2.0, 3.0, 4.0]
    quantiles = histograms_to_quantiles(hists, bin_edges=edges, n_quantiles=20)
    assert len(quantiles) == 2
    assert len(quantiles[0]) == 20
    assert len(quantiles[1]) == 20

    # Hist 1 quantiles should be mostly in [0, 1]
    assert quantiles[0][5] < 1.0
    # Hist 2 quantiles should be mostly in [3, 4]
    assert quantiles[1][15] > 3.0

    # Verify quantiles are non-decreasing
    for q_row in quantiles:
        for i in range(len(q_row) - 1):
            assert q_row[i] <= q_row[i + 1] + 1e-9


def test_distributions_to_quantiles_conversion():
    rng = random.Random(42)
    # Generate 50 samples for each of 2 distributions: N(-3, 1) and N(+3, 1)
    samples = [
        [rng.gauss(-3.0, 0.5) for _ in range(50)],
        [rng.gauss(3.0, 0.5) for _ in range(50)],
    ]
    quantiles = distributions_to_quantiles(samples, n_quantiles=25)
    assert len(quantiles) == 2
    assert len(quantiles[0]) == 25
    assert len(quantiles[1]) == 25

    # Check median quantile (tau ~ 0.5 at index 12)
    assert -4.0 < quantiles[0][12] < -2.0
    assert 2.0 < quantiles[1][12] < 4.0


def test_wasserstein_quantiles_split_and_predict_vector():
    rng = random.Random(123)
    n = 150
    n_quantiles = 20

    X = []
    Y_quantiles = []

    for _ in range(n):
        x0 = rng.uniform(-1.0, 1.0)
        x1 = rng.uniform(-1.0, 1.0)  # noise feature
        X.append([x0, x1])

        # Distribution shift controlled by x0:
        # If x0 < 0: distribution centered at -2.0
        # If x0 >= 0: distribution centered at +2.0
        center = -2.0 if x0 < 0.0 else 2.0
        samples = [rng.gauss(center, 0.4) for _ in range(40)]
        q_row = distributions_to_quantiles([samples], n_quantiles=n_quantiles)[0]
        Y_quantiles.append(q_row)

    tree = ModelTree(gain="wasserstein", max_depth=1, min_rows=15)
    tree.fit(X, Y_quantiles, col_names=["x0", "x1"])

    splits = tree.splits()
    assert len(splits) == 1
    root = splits[0]
    assert root["col"] in ("x0", 0), "Wasserstein tree must split on signal column x0"
    assert abs(root["threshold"] - 0.0) < 0.3

    # Check vector prediction
    test_X = [[-0.8, 0.0], [0.8, 0.0]]
    preds = tree.predict_vector(test_X)
    assert len(preds) == 2
    assert len(preds[0]) == n_quantiles
    assert len(preds[1]) == n_quantiles

    # Leaf 0 should have negative quantiles, Leaf 1 positive
    mean_q0 = sum(preds[0]) / n_quantiles
    mean_q1 = sum(preds[1]) / n_quantiles
    assert mean_q0 < -1.0
    assert mean_q1 > 1.0

    # Test kind="vector" in predict
    preds_kind = tree.predict(test_X, kind="response")
    assert preds_kind == preds


def test_fairness_constrained_credit_demographic_parity():
    rng = random.Random(999)
    n = 300

    X = []
    y = []

    for _ in range(n):
        gender = 1.0 if rng.random() < 0.5 else 0.0
        # Proxy feature: heavily correlated with gender (unfair proxy)
        gender_proxy = gender + rng.gauss(0.0, 0.05)
        # Merit feature: independent of gender, determines real creditworthiness
        credit_score = rng.uniform(0.0, 100.0)

        # Real outcome depends on credit_score + slight noise
        approved = 1.0 if credit_score > 50.0 else 0.0

        # Inject strong correlation between proxy and outcome in training data
        # so unconstrained CART chooses proxy over credit_score
        target = approved + 3.0 * gender

        X.append([gender, gender_proxy, credit_score])
        y.append(target)

    # 1. Unconstrained tree (fairness_penalty = 0.0)
    tree_unconstrained = ModelTree(
        max_depth=1,
        min_rows=25,
        fairness_penalty=0.0,
        protected_col=0,
    )
    tree_unconstrained.fit(X, y, col_names=["gender", "gender_proxy", "credit_score"])
    splits_uncon = tree_unconstrained.splits()
    assert len(splits_uncon) == 1
    uncon_col = splits_uncon[0]["col"]

    # 2. Fair tree (gain="fair", fairness_penalty = 1.0, protected_attribute="gender")
    tree_fair = ModelTree(
        gain="fair",
        max_depth=1,
        min_rows=25,
        fairness_penalty=1.0,
        protected_col=0,
    )
    tree_fair.fit(X, y, col_names=["gender", "gender_proxy", "credit_score"])
    splits_fair = tree_fair.splits()
    assert len(splits_fair) == 1
    fair_col = splits_fair[0]["col"]

    # The fair tree must reject the gender proxy and choose the fair credit_score column
    assert fair_col in ("credit_score", 2), f"Fair tree must split on fair credit_score, got {fair_col}"
    assert uncon_col in ("gender_proxy", "gender", 0, 1), f"Unconstrained tree chose {uncon_col}"

    # Verify demographic parity gap on predictions
    preds_uncon = tree_unconstrained.predict(X)
    preds_fair = tree_fair.predict(X)

    g1_idx = [i for i, row in enumerate(X) if row[0] == 1.0]
    g0_idx = [i for i, row in enumerate(X) if row[0] == 0.0]

    uncon_gap = abs(
        (sum(preds_uncon[i] for i in g1_idx) / len(g1_idx))
        - (sum(preds_uncon[i] for i in g0_idx) / len(g0_idx))
    )
    fair_gap = abs(
        (sum(preds_fair[i] for i in g1_idx) / len(g1_idx))
        - (sum(preds_fair[i] for i in g0_idx) / len(g0_idx))
    )

    assert fair_gap <= uncon_gap + 1e-9, f"Fair gap {fair_gap} should be <= unconstrained gap {uncon_gap}"
