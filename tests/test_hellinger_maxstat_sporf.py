"""Tests for Hellinger distance, MaxStat rank splitting, SPORF oblique splits,
and split gate diagnostics inspection.
"""

import random

from mltreelib import ModelTree


def test_hellinger_extreme_imbalance():
    """Hellinger distance splitting reliably isolates rare minority class without skew collapse."""
    n = 200
    X = [[float(i)] for i in range(n)]
    # Rare class (5%) concentrated at feature < 10
    y = [1.0 if i < 10 else 0.0 for i in range(n)]

    tree = ModelTree(gain="hellinger", max_depth=1, min_rows=5)
    tree.fit(X, y, col_names=["risk_score"])
    rs = tree.root_split()

    assert rs is not None, "Tree should make a split on the rare cluster"
    assert rs["col"] == "risk_score"
    assert 5.0 <= rs["threshold"] <= 15.0, f"Expected threshold near 10, got {rs['threshold']}"

    preds = tree.predict(X)
    assert len(preds) == n


def test_maxstat_rank_splitting():
    """MaxStat standardized linear rank splitting optimizes rank divergence."""
    n = 150
    X = [[-1.5 + 3.0 * i / n] for i in range(n)]
    # Non-linear monotone step at 0
    y = [-50.0 + x[0] if x[0] < 0 else 50.0 + x[0] for x in X]

    tree = ModelTree(gain="maxstat", max_depth=1, min_rows=10)
    tree.fit(X, y, col_names=["rank_feature"])
    rs = tree.root_split()

    assert rs is not None
    assert rs["col"] == "rank_feature"
    assert abs(rs["threshold"]) < 0.25, f"Expected threshold near 0, got {rs['threshold']}"


def test_sporf_oblique_split_equation_and_prediction():
    """SPORF produces sparse random linear combination hyperplanes that solve diagonal boundaries."""
    random.seed(42)
    n = 250
    X = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] for _ in range(n)]
    # Planted diagonal hyperplane x0 + x1 > 0
    y = [20.0 if (x[0] + x[1] > 0.0) else -20.0 for x in X]

    tree = ModelTree(sporf_d=2, sporf_projections=30, max_depth=1, min_rows=10, seed=42)
    tree.fit(X, y, col_names=["x0", "x1", "noise"])
    rs = tree.root_split()

    assert rs is not None
    assert "oblique" in rs, "Split must include oblique projection metadata"
    ob = rs["oblique"]
    assert len(ob["cols"]) == 2, f"SPORF with d=2 must combine 2 features, got {ob['cols']}"
    assert len(ob["weights"]) == 2
    assert "equation" in ob and "<=" in ob["equation"]

    preds = tree.predict(X)
    assert len(preds) == n
    # Oblique cut should accurately classify the diagonal
    correct = sum(1 for p, actual in zip(preds, y) if (p > 0 and actual > 0) or (p < 0 and actual < 0))
    acc = correct / n
    assert acc > 0.85, f"Expected accuracy > 85%, got {acc:.2%}"


def test_root_split_gate_diagnostics_inspection():
    """Root split must expose asymptotic p-values and test statistics for candidate variables."""
    n = 100
    X = [[-2.0 + 4.0 * i / n, random.uniform(-1, 1)] for i in range(n)]
    # Break on x0 at 0
    y = [1.0 * x[0] if x[0] < 0 else 5.0 * x[0] for x in X]

    tree = ModelTree(split_test="mob", max_depth=1, min_rows=10)
    tree.fit(X, y, col_names=["x0", "noise"])
    rs = tree.root_split()

    assert rs is not None
    assert "gate" in rs, "root_split must contain 'gate' diagnostics when split_test is active"
    gate = rs["gate"]
    assert gate["test_name"] == "mob"
    assert "p_value" in gate and "statistic" in gate
    assert gate["passed"] is True
    assert "variables" in gate and len(gate["variables"]) >= 1

    # Check candidate diagnostics
    vars_diag = gate["variables"]
    for v in vars_diag:
        assert "name" in v and "statistic" in v and "p_value" in v
        assert 0.0 <= v["p_value"] <= 1.0
