"""Tests verifying C5.0 rule extraction, statsmodels leaf summary, and Cubist k-NN neighbor adjustment."""

from __future__ import annotations

import random

from mltreelib import ModelTree, Rule, RuleSet


def test_rule_extraction_c50():
    random.seed(42)
    n = 200

    X: list[list[float]] = []
    y: list[float] = []

    for i in range(n):
        x0 = float(i) / float(n)
        x1 = float((i * 7) % n) / float(n)
        val = 10.0 if x0 > 0.5 and x1 > 0.5 else 1.0
        X.append([x0, x1])
        y.append(val)

    tree = ModelTree(
        leaf="mean",
        max_depth=4,
        min_rows=5,
    )
    tree.fit(X, y, col_names=["feature_a", "feature_b"])

    ruleset = tree.extract_rules(X, y, prune=True, cf=0.25)
    assert isinstance(ruleset, RuleSet)
    assert len(ruleset) > 0
    assert len(ruleset) <= tree.n_leaves()

    rules = ruleset.rules
    assert all(isinstance(r, Rule) for r in rules)
    first_rule = rules[0]
    assert first_rule.cover > 0
    assert 0.0 <= first_rule.confidence <= 1.0
    assert len(first_rule.conditions) > 0

    # Test prediction
    preds = ruleset.predict(X)
    assert len(preds) == n
    mae = sum(abs(p - t) for p, t in zip(preds, y)) / n
    assert mae < 1.0

    # Test string formatting
    rule_text = str(ruleset)
    assert "Rule" in rule_text
    assert "Default Rule:" in rule_text


def test_leaf_summary_r_statsmodels_format():
    random.seed(42)
    n = 100

    X: list[list[float]] = []
    y: list[float] = []

    for i in range(n):
        x0 = float(i) / float(n)
        x1 = float((i * 3) % n) / float(n)
        noise = (float(i % 5) - 2.0) * 0.05
        target = 2.0 + 3.0 * x0 - 1.5 * x1 + noise
        X.append([x0, x1])
        y.append(target)

    tree = ModelTree(
        leaf="linear",
        leaf_cols="all",
        max_depth=1,
        min_rows=10,
    )
    tree.fit(X, y, col_names=["f1", "f2"])

    summary_str = tree.summary()
    assert "Residual standard error:" in summary_str
    assert "Multiple R-squared:" in summary_str
    assert "F-statistic:" in summary_str
    assert "Estimate" in summary_str
    assert "Std. Error" in summary_str
    assert "t value" in summary_str
    assert "Pr(>|t|)" in summary_str

    leaf0 = tree.leaf_summary(0)
    assert "Estimate" in leaf0
    assert "(Intercept)" in leaf0


def test_cubist_knn_neighbor_adjustment():
    random.seed(42)
    n = 120

    X: list[list[float]] = []
    y: list[float] = []

    for i in range(n):
        v = (float(i) / float(n)) * 10.0
        # Target with non-linear shape
        target = 5.0 * (v * 0.8) + 2.0
        X.append([v])
        y.append(target)

    tree = ModelTree(
        leaf="mean",
        max_depth=2,
        min_rows=15,
    )
    tree.fit(X, y, col_names=["x"])

    # Base predictions without adjustment
    unadj = tree.predict(X)

    # Predictions with Cubist k-NN adjustment
    adj = tree.predict(X, neighbors=5)
    adj_alt = tree.predict_adjusted(X, neighbors=5)

    assert len(adj) == n
    assert adj == adj_alt

    unadj_sse = sum((p - t) ** 2 for p, t in zip(unadj, y))
    adj_sse = sum((p - t) ** 2 for p, t in zip(adj, y))

    # Neighbor adjustment reduces error by borrowing strength from training neighbors
    assert adj_sse < unadj_sse
