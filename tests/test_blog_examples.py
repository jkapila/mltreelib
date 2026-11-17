"""Tests verifying end-to-end functionality of blog examples.

Ensures that code examples published in docs/blog/ remain accurate,
fully runnable, and mathematically sound across releases.
"""

from __future__ import annotations

import math
import random

from mltreelib import Metrics, ModelTree


def test_blog_causal_uplift():
    """Verify Honest Causal Tree example from uplift-modeling-with-causal-trees.md."""
    random.seed(42)
    n = 600

    X: list[list[float]] = []
    y: list[float] = []
    w: list[float] = []

    for _ in range(n):
        x0 = random.uniform(-1.0, 1.0)
        x1 = random.uniform(-1.0, 1.0)
        treat = 1.0 if random.random() < 0.5 else 0.0

        if x0 > 0.0:
            # Persuadables: positive treatment uplift
            prob = 0.80 if treat == 1.0 else 0.20
        elif x1 > 0.0:
            # Sleeping dogs: negative treatment uplift
            prob = 0.15 if treat == 1.0 else 0.60
        else:
            prob = 0.30

        outcome = 1.0 if random.random() < prob else 0.0
        X.append([x0, x1])
        y.append(outcome)
        w.append(treat)

    tree = ModelTree(
        gain="causal",
        max_depth=3,
        min_rows=20,
        prune="none",
        honesty=True,
        honesty_fraction=0.5,
        seed=42,
    )
    tree.fit(X[:450], (y[:450], w[:450]))

    # Introspect Graphviz tree representation
    dot = tree.to_dot()
    assert "digraph tree" in dot
    assert len(dot) > 100

    # Test uplift predictions
    tau_pred = tree.predict(X[450:], kind="uplift")
    assert len(tau_pred) == 150
    assert all(isinstance(v, float) for v in tau_pred)

    # Persuadable sample (x0=0.8, x1=0.0) vs Sleeping Dog sample (x0=-0.8, x1=0.8)
    tau_persuadable = tree.predict([[0.8, 0.0]], kind="uplift")[0]
    tau_sleeping_dog = tree.predict([[-0.8, 0.8]], kind="uplift")[0]

    assert tau_persuadable > 0.20, (
        f"Expected positive uplift for persuadables, got {tau_persuadable}"
    )
    assert tau_sleeping_dog < -0.10, (
        f"Expected negative uplift for sleeping dogs, got {tau_sleeping_dog}"
    )


def test_blog_credit_scoring():
    """Verify Monotone Glass-Box Tree example from why-credit-scoring-needs-glass-box-trees.md."""
    random.seed(123)
    n = 600

    X: list[list[float]] = []
    y: list[float] = []

    for _ in range(n):
        cibil = random.uniform(300.0, 850.0)
        income = random.uniform(2000.0, 15000.0)
        delinq = float(random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.1, 0.1])[0])
        dti = random.uniform(0.1, 0.7)

        logit = 2.0 - 0.01 * (cibil - 600.0) + 0.8 * delinq + 1.8 * (dti - 0.3)
        prob = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, logit))))
        default_flag = 1.0 if random.random() < prob else 0.0

        X.append([cibil, income, delinq, dti])
        y.append(default_flag)

    tree = ModelTree(
        gain="gini",
        max_depth=3,
        min_rows=25,
        monotone=[-1, 0, 1, 1],
        prune="none",
        seed=123,
    )
    tree.fit(X[:400], y[:400])

    dot = tree.to_dot()
    assert "digraph tree" in dot

    scores = tree.predict(X[400:])
    assert len(scores) == 200

    # Ensure higher CIBIL score applicant receives lower or equal default score
    score_low_cibil = tree.predict([[500.0, 5000.0, 1.0, 0.4]])[0]
    score_high_cibil = tree.predict([[780.0, 5000.0, 1.0, 0.4]])[0]
    assert score_high_cibil <= score_low_cibil

    m = Metrics()
    roc = m.roc_curve(y[400:], scores)
    assert roc["auc"] > 0.65


def test_blog_predictive_maintenance():
    """Verify Piecewise Linear Model Tree with Missing Routing from zero-imputation-predictive-maintenance.md."""
    random.seed(999)
    n = 600

    X: list[list[float | None]] = []
    y: list[float] = []

    for _ in range(n):
        temp = random.uniform(40.0, 110.0)
        vib = random.uniform(0.5, 6.0)

        # Dropout mechanism on high temp
        if temp > 80.0 and random.random() < 0.5:
            pressure = None
        else:
            pressure = random.uniform(20.0, 50.0)

        rul = 300.0 - 2.5 * temp - 20.0 * vib
        if pressure is None:
            rul -= 35.0
        else:
            rul += 0.5 * pressure

        rul = max(10.0, rul + random.gauss(0, 10.0))
        X.append([temp, vib, pressure])
        y.append(rul)

    tree = ModelTree(
        leaf="linear",
        max_depth=3,
        min_rows=25,
        maxsurrogate=1,
        prune="none",
        seed=999,
    )
    tree.fit(X[:400], y[:400])

    dot = tree.to_dot()
    assert "digraph tree" in dot

    # Test prediction on unseen rows with None missing values
    preds = tree.predict(X[400:])
    assert len(preds) == 200
    assert all(isinstance(p, float) for p in preds)

    reg = Metrics().regression(y[400:], preds)
    assert reg["r2"] > 0.60


def test_blog_quinlan_classics():
    """Verify modernizing-quinlan-classics-in-rust.md blog examples."""
    random.seed(42)
    n = 150
    X: list[list[float]] = []
    y: list[float] = []

    for _ in range(n):
        sqft = random.uniform(800.0, 3000.0)
        beds = float(random.randint(1, 4))
        age = random.uniform(1.0, 40.0)
        tax = random.uniform(1.0, 3.0)

        # Non-linear localized surface
        val = 80.0 + 0.12 * sqft + 15.0 * beds - 0.8 * age - 3.0 * tax
        val += 25.0 * math.sin(sqft / 300.0)
        val += random.gauss(0.0, 3.0)

        X.append([sqft, beds, age, tax])
        y.append(val)

    tree = ModelTree(
        leaf="linear",
        leaf_cols="all",
        max_depth=2,
        min_rows=20,
        seed=42,
    )
    col_names = ["sqft", "beds", "age", "tax_rate"]
    tree.fit(X[:100], y[:100], col_names=col_names)

    # 1. Test Leaf Summary
    summary = tree.leaf_summary(0)
    assert "Residual standard error:" in summary
    assert "Multiple R-squared:" in summary
    assert "Coefficients:" in summary
    assert "Estimate" in summary

    # 2. Test Rule Extraction
    ruleset = tree.extract_rules(X[:100], y[:100], prune=True, cf=0.25)
    assert len(ruleset) > 0
    rule_preds = ruleset.predict(X[100:])
    assert len(rule_preds) == 50

    # 3. Test Cubist k-NN Adjustment
    base_preds = tree.predict(X[100:])
    adj_preds = tree.predict(X[100:], neighbors=5)
    assert len(base_preds) == 50
    assert len(adj_preds) == 50

    base_mse = sum((p - t) ** 2 for p, t in zip(base_preds, y[100:])) / 50
    adj_mse = sum((p - t) ** 2 for p, t in zip(adj_preds, y[100:])) / 50
    assert adj_mse < base_mse
