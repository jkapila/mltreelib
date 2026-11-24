import math
import pytest
from mltreelib import ModelTree


def test_expectile_linear_leaf_fit_predict():
    x = [[i * 0.1, math.sin(i * 0.1)] for i in range(80)]
    y = [3.0 + 2.0 * row[0] - 1.5 * row[1] for row in x]

    tree = ModelTree(
        max_depth=3,
        min_rows=10,
        leaf="linear",
        leaf_family="expectile",
        expectile_tau=0.75,
    )
    tree.fit(x, y)
    preds = tree.predict(x)

    assert len(preds) == 80
    assert all(math.isfinite(p) for p in preds)
    assert tree.expectile_tau == 0.75
    assert tree.leaf_family == "expectile"


def test_expectile_constant_leaf():
    x = [[float(i)] for i in range(50)]
    y = [float(i * 2) for i in range(50)]

    tree = ModelTree(
        max_depth=2,
        min_rows=10,
        leaf="expectile",
        expectile_tau=0.5,
    )
    tree.fit(x, y)
    preds = tree.predict(x)

    assert len(preds) == 50
    assert all(math.isfinite(p) for p in preds)


def test_expectile_level_monotonicity():
    x = [[float(i % 10)] for i in range(100)]
    # Asymmetric noise
    y = [5.0 + 2.0 * row[0] + (10.0 if i % 4 == 0 else -2.0) for i, row in enumerate(x)]

    tree_low = ModelTree(
        max_depth=1,
        min_rows=20,
        leaf="linear",
        leaf_family="expectile",
        expectile_tau=0.1,
    )
    tree_high = ModelTree(
        max_depth=1,
        min_rows=20,
        leaf="linear",
        leaf_family="expectile",
        expectile_tau=0.9,
    )

    tree_low.fit(x, y)
    tree_high.fit(x, y)

    preds_low = tree_low.predict(x)
    preds_high = tree_high.predict(x)

    mean_low = sum(preds_low) / len(preds_low)
    mean_high = sum(preds_high) / len(preds_high)
    assert mean_low < mean_high, f"Expected mean_low ({mean_low}) < mean_high ({mean_high})"


def test_nnls_positive_coefficients():
    # Negative relationship: y = 20 - 3*x1 + 4*x2
    x = [[float(i), float(80 - i)] for i in range(80)]
    y = [20.0 - 3.0 * row[0] + 4.0 * row[1] for row in x]

    tree = ModelTree(
        max_depth=2,
        min_rows=15,
        leaf="linear",
        non_negative=True,
        non_negative_intercept=False,
    )
    tree.fit(x, y)
    preds = tree.predict(x)
    assert len(preds) == 80

    for leaf in tree.leaves():
        if leaf.get("coef") is not None:
            # Slope coefficients should be >= 0
            for idx, c in enumerate(leaf["coef"]):
                assert c >= -1e-6, f"Slope coefficient {idx} ({c}) must be non-negative"


def test_nnls_all_non_negative_including_intercept():
    x = [[float(i)] for i in range(50)]
    y = [float(5 + 2 * i) for i in range(50)]

    tree = ModelTree(
        max_depth=2,
        min_rows=10,
        leaf="linear",
        non_negative=True,
        non_negative_intercept=True,
    )
    tree.fit(x, y)

    for leaf in tree.leaves():
        if leaf.get("coef") is not None:
            for idx, c in enumerate(leaf["coef"]):
                assert c >= -1e-6, f"Coefficient {idx} ({c}) must be non-negative"
        if leaf.get("intercept") is not None:
            assert leaf["intercept"] >= -1e-6, f"Intercept ({leaf['intercept']}) must be non-negative"


def test_negative_binomial_count_regression():
    x = [[i * 0.05] for i in range(60)]
    y = [round(math.exp(1.0 + 0.4 * row[0]) + (i % 3)) for i, row in enumerate(x)]

    tree = ModelTree(
        max_depth=2,
        min_rows=10,
        leaf="linear",
        leaf_family="negative_binomial",
    )
    tree.fit(x, y)
    preds = tree.predict(x)

    assert len(preds) == 60
    assert all(p >= 0.0 for p in preds)
    assert tree.leaf_family == "negative_binomial"


def test_beta_bounded_regression():
    x = [[i * 0.1 - 3.0] for i in range(60)]
    # Logit link
    y = [1.0 / (1.0 + math.exp(-0.5 * row[0])) for row in x]
    # Bound strictly inside (0, 1)
    y = [max(0.01, min(0.99, val)) for val in y]

    tree = ModelTree(
        max_depth=2,
        min_rows=10,
        leaf="linear",
        leaf_family="beta",
    )
    tree.fit(x, y)
    preds = tree.predict(x)

    assert len(preds) == 60
    assert all(0.0 < p < 1.0 for p in preds)
    assert tree.leaf_family == "beta"
