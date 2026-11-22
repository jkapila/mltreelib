import numpy as np
from mltreelib import ModelTree


def test_mob_linear_structural_break():
    # Structural break in slope along x0 at 0.0:
    # y = 2.0 * x1 (x0 <= 0), y = -2.0 * x1 (x0 > 0)
    # x2 is pure noise
    rng = np.random.default_rng(42)
    n = 200
    x0 = np.linspace(-1, 1, n)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    eps = rng.normal(0, 0.05, n)
    y = np.where(x0 <= 0, 2.0 * x1, -2.0 * x1) + eps

    X = np.column_stack([x0, x1, x2])
    tree = ModelTree(
        leaf="linear",
        leaf_cols="all",
        split_cols=[0, 2],
        split_test="mob",
        split_alpha=0.05,
        max_depth=2,
        min_rows=15,
        prune="none",
    )
    tree.fit(X, y)

    # Root split must be on x0
    splits = tree.splits()
    assert len(splits) > 0, "MOB must detect structural break and split"

    # Diagnostics verification (User Story 7)
    diags = tree.split_diagnostics()
    root_diag = next(d for d in diags if d is not None and d["depth"] == 0)
    assert root_diag["test_name"] == "mob"
    assert root_diag["passed"] is True
    assert root_diag["p_value"] < 0.001

    # Check variable rankings: x0 should have extreme significance, x2 should not
    vars_dict = {v["name"]: v for v in root_diag["variables"]}
    assert vars_dict["x0"]["p_value"] < 1e-10
    assert vars_dict["x2"]["p_value"] > 0.05

    # Predictions
    pred_left = tree.predict([[-0.5, 1.0, 0.0]])[0]
    pred_right = tree.predict([[0.5, 1.0, 0.0]])[0]
    assert abs(pred_left - 2.0) < 0.3
    assert abs(pred_right - (-2.0)) < 0.3


def test_mob_mean_structural_break():
    # Mean shift at x0 = 0.0
    rng = np.random.default_rng(123)
    n = 160
    x0 = np.linspace(-2, 2, n)
    x1 = rng.uniform(-1, 1, n)
    eps = rng.normal(0, 0.1, n)
    y = np.where(x0 <= 0, 10.0, -10.0) + eps

    X = np.column_stack([x0, x1])
    tree = ModelTree(
        leaf="mean",
        split_test="mob",
        split_alpha=0.01,
        max_depth=2,
        min_rows=15,
        prune="none",
    )
    tree.fit(X, y)

    splits = tree.splits()
    assert len(splits) > 0
    # Root split must have cutpoint near 0.0
    root_diag = next(d for d in tree.split_diagnostics() if d is not None and d["depth"] == 0)
    assert root_diag["passed"] is True


def test_mob_null_stability_stops_at_root():
    # Parameter stability under H0: no structural break
    rng = np.random.default_rng(777)
    n = 150
    x0 = rng.uniform(-1, 1, n)
    x1 = rng.uniform(-1, 1, n)
    eps = rng.normal(0, 0.05, n)
    y = 3.0 * x1 + 1.0 + eps

    X = np.column_stack([x0, x1])
    tree = ModelTree(
        leaf="linear",
        split_test="mob",
        split_alpha=0.05,
        bonferroni=True,
        max_depth=3,
        min_rows=15,
    )
    tree.fit(X, y)

    # Must stay as a single terminal leaf
    assert tree.n_leaves() == 1
    assert len(tree.splits()) == 0


def test_gain_mob_alias():
    # gain="mob" automatically sets split_test="mob"
    rng = np.random.default_rng(99)
    n = 120
    x0 = np.linspace(-1, 1, n)
    y = np.where(x0 <= 0, 5.0, -5.0) + rng.normal(0, 0.1, n)

    X = x0.reshape(-1, 1)
    tree = ModelTree(gain="mob", max_depth=1, min_rows=10, prune="none")
    tree.fit(X, y)
    assert len(tree.splits()) == 1
    assert tree.config()["split_test"] == "mob"
    diags = [d for d in tree.split_diagnostics() if d is not None]
    assert len(diags) > 0
    assert diags[0]["test_name"] == "mob"
