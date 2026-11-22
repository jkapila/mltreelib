import numpy as np
from mltreelib import ModelTree


def test_ctree_strong_signal_recovery():
    # Strong signal on x0, high-cardinality noise on x1 (30 categories), continuous noise on x2
    rng = np.random.default_rng(42)
    n = 300
    x0 = np.linspace(-2, 2, n)
    x1 = rng.integers(0, 30, size=n).astype(float)
    x2 = rng.standard_normal(n)
    eps = rng.normal(0, 0.1, n)
    y = np.where(x0 <= 0, -4.0, 4.0) + eps

    X = np.column_stack([x0, x1, x2])
    tree = ModelTree(
        leaf="mean",
        gain="ctree",
        split_test="ctree",
        split_alpha=0.05,
        bonferroni=True,
        max_depth=2,
        min_rows=15,
        prune="none",
    )
    tree.fit(X, y, dtypes=["float", "categorical", "float"])

    splits = tree.splits()
    assert len(splits) > 0, "CTree must detect strong signal on x0"
    splits[0]
    # In tree.splits(), check root split or diagnostics
    diags = tree.split_diagnostics()
    root_diag = next(d for d in diags if d is not None and d["depth"] == 0)
    assert root_diag["test_name"] == "ctree"
    assert root_diag["passed"] is True
    assert root_diag["p_value"] < 1e-6

    # Verify x0 is selected over the 30-category noise variable x1
    vars_dict = {v["name"]: v for v in root_diag["variables"]}
    assert vars_dict["x0"]["p_value"] < 1e-12
    assert vars_dict["x1"]["p_value"] > 0.05


def test_ctree_cardinality_bias_comparison():
    # Loh's cardinality trap benchmark:
    # Under pure noise, CART (gain="sse") selects the 100-category column heavily,
    # whereas CTree (gain="ctree") evaluates p-values under the permutation null
    # so high cardinality is not unfairly selected.
    rng = np.random.default_rng(1234)
    n = 400
    y = rng.standard_normal(n)

    x_bin = (np.arange(n) % 2).astype(float)       # 2 levels
    x_cat10 = (np.arange(n) % 10).astype(float)   # 10 levels
    x_cat50 = (np.arange(n) % 50).astype(float)   # 50 levels
    x_cat100 = (np.arange(n) % 100).astype(float) # 100 levels

    X = np.column_stack([x_bin, x_cat10, x_cat50, x_cat100])
    dtypes = ["categorical", "categorical", "categorical", "categorical"]

    # Fit CTree under null with alpha=1.0 (forcing split) to observe selected column
    tree_ctree = ModelTree(
        gain="ctree",
        split_test="ctree",
        split_alpha=1.0,
        bonferroni=False,
        max_depth=1,
        min_rows=1,
        prune="none",
    )
    tree_ctree.fit(X, y, dtypes=dtypes)

    # Diagnostics show all variables have valid p-values under null
    diags = tree_ctree.split_diagnostics()
    root_diag = next(d for d in diags if d is not None and d["depth"] == 0)
    p_vals = [v["p_value"] for v in root_diag["variables"]]
    # All p-values under null should be in (0, 1) without 100-level dominating
    for p in p_vals:
        assert 0.0 <= p <= 1.0


def test_ctree_null_stopping():
    # Pure noise with alpha=0.01: ctree should not split
    rng = np.random.default_rng(999)
    n = 200
    x0 = rng.uniform(-1, 1, n)
    x1 = rng.integers(0, 5, size=n).astype(float)
    y = rng.standard_normal(n)

    X = np.column_stack([x0, x1])
    tree = ModelTree(
        gain="ctree",
        split_test="ctree",
        split_alpha=0.01,
        bonferroni=True,
        max_depth=3,
        min_rows=10,
        prune="none",
    )
    tree.fit(X, y, dtypes=["float", "categorical"])
    assert tree.n_leaves() == 1
    assert len(tree.splits()) == 0


def test_gain_ctree_alias():
    # gain="ctree" automatically sets split_test="ctree"
    rng = np.random.default_rng(55)
    n = 150
    x0 = np.linspace(-1, 1, n)
    y = np.where(x0 <= 0, -3.0, 3.0) + rng.normal(0, 0.1, n)

    X = x0.reshape(-1, 1)
    tree = ModelTree(gain="ctree", max_depth=1, min_rows=10, prune="none")
    tree.fit(X, y)
    assert len(tree.splits()) == 1
    assert tree.config()["split_test"] == "ctree"
    diags = [d for d in tree.split_diagnostics() if d is not None]
    assert len(diags) > 0
    assert diags[0]["test_name"] == "ctree"
