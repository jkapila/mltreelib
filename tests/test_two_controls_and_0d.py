"""Tests for two-control architecture, rich 0D stats, dual monotonicity, weights, and interaction constraints."""

from mltreelib import LeafControl, ModelTree


def test_leaf_control_and_model_tree_integration():
    ctrl = LeafControl(leaf="prob", leaf_monotone=[1, -1])
    assert ctrl.leaf == "prob"
    assert ctrl.leaf_monotone == [1, -1]

    # Initialize ModelTree with control
    m = ModelTree(control=ctrl, max_depth=2, min_rows=5)
    params = m.get_params()
    assert params["leaf"] == "prob"
    assert params["leaf_monotone"] == [1, -1]


def test_leaf_0d_stats_dictionary():
    # 60 rows binary classification with distinct subgroups
    x = [[float(i), float(i % 2)] for i in range(60)]
    y = [float(i % 2) for i in range(60)]
    m = ModelTree(leaf="prob", max_depth=2, min_rows=5)
    m.fit(x, y)

    leaves = m.leaves()
    assert len(leaves) >= 2
    for leaf in leaves:
        assert "stats" in leaf
        st = leaf["stats"]
        expected_keys = {
            "mean", "std", "var", "median", "mad", "iqr", "min", "max",
            "skew", "kurtosis", "prob", "odds", "log_odds", "majority_class",
            "entropy", "gini", "zero_fraction", "dispersion",
        }
        assert expected_keys.issubset(set(st.keys()))
        # Check direct aliases match stats
        assert leaf["prob"] == st["prob"]
        assert leaf["median"] == st["median"]
        assert leaf["odds"] == st["odds"]
        assert leaf["log_odds"] == st["log_odds"]

    # Also test m.leaf_stats()
    lstats = m.leaf_stats()
    assert len(lstats) == len(leaves)
    assert lstats[0]["prob"] == leaves[0]["stats"]["prob"]


def test_predict_kinds():
    x = [[float(i), float(i % 2)] for i in range(60)]
    y = [float(i % 2) for i in range(60)]
    m = ModelTree(leaf="prob", max_depth=2, min_rows=5)
    m.fit(x, y)

    test_x = x[:4]
    p_prob = m.predict(test_x, kind="prob")
    p_odds = m.predict(test_x, kind="odds")
    p_lo = m.predict(test_x, kind="log_odds")
    p_med = m.predict(test_x, kind="median")
    p_stats = m.predict(test_x, kind="stats")

    assert len(p_prob) == 4
    assert len(p_odds) == 4
    assert len(p_lo) == 4
    assert len(p_med) == 4
    assert len(p_stats) == 4
    assert isinstance(p_stats[0], dict)
    assert "prob" in p_stats[0]
    assert "entropy" in p_stats[0]


def test_dual_monotone_constraints():
    # tree_monotone governs splits; leaf_monotone governs linear leaf coefs
    m = ModelTree(
        tree_monotone=[1, 0],
        leaf_monotone=[0, -1],
        leaf="linear",
        max_depth=2,
        min_rows=5,
    )
    p = m.get_params()
    assert p["tree_monotone"] == [1, 0]
    assert p["leaf_monotone"] == [0, -1]


def test_weights_row_and_column():
    x = [[float(i), float(i * 2)] for i in range(40)]
    y = [float(i) for i in range(40)]
    row_w = [1.0] * 40
    col_w = [2.0, 0.5]

    # Col weights passed in init
    m = ModelTree(col_weights=col_w, max_depth=2, min_rows=5)
    m.fit(x, y, sample_weight=row_w)
    assert m.n_leaves() >= 1
    assert m.get_params()["col_weights"] == col_w

    # Col weights passed in fit
    m2 = ModelTree(max_depth=2, min_rows=5)
    m2.fit(x, y, row_weights=row_w, col_weights=[1.0, 3.0])
    assert m2.get_params()["col_weights"] == [1.0, 3.0]


def test_interaction_groups_with_column_names():
    x = [[float(i), float(i % 3), float(i % 5)] for i in range(60)]
    y = [float(i % 3) for i in range(60)]
    m = ModelTree(
        max_depth=3,
        min_rows=5,
        interaction_groups=[["featA", "featB"], ["featC"]],
    )
    m.fit(x, y, col_names=["featA", "featB", "featC"])
    assert m.n_leaves() >= 1
    # Check resolved indices in params
    assert m.get_params()["interaction_groups"] == [[0, 1], [2]]
