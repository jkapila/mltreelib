"""Raw-cell ingest: the model runs on data as-is (int / float / level)."""

import math

from mltreelib import ModelTree


def test_strings_enter_as_levels():
    # 300 rows; a 3-level categorical with planted per-level means
    levels = ["alpha", "beta", "gamma"]
    x = [[levels[i % 3]] for i in range(300)]
    y = [float(i % 3) * 10.0 for i in range(300)]
    m = ModelTree(leaf="mean", max_depth=2, min_rows=10)
    m.fit(x, y, dtypes=["categorical"])
    pred = m.predict([["alpha"], ["beta"], ["gamma"]])
    assert pred[0] < 2.0, pred
    assert abs(pred[1] - 10.0) < 2.0, pred
    assert pred[2] > 18.0, pred


def test_unseen_level_and_nan_do_not_block():
    x = [["a"], ["b"], ["a"], ["b"]]
    y = [0.0, 10.0, 0.0, 10.0]
    m = ModelTree(leaf="mean", max_depth=1, min_rows=2)
    m.fit(x, y, dtypes=["categorical"])
    p = m.predict([["zzz"], [None], [float("nan")], ["a"]])
    assert len(p) == 4 and all(isinstance(v, float) and not math.isnan(v) for v in p)


def test_integer_and_float_cells_pass_through():
    x = [[i, i / 2.0, str(i % 2)] for i in range(200)]
    y = [float(i % 2) * 3.0 + i / 200.0 for i in range(200)]
    m = ModelTree(leaf="mean", max_depth=2, min_rows=10)
    m.fit(x, y, dtypes=["integer", "float", "categorical"])
    assert m.n_leaves() >= 1
    assert len(m.predict([[5, 2.5, "1"]])) == 1


def test_split_eval_fast_and_exhaustive_modes():
    x = [[float(i), float(i % 5)] for i in range(100)]
    y = [2.0 * float(i) + 1.0 for i in range(100)]

    # Fast mode (default, O(1) variance reduction)
    m_fast = ModelTree(leaf="linear", split_eval="fast", l2=0.0, max_depth=2, min_rows=10)
    m_fast.fit(x, y)
    assert m_fast.n_leaves() >= 1
    p_fast = m_fast.predict([[10.0, 0.0]])
    assert abs(p_fast[0] - 21.0) < 1e-4

    # Exhaustive mode (candidate linear solves)
    m_exh = ModelTree(leaf="linear", split_eval="exhaustive", l2=0.0, max_depth=2, min_rows=10)
    m_exh.fit(x, y)
    assert m_exh.n_leaves() >= 1
    p_exh = m_exh.predict([[10.0, 0.0]])
    assert abs(p_exh[0] - 21.0) < 1e-4


def test_model_tree_apply_and_leaf_samples():
    x = [[float(i)] for i in range(40)]
    y = [0.0 if i < 20 else 10.0 for i in range(40)]
    m = ModelTree(max_depth=2, min_rows=5, prune="none")
    m.fit(x, y)

    leaves = m.leaves()
    assert len(leaves) == 2
    assert leaves[0]["id"] == 0
    assert leaves[1]["id"] == 1
    assert "node_id" in leaves[0]

    applied = m.apply(x)
    assert len(applied) == 40
    assert applied[:20] == [0] * 20
    assert applied[20:] == [1] * 20

    samples = m.leaf_samples(x)
    assert set(samples.keys()) == {0, 1}
    assert samples[0] == list(range(20))
    assert samples[1] == list(range(20, 40))


def test_predict_label_and_interval():
    # 1. predict_label on binomial
    x = [[float(i)] for i in range(40)]
    y_bin = [0.0 if i < 20 else 1.0 for i in range(40)]
    m_bin = ModelTree(leaf_family="binomial", max_depth=1, min_rows=5, prune="none")
    m_bin.fit(x, y_bin)

    labels = m_bin.predict_label(x, threshold=0.5)
    assert len(labels) == 40
    assert labels[:20] == [0] * 20
    assert labels[20:] == [1] * 20

    # 2. predict_interval on regression with leaf_digest
    y_reg = [float(i) * 3.0 for i in range(40)]
    m_reg = ModelTree(max_depth=1, min_rows=5, leaf_digest=True, prune="none")
    m_reg.fit(x, y_reg)

    intervals = m_reg.predict_interval(x, coverage=0.90)
    assert len(intervals) == 40
    for lo, hi in intervals:
        assert lo <= hi


def test_retrain_and_leaf_control():
    from mltreelib import LeafControl

    x = [[float(i), float(i) * 0.2] for i in range(60)]
    y = [float(i) * 2.0 if i < 30 else 100.0 + float(i) * 0.5 for i in range(60)]

    m = ModelTree(max_depth=1, min_rows=10, leaf="linear", l2=1e-3, prune="none")
    m.fit(x, y)

    orig_splits = m.splits()
    orig_preds = m.predict(x)
    assert len(orig_splits) == 1

    # 1. Retrain to Mean leaves via LeafControl instance
    ctrl = LeafControl(leaf="mean")
    m.retrain(x, y, control=ctrl)

    # Topology preserved
    new_splits = m.splits()
    assert len(new_splits) == len(orig_splits)
    assert new_splits[0]["col"] == orig_splits[0]["col"]
    assert new_splits[0]["threshold"] == orig_splits[0]["threshold"]

    # Mean predictions are constant per leaf
    mean_preds = m.predict(x)
    assert mean_preds != orig_preds
    assert mean_preds[0] == mean_preds[15]
    assert mean_preds[35] == mean_preds[55]

    # 2. Retrain to Linear with high ridge penalty via kwargs
    m.retrain(x, y, leaf="linear", l2=100.0)
    ridge_splits = m.splits()
    assert len(ridge_splits) == len(orig_splits)
    ridge_preds = m.predict(x)
    assert ridge_preds != orig_preds
    assert ridge_preds != mean_preds
