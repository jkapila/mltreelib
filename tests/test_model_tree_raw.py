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
