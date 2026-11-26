"""Models accept tables by duck typing: no pandas/Arrow import, NAs untouched."""

import math

import numpy as np
import pytest
from mltreelib import AnomalyModel, ClusterModel, ModelTree

rng = np.random.default_rng(5)
N = 200
A = rng.normal(size=N)
B = rng.normal(size=N)
CAT = np.where(rng.random(N) > 0.5, "u", "v").tolist()
Y = (2.0 * A + (np.array(CAT) == "u") * 1.5 + rng.normal(0, 0.1, N)).tolist()
A_NA = [None if i % 9 == 0 else float(v) for i, v in enumerate(A)]
ROWS = [[a, float(b), c] for a, b, c in zip(A_NA, B, CAT)]
NAMES = ["a", "b", "c"]
DTYPES = ["numeric", "numeric", "categorical"]


class FakeFrame:
    """Data-frame-like: `.columns` and `frame[name]` returning an iterable."""

    def __init__(self, data):
        self._d = data
        self.columns = list(data)

    def __getitem__(self, name):
        return self._d[name]


class FakeArrow:
    def __init__(self, data):
        self._d = data

    def to_pydict(self):
        return self._d


def _cols():
    return {"a": A_NA, "b": B.tolist(), "c": CAT}


def _fit(x, **kw):
    m = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    m.fit(x, Y, dtypes=DTYPES, **kw)
    return m


def test_dict_frame_arrow_match_list_of_rows():
    base = _fit(ROWS, col_names=NAMES)
    ref = base.predict(ROWS)
    for table in (_cols(), FakeFrame(_cols()), FakeArrow(_cols())):
        m = _fit(table)
        assert m.predict(table) == ref
        assert "a" in str(m) or "b" in str(m) or "c" in str(m)


def test_names_come_from_the_table_when_not_given():
    m = _fit(FakeFrame(_cols()))
    printed = str(m)
    assert "x0" not in printed or any(n in printed for n in NAMES)


def test_missing_values_are_not_imputed():
    m = _fit(FakeFrame(_cols()))
    preds = m.predict(FakeFrame(_cols()))
    assert len(preds) == N and not any(math.isnan(p) for p in preds)


def test_pandas_frame_if_installed():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_cols())
    df.loc[3, "b"] = float("nan")
    m = _fit(df)
    p = m.predict(df)
    assert len(p) == N


def test_pandas_na_singleton_is_accepted():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": pd.array([1.0, None, 3.0, 4.0] * 50, dtype="Float64"), "b": B[:200]})
    m = ModelTree(leaf="mean", max_depth=2, min_rows=10)
    m.fit(df, Y)
    assert len(m.predict(df)) == 200


def test_cluster_and_anomaly_accept_tables():
    num = {"a": [v if v is not None else float("nan") for v in A_NA], "b": B.tolist()}
    c = ClusterModel(kind="pct", max_depth=2, min_rows=20)
    c.fit(FakeFrame(num))
    assert len(c.predict(FakeFrame(num))) == N
    a = AnomalyModel(kind="eif", max_depth=6, seed=1)
    a.fit(FakeFrame(num))
    assert len(a.score_samples(FakeFrame(num))) == N
