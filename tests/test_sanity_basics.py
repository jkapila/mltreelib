"""Sanity matrix: tasks, inputs, missing values, edge sizes, determinism, persistence, defaults, metrics."""
import math
import random

import pytest
from mltreelib import (
    AnomalyModel,
    ClusterModel,
    MetricsReport,
    ModelTree,
    RuleModel,
)
from mltreelib.data import Dataset

N = 60
TASKS = ["regression", "classification", "survival", "uplift", "cluster", "anomaly"]
NUMERIC = ["regression", "classification", "survival", "cluster", "anomaly"]

def _data():
    rng = random.Random(7)
    a = [rng.gauss(0, 1) for _ in range(N)]
    b = [rng.gauss(0, 1) for _ in range(N)]
    yr = [2 * u + rng.gauss(0, 0.1) for u in a]
    ev = [int(u > 0) for u in a]
    return {"x": [[u, v] for u, v in zip(a, b)], "a": a, "b": b, "regression": yr,
            "classification": ev, "survival": ([1.0 + abs(v) for v in b], ev),
            "treat": [float(i % 2) for i in range(N)]}

def _model(task, seed=None):
    kw = {} if seed is None else {"seed": seed}
    tree = {"leaf": "mean", "max_depth": 3, "min_rows": 10}
    return {
        "regression": lambda: ModelTree(**tree),
        "classification": lambda: ModelTree(gain="gini", **tree),
        "survival": lambda: ModelTree(gain="pseudo_value", **tree),
        "uplift": lambda: ModelTree(gain="causal", leaf="mean", max_depth=3, min_rows=20),
        "cluster": lambda: ClusterModel(kind="pct", n_clusters=2, **tree, **kw),
        "anomaly": lambda: AnomalyModel(kind="eif", max_depth=4, **kw),
    }[task]()

def _case(task, x=None, seed=None):
    d = _data()
    m = _model(task, seed)
    if task == "uplift":  # Dataset input: treatment and target roles travel with the data
        ds = Dataset({"a": d["a"], "b": d["b"], "arm": d["treat"], "y": d["regression"]}, target="y", treatment="arm")
        m.fit(ds)
        return m, ds, lambda mm, z: mm.predict(z, kind="uplift")
    xs = d["x"] if x is None else x
    m.fit(xs) if task in ("cluster", "anomaly") else m.fit(xs, d[task])
    return m, xs, lambda mm, z: mm.predict(z)

def _finite(v):
    return all(_finite(z) for z in v) if isinstance(v, (list, tuple)) else math.isfinite(float(v))

@pytest.mark.parametrize("task", TASKS)  # 1
def test_fit_predict_length_and_finite(task):
    m, z, pred = _case(task)
    p = pred(m, z)
    assert len(p) == N and _finite(p)

@pytest.mark.parametrize("form", ["list", "numpy", "pandas"])  # 2
@pytest.mark.parametrize("task", NUMERIC)
def test_input_forms(task, form):
    conv = {"list": lambda x: x, "numpy": lambda x: pytest.importorskip("numpy").array(x),
            "pandas": lambda x: pytest.importorskip("pandas").DataFrame(x, columns=["a", "b"])}
    m, z, pred = _case(task, x=conv[form](_data()["x"]))
    assert len(pred(m, z)) == N

@pytest.mark.parametrize("miss", [None, math.nan], ids=["none", "nan"])  # 3
@pytest.mark.parametrize("task", NUMERIC)
def test_missing_values_in_x(task, miss):
    x = [row[:] for row in _data()["x"]]
    x[3][0] = x[7][1] = miss
    m, z, pred = _case(task, x=x)
    p = pred(m, z)
    assert len(p) == N and _finite(p)

RAMP = [0.1 * i for i in range(N)]
EDGE = {  # 4
    "one_row": ([[1.0, 2.0]], [1.0]), "two_rows": ([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0]),
    "all_missing_col": ([[math.nan, v] for v in RAMP], RAMP), "constant_col": ([[3.0, v] for v in RAMP], RAMP),
}

@pytest.mark.parametrize("case", list(EDGE))
@pytest.mark.parametrize("task", ["regression", "cluster", "anomaly"])
def test_edge_inputs_fit_or_clear_error(task, case):
    x, y = EDGE[case]
    m = _model(task)
    try:
        m.fit(x) if task in ("cluster", "anomaly") else m.fit(x, y)
    except Exception as exc:  # noqa: BLE001  pyo3 PanicException is a BaseException and escapes this
        assert str(exc).strip(), "exception has no message"
        return
    assert len(m.predict(x)) == len(x), "silent wrong-length output"

@pytest.mark.parametrize("task", TASKS)  # 5: same seed, then save/load round trip
def test_seed_determinism_and_save_load_round_trip(task, tmp_path):
    m1, z, pred = _case(task, seed=1)
    m2, _, _ = _case(task, seed=1)
    assert pred(m1, z) == pred(m2, z)
    m1.save(tmp_path / "model.json")
    assert pred(type(m1).load(tmp_path / "model.json"), z) == pred(m1, z)

# Each family owns its defaults; ModelTree's are the headline ones.
_FAMILY_DEFAULTS = {
    ModelTree: {"max_depth": 6, "min_rows": 15, "l2": 0.5},
    ClusterModel: {"max_depth": 6, "min_rows": 20},
    RuleModel: {"max_depth": 6, "min_rows": 20},
    AnomalyModel: {"max_depth": 10, "min_rows": 20},
}

@pytest.mark.parametrize("cls", list(_FAMILY_DEFAULTS))
def test_family_defaults(cls):
    cfg = cls().config()
    assert {k: cfg[k] for k in _FAMILY_DEFAULTS[cls]} == _FAMILY_DEFAULTS[cls]

def test_headline_defaults_in_rust_core():  # 6
    c = ModelTree.core_defaults()
    assert [c["max_depth"], c["min_rows"], c["l2"]] == [6, 15, 0.5]

def test_perfect_predictions_give_best_score():  # 7
    d = _data()
    assert MetricsReport("regression", y_true=d["regression"], y_pred=d["regression"])["rmse"] == 0
    assert MetricsReport("classification", y_true=d["classification"], y_pred=d["classification"])["accuracy"] == 1

@pytest.mark.parametrize("task,y_true,y_pred", [  # 7
    ("regression", [1.0, math.nan, 3.0], [1.0, 2.0, 3.0]),
    ("regression", [1.0, 2.0, 3.0], [1.0, math.nan, 3.0]),
    ("classification", [1, math.nan, 0], [1, 0, 0]),
])
def test_nan_in_metrics_is_dropped_and_reported(task, y_true, y_pred):
    rep = MetricsReport(task, y_true=y_true, y_pred=y_pred)
    assert rep["n"] == 2 and rep["n_missing"] == 1

@pytest.mark.parametrize("task,y_true,y_pred", [
    ("regression", [1.0, math.nan, 3.0], [1.0, 2.0, 3.0]),
    ("classification", [1, math.nan, 0], [1, 0, 0]),
])
def test_on_missing_error_raises(task, y_true, y_pred):
    with pytest.raises(ValueError, match="on_missing"):
        MetricsReport(task, y_true=y_true, y_pred=y_pred, on_missing="error")


def test_polars_and_pyarrow_table_inputs():  # issue #30
    pa = pytest.importorskip("pyarrow")
    d = _data()
    tbl = pa.table({"a": d["a"], "b": d["b"]})
    m = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    m.fit(tbl, d["regression"])
    preds = m.predict(tbl)
    assert len(preds) == N and _finite(preds)

    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": d["a"], "b": d["b"]})
    m_pl = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    m_pl.fit(df, d["regression"])
    assert m_pl.predict(df) == preds


def test_predict_column_count_check():  # issue #31a
    d = _data()
    m = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    m.fit({"a": d["a"], "b": d["b"]}, d["regression"])
    with pytest.warns(UserWarning, match="missing"):
        p_few = m.predict([[row[0]] for row in d["x"]])
    assert len(p_few) == N
    with pytest.raises(ValueError, match="missing"):
        m.predict([[row[0]] for row in d["x"]], on_missing_columns="error")
    with pytest.raises(ValueError, match="expected 2"):
        m.predict([[row[0], row[1], 0.0] for row in d["x"]])


def test_fit_missing_y_warns_or_errors():  # issue #31b
    d = _data()
    y_na = [None] + d["regression"][1:]
    with pytest.warns(UserWarning, match="missing values"):
        m = ModelTree(leaf="mean", max_depth=3, min_rows=10)
        m.fit(d["x"], y_na)
    assert len(m.predict(d["x"])) == N
    with pytest.raises(ValueError, match="missing values"):
        ModelTree(leaf="mean", max_depth=3, min_rows=10).fit(d["x"], y_na, y_missing="error")


def test_id_col_resolution_and_predict_layout():  # issue #32a, #32b
    d = _data()
    x_with_id = [[float(i), row[0], row[1]] for i, row in enumerate(d["x"])]
    with pytest.raises(ValueError, match="id_col"):
        ModelTree(max_depth=3, min_rows=10).fit(x_with_id, d["regression"], id_col="id")
    with pytest.raises(ValueError, match="id_col"):
        ModelTree(max_depth=3, min_rows=10).fit(
            x_with_id, d["regression"], id_col="nope", col_names=["id", "a", "b"]
        )
    m = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    m.fit(x_with_id, d["regression"], id_col=0, col_names=["id", "a", "b"])
    assert m.predict(x_with_id) == m.predict(d["x"])


def test_input_boundary_error_messages():  # issue #33a-d
    np = pytest.importorskip("numpy")
    d = _data()
    m = ModelTree(max_depth=3, min_rows=10)
    with pytest.raises(ValueError, match="2-D"):
        m.fit(np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same length"):
        m.fit({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]}, [1.0, 2.0, 3.0])
    pa = pytest.importorskip("pyarrow")
    y_pa = pa.array([None] + d["regression"][1:])
    with pytest.warns(UserWarning, match="missing values"):
        m.fit(d["x"], y_pa)
    with pytest.raises(TypeError, match="encode labels"):
        ModelTree(gain="gini", max_depth=3, min_rows=10).fit(d["x"], ["a", "b"] * (N // 2))


def test_metrics_report_competing_risks():
    dur = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ev = [1, 0, 2, 1, 0, 2]
    times = [2.0, 4.0]
    cif1 = [[0.2, 0.4], [0.1, 0.2], [0.1, 0.3], [0.3, 0.5], [0.1, 0.2], [0.2, 0.3]]
    cif2 = [[0.1, 0.2], [0.1, 0.2], [0.3, 0.5], [0.1, 0.2], [0.1, 0.2], [0.4, 0.6]]
    rep = MetricsReport(
        "competing_risks",
        durations=dur,
        events=ev,
        cif_curves={1: cif1, 2: cif2},
        times=times,
    )
    assert rep["causes"] == 2
    assert rep["cause_1_ibs"] >= 0.0
    assert "per_cause" in rep.tables

    # Issue #34: NaN/None in durations must drop incomplete rows cleanly (no hang or panic)
    dur_na = [math.nan] + dur
    ev_na = [1] + ev
    c1_na = [[0.2, 0.4]] + cif1
    c2_na = [[0.1, 0.2]] + cif2
    rep_na = MetricsReport(
        "competing_risks",
        durations=dur_na,
        events=ev_na,
        cif_curves={1: c1_na, 2: c2_na},
        times=times,
    )
    assert rep_na["causes"] == 2
    with pytest.raises(ValueError, match="on_missing"):
        MetricsReport(
            "competing_risks",
            durations=dur_na,
            events=ev_na,
            cif_curves={1: c1_na, 2: c2_na},
            times=times,
            on_missing="error",
        )


def test_group_col_and_leaf_samples_validation():  # issues #35, #36b, #36c
    d = _data()
    with pytest.raises(ValueError, match="group_col"):
        ModelTree(max_depth=2, min_rows=10).fit(
            d["x"], d["regression"], group_col="zzz", col_names=["a", "b"]
        )
    with pytest.raises(ValueError, match="group_col"):
        ModelTree(max_depth=2, min_rows=10).fit(d["x"], d["regression"], group_col=7)
    with pytest.raises(ValueError, match="group_col"):
        ModelTree(max_depth=2, min_rows=10).fit(d["x"], d["regression"], group_col="g")

    with pytest.raises(ValueError, match="dtypes="):
        ModelTree(max_depth=2, min_rows=10).fit([["a", 1.0], ["b", 2.0]], [1.0, 2.0])

    x_with_id = [[float(i), row[0], row[1]] for i, row in enumerate(d["x"])]
    m = ModelTree(leaf="mean", max_depth=3, min_rows=10)
    m.fit(x_with_id, d["regression"], id_col=0, col_names=["id", "a", "b"])
    with pytest.raises(ValueError, match="without the id column"):
        m.leaf_samples(d["x"], id_col=0)
