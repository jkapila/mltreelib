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
