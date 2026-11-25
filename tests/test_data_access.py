"""Tests for mltreelib.data.access (xy, survival_xy, uplift_xy, select_rows)."""

from __future__ import annotations

import copy
import math

import pytest
from mltreelib.data.access import select_rows, survival_xy, uplift_xy, xy

NAN = math.nan
INF = math.inf


def _same(a, b) -> bool:
    """NaN-safe deep equality for nested lists of plain values."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    return type(a) is type(b) and a == b


def _load_result(columns: dict, dtypes: dict) -> dict:
    """A dict shaped like load_dataset output."""
    return {
        "name": "toy",
        "columns": columns,
        "dtypes": dtypes,
        "n_rows": len(next(iter(columns.values()))),
        "n_cols": len(columns),
        "na_counts": {k: sum(v is None for v in col) for k, col in columns.items()},
    }


# ---------------------------------------------------------------- xy


def test_xy_normal_case_hand_checked():
    cols = {
        "a": [1, 2, 3],
        "b": [0.5, 1.5, 2.5],
        "c": ["u", "v", "u"],
        "y": [10.0, 20.0, 30.0],
    }
    out = xy(cols, "y")
    assert out["x"] == [[1, 0.5, "u"], [2, 1.5, "v"], [3, 2.5, "u"]]
    assert out["y"] == [10.0, 20.0, 30.0]
    assert out["names"] == ["a", "b", "c"]
    assert out["dtypes"] == ["numeric", "numeric", "categorical"]
    assert out["n_y_missing"] == 0


def test_xy_missing_and_nan_kept_and_counted_inputs_unchanged():
    cols = {
        "a": [1.0, None, NAN, 4.0],
        "c": ["p", None, "q", NAN],
        "y": [1.0, None, NAN, 2.0],
    }
    before = copy.deepcopy(cols)
    out = xy(cols, "y")
    assert len(out["x"]) == 4  # no row dropped
    assert _same(out["x"], [[1.0, "p"], [None, None], [NAN, "q"], [4.0, NAN]])
    assert _same(out["y"], [1.0, None, NAN, 2.0])
    assert out["n_y_missing"] == 2
    assert out["dtypes"] == ["numeric", "categorical"]
    assert _same(cols, before)


def test_xy_infinity_passes_through():
    cols = {"a": [INF, -INF, 1.0], "y": [0.0, 1.0, INF]}
    out = xy(cols, "y")
    assert out["x"] == [[INF], [-INF], [1.0]]
    assert out["y"] == [0.0, 1.0, INF]
    assert out["n_y_missing"] == 0


def test_xy_empty_and_all_missing_columns():
    empty = xy({"a": [], "y": []}, "y")
    assert empty["x"] == [] and empty["y"] == [] and empty["n_y_missing"] == 0
    assert empty["dtypes"] == ["numeric"]

    all_missing = xy({"a": [None, None], "y": [NAN, None]}, "y")
    assert len(all_missing["x"]) == 2
    assert all_missing["x"] == [[None], [None]]
    assert all_missing["dtypes"] == ["numeric"]
    assert all_missing["n_y_missing"] == 2


def test_xy_load_dataset_shape_uses_declared_dtypes():
    # Integer codes that would infer as numeric; the dataset says categorical.
    res = _load_result(
        {"code": [1, 2, 1, None], "x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 2.0, 3.0, 4.0]},
        {"code": "categorical", "x": "numeric", "y": "numeric"},
    )
    out = xy(res, "y")
    assert out["names"] == ["code", "x"]
    assert out["dtypes"] == ["categorical", "numeric"]
    assert out["x"][0] == [1, 0.1]


def test_xy_infers_dtype_without_declared_dtypes():
    out = xy({"flag": [True, False], "s": ["a", None], "n": [1, 2], "y": [0, 1]}, "y")
    assert out["dtypes"] == ["categorical", "categorical", "numeric"]


def test_xy_features_and_exclude():
    cols = {"a": [1, 2], "b": [3, 4], "c": [5, 6], "y": [0, 1]}
    assert xy(cols, "y", exclude=["b"])["names"] == ["a", "c"]
    assert xy(cols, "y", features=["c", "a"])["x"] == [[5, 1], [6, 2]]
    assert xy(cols, "y", features=["c", "a", "b"], exclude=("a",))["names"] == [
        "c",
        "b",
    ]
    assert xy(cols, "y", features=[])["x"] == [[], []]


def test_xy_rows_are_new_lists():
    cols = {"a": [1, 2], "y": [0, 1]}
    out = xy(cols, "y")
    out["x"][0].append(99)
    out["y"].append(7)
    assert cols == {"a": [1, 2], "y": [0, 1]}


def test_xy_errors():
    cols = {"a": [1, 2], "y": [0, 1]}
    with pytest.raises(KeyError, match="target.*nope"):
        xy(cols, "nope")
    with pytest.raises(KeyError, match="feature.*nope"):
        xy(cols, "y", features=["nope"])
    with pytest.raises(KeyError, match="exclude.*nope"):
        xy(cols, "y", exclude=["nope"])
    with pytest.raises(ValueError, match="ragged"):
        xy({"a": [1, 2, 3], "y": [0, 1]}, "y")
    with pytest.raises(ValueError):
        xy(cols, "y", features=["y"])
    with pytest.raises(ValueError):
        xy(cols, "y", features=["a", "a"])


def test_xy_deterministic():
    cols = {"a": [1, None, 3], "y": [0.0, 1.0, NAN]}
    assert _same(xy(cols, "y")["x"], xy(cols, "y")["x"])


# ---------------------------------------------------------------- survival_xy


def test_survival_xy_hand_checked_and_codes_untouched():
    cols = {"t": [5.0, None, 9.0], "died": [1, 2, 0], "age": [60, 70, NAN]}
    before = copy.deepcopy(cols)
    out = survival_xy(cols, "t", "died")
    assert out["names"] == ["age"]
    assert out["dtypes"] == ["numeric"]
    assert _same(out["x"], [[60], [70], [NAN]])
    assert _same(out["time"], [5.0, None, 9.0])
    assert out["event"] == [1, 2, 0]  # 1/2 codes not recoded
    assert out["n_time_missing"] == 1
    assert out["n_event_missing"] == 0
    assert _same(cols, before)


def test_survival_xy_missing_event_counted_and_kept():
    out = survival_xy({"t": [1.0, 2.0], "e": [None, NAN], "a": ["x", "y"]}, "t", "e")
    assert len(out["x"]) == 2
    assert out["n_event_missing"] == 2
    assert out["dtypes"] == ["categorical"]


def test_survival_xy_empty_and_all_missing():
    assert survival_xy({"t": [], "e": [], "a": []}, "t", "e")["x"] == []
    out = survival_xy({"t": [None], "e": [NAN], "a": [1]}, "t", "e")
    assert out["n_time_missing"] == 1 and out["n_event_missing"] == 1
    assert out["x"] == [[1]]


def test_survival_xy_features_exclude_and_errors():
    cols = {"t": [1.0, 2.0], "e": [0, 1], "a": [1, 2], "b": [3, 4]}
    assert survival_xy(cols, "t", "e", exclude=["a"])["names"] == ["b"]
    assert survival_xy(cols, "t", "e", features=["b"])["names"] == ["b"]
    with pytest.raises(KeyError, match="event.*zz"):
        survival_xy(cols, "t", "zz")
    with pytest.raises(ValueError, match="ragged"):
        survival_xy({"t": [1.0], "e": [0, 1], "a": [1, 2]}, "t", "e")
    with pytest.raises(ValueError):
        survival_xy(cols, "t", "e", features=["t"])


def test_survival_xy_load_dataset_shape():
    res = _load_result(
        {"time": [3.0, 4.0], "status": [2, 1], "trt": ["A", "B"]},
        {"time": "numeric", "status": "numeric", "trt": "categorical"},
    )
    out = survival_xy(res, "time", "status")
    assert out["x"] == [["A"], ["B"]]
    assert out["dtypes"] == ["categorical"]


# ---------------------------------------------------------------- uplift_xy


def test_uplift_xy_hand_checked_and_treatment_untouched():
    cols = {
        "treat": [1, 0, None, 1],
        "conv": [0.0, 1.0, NAN, 1.0],
        "age": [30, 40, 50, 60],
    }
    before = copy.deepcopy(cols)
    out = uplift_xy(cols, "treat", "conv")
    assert out["names"] == ["age"]
    assert out["x"] == [[30], [40], [50], [60]]
    assert out["treatment"] == [1, 0, None, 1]
    assert _same(out["outcome"], [0.0, 1.0, NAN, 1.0])
    assert out["n_treatment_missing"] == 1
    assert out["n_outcome_missing"] == 1
    assert _same(cols, before)


def test_uplift_xy_empty_all_missing_and_errors():
    empty = uplift_xy({"w": [], "o": [], "a": []}, "w", "o")
    assert empty["x"] == [] and empty["n_treatment_missing"] == 0
    out = uplift_xy({"w": [None, None], "o": [1.0, 2.0], "a": [NAN, 1.0]}, "w", "o")
    assert out["n_treatment_missing"] == 2
    assert _same(out["x"], [[NAN], [1.0]])
    with pytest.raises(KeyError, match="outcome"):
        uplift_xy({"w": [0], "a": [1]}, "w", "o")
    with pytest.raises(KeyError, match="treatment"):
        uplift_xy({"o": [0], "a": [1]}, "w", "o")
    with pytest.raises(ValueError, match="ragged"):
        uplift_xy({"w": [0, 1], "o": [0], "a": [1, 2]}, "w", "o")


def test_uplift_xy_features_and_exclude():
    cols = {"w": [0, 1], "o": [1, 0], "a": [1, 2], "b": [3, 4]}
    assert uplift_xy(cols, "w", "o", exclude=["b"])["names"] == ["a"]
    assert uplift_xy(cols, "w", "o", features=["b", "a"])["x"] == [[3, 1], [4, 2]]


# ---------------------------------------------------------------- select_rows


def test_select_rows_hand_checked_and_counts_recomputed():
    res = xy({"a": [1, 2, 3], "y": [0, None, 1]}, "y")
    sub = select_rows(res, [2, 0])
    assert sub["x"] == [[3], [1]]
    assert sub["y"] == [1, 0]
    assert sub["n_y_missing"] == 0
    assert sub["names"] == ["a"] and sub["dtypes"] == ["numeric"]


def test_select_rows_survival_and_uplift_fields():
    s = survival_xy(
        {"t": [1.0, None, 3.0], "e": [1, 2, None], "a": [7, 8, 9]}, "t", "e"
    )
    sub = select_rows(s, [1, 2])
    assert sub["time"][0] is None and sub["time"][1] == 3.0
    assert sub["n_time_missing"] == 1 and sub["n_event_missing"] == 1
    u = uplift_xy({"w": [1, None], "o": [0, 1], "a": [5, 6]}, "w", "o")
    sub_u = select_rows(u, [1])
    assert sub_u["treatment"] == [None] and sub_u["n_treatment_missing"] == 1
    assert sub_u["x"] == [[6]]


def test_select_rows_missing_nan_kept_and_input_unchanged():
    res = xy({"a": [NAN, None], "y": [NAN, 2.0]}, "y")
    before = copy.deepcopy(res)
    sub = select_rows(res, [0, 1, 0])
    assert len(sub["x"]) == 3
    assert _same(sub["y"], [NAN, 2.0, NAN])
    assert sub["n_y_missing"] == 2
    assert _same(res, before)


def test_select_rows_rows_are_new_lists_and_empty_selection():
    res = xy({"a": [1, 2], "y": [0, 1]}, "y")
    sub = select_rows(res, [0])
    sub["x"][0].append(5)
    assert res["x"] == [[1], [2]]
    empty = select_rows(res, [])
    assert empty["x"] == [] and empty["y"] == [] and empty["n_y_missing"] == 0


def test_select_rows_negative_index_and_out_of_range():
    res = xy({"a": [1, 2, 3], "y": [0, 1, 2]}, "y")
    assert select_rows(res, [-1])["y"] == [2]
    with pytest.raises(IndexError):
        select_rows(res, [3])


def test_select_rows_deterministic():
    res = xy({"a": [1, 2, 3], "y": [0, 1, 2]}, "y")
    assert select_rows(res, [1, 1]) == select_rows(res, [1, 1])


# ---------------------------------------------------------------- real data


def test_real_dataset_fits_model_tree():
    from mltreelib import ModelTree, load_dataset

    try:
        ds = load_dataset("concrete_compressive_strength")
    except Exception as exc:  # noqa: BLE001 - offline, no cache, or no pyarrow
        pytest.skip(f"dataset not available offline: {exc}")

    target = "ConcreteCompressiveStrength"
    out = xy(ds, target)
    rows = select_rows(out, range(400))
    assert len(rows["x"]) == 400
    assert rows["n_y_missing"] == 0
    assert rows["dtypes"] == [ds["dtypes"][c] for c in rows["names"]]

    model = ModelTree(leaf="mean", max_depth=3, min_rows=20)
    model.fit(rows["x"], rows["y"], col_names=rows["names"], dtypes=rows["dtypes"])
    pred = model.predict(rows["x"][:10])
    assert len(pred) == 10
    assert all(isinstance(p, float) and math.isfinite(p) for p in pred)
