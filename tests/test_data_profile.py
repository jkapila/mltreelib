"""Behaviour tests for mltreelib.data.profile (standard library only, hand-checked values)."""

from __future__ import annotations

import copy
import math

import pytest
from mltreelib.data.profile import (
    class_balance,
    constant_columns,
    describe,
    duplicate_columns,
    duplicate_rows,
    infer_types,
    missing_profile,
    missing_vs_target,
    near_zero_variance,
    outlier_flags,
    schema_capture,
    schema_validate,
)

NAN = float("nan")
INF = float("inf")


def _snapshot(obj):
    """NaN-safe snapshot: repr is stable for NaN, and deepcopy guards the original."""
    return repr(obj), copy.deepcopy(obj)


def _unchanged(obj, snap) -> bool:
    return repr(obj) == snap[0]


# ---------------------------------------------------------------- describe


def test_describe_numeric_and_categorical_hand_values():
    x = [1.5, 2.5, 3.5, 4.5, None, NAN]
    c = ["b", "a", "b", None, "c", "b"]
    cols = {"x": x, "c": c}
    snap = _snapshot(cols)
    rows = describe(cols)
    assert _unchanged(cols, snap)
    num, cat = rows
    assert [r["name"] for r in rows] == ["x", "c"]
    assert num["kind"] == "numeric" and num["kind_source"] == "inferred"
    assert num["n"] == 6 and num["n_missing"] == 2
    assert math.isclose(num["pct_missing"], 100 * 2 / 6)
    assert num["n_inf"] == 0 and num["n_distinct"] == 4
    assert math.isclose(num["mean"], 3.0)
    assert math.isclose(num["std"], math.sqrt(5 / 3))
    assert (num["min"], num["max"]) == (1.5, 4.5)
    assert math.isclose(num["q1"], 2.25) and math.isclose(num["median"], 3.0)
    assert math.isclose(num["q3"], 3.75)
    assert math.isclose(num["skew"], 0.0, abs_tol=1e-12)
    # adjusted excess kurtosis: g2 = -1.36, G2 = (5*g2 + 6) * 3 / 2 = -1.2
    assert math.isclose(num["kurtosis"], -1.2)
    # percentiles (R type 7): p05 = 1.5 + 0.15, p95 = 3.5 + 0.85
    assert math.isclose(num["p05"], 1.65) and math.isclose(num["p25"], 2.25)
    assert math.isclose(num["p50"], 3.0) and math.isclose(num["p75"], 3.75)
    assert math.isclose(num["p95"], 4.35)
    assert math.isclose(num["iqr"], 1.5)
    assert math.isclose(num["mad"], 1.0)  # |x - 3| = 1.5, .5, .5, 1.5 -> median 1.0
    assert math.isclose(num["cv"], math.sqrt(5 / 3) / 3)
    # trim 0.1 of 4 values removes floor(0.4) = 0 per end
    assert math.isclose(num["trimmed_mean"], 3.0)
    assert math.isclose(num["trimmed_std"], math.sqrt(5 / 3))
    assert num["n_outliers_iqr"] == 0 and num["pct_outliers_iqr"] == 0.0
    assert (num["n_zero"], num["n_negative"], num["n_positive"]) == (0, 0, 4)
    # numeric statistics do not hide categorical ones: mode and top for every column
    assert num["mode"] == 1.5
    assert num["top"] == [(1.5, 1), (2.5, 1), (3.5, 1), (4.5, 1)]

    assert cat["kind"] == "categorical"
    assert cat["n_missing"] == 1 and cat["n_distinct"] == 3
    assert cat["mode"] == "b"
    assert cat["top"] == [("b", 3), ("a", 1), ("c", 1)]
    assert all(
        cat[k] is None
        for k in ("mean", "std", "min", "q1", "skew", "kurtosis", "cv", "iqr", "p05")
    )
    assert set(num) == set(cat)


def test_describe_infinity_counted_and_ignored_by_statistics():
    z = describe({"z": [INF, 1.5, None]})[0]
    assert z["kind"] == "numeric"
    assert z["n_inf"] == 1 and z["n_missing"] == 1
    assert z["mean"] == 1.5 and z["std"] is None  # one finite value only


def test_describe_inf_as_missing_changes_counts_only():
    cols = {"z": [INF, 1.5, None]}
    snap = _snapshot(cols)
    z = describe(cols, inf_as_missing=True)[0]
    assert _unchanged(cols, snap)
    assert z["n_missing"] == 2 and math.isclose(z["pct_missing"], 200 / 3)
    assert z["n_inf"] == 1  # still reported separately
    assert z["mean"] == 1.5 and z["std"] is None  # statistics unchanged


def test_describe_empty_and_all_missing_do_not_raise():
    rows = describe({"e": [], "m": [None, NAN]})
    assert rows[0]["n"] == 0 and rows[0]["pct_missing"] == 0.0
    assert rows[0]["mean"] is None and rows[0]["n_distinct"] == 0
    assert rows[1]["n_missing"] == 2 and rows[1]["pct_missing"] == 100.0
    assert rows[1]["kind"] == "numeric" and rows[1]["median"] is None
    assert rows[1]["p50"] is None and rows[1]["trimmed_mean"] is None
    assert describe({}) == []


def test_describe_flag_is_ordinal_candidate_with_mean():
    cols = {"flag": [0, 1, 1, 0, None]}
    snap = _snapshot(cols)
    f = describe(cols)[0]
    assert _unchanged(cols, snap)
    assert f["kind"] == "ordinal_candidate" and f["kind_source"] == "inferred"
    assert f["n_missing"] == 1 and f["n_distinct"] == 2
    # observed 0, 1, 1, 0
    assert f["mean"] == 0.5
    assert math.isclose(f["std"], math.sqrt(1 / 3))
    assert f["median"] == 0.5 and f["iqr"] == 1.0
    assert f["n_outliers_iqr"] == 0
    assert (f["n_zero"], f["n_negative"], f["n_positive"]) == (2, 0, 2)
    assert f["mode"] == 0  # tie at 2 each: smaller str first
    assert f["top"] == [(0, 2), (1, 2)]


def test_describe_rating_1_to_5_is_ordinal_candidate_with_stats():
    r = describe({"r": [1, 2, 3, 4, 5, 5, 4, None]})[0]
    assert r["kind"] == "ordinal_candidate"
    assert r["n_distinct"] == 5
    assert math.isclose(r["mean"], 24 / 7)  # 1+2+3+4+5+5+4 = 24 over 7
    assert r["mode"] == 4  # 4 and 5 twice each: smaller str first
    assert r["top"] == [(4, 2), (5, 2), (1, 1), (2, 1), (3, 1)]


def test_describe_six_integer_levels_are_categorical_with_numeric_stats():
    codes = describe({"c": [0, 1, 2, 3, 4, 5]})[0]
    assert codes["kind"] == "categorical"
    assert codes["mean"] == 2.5  # numeric statistics are computed for numeric values
    assert codes["mode"] == 0  # all tied at 1: smallest str first


def test_describe_kinds_override_is_echoed_and_marked_user():
    cols = {"r": [1, 2, 3, 4, 5], "c": ["a", "b", "a", "b", "c"]}
    snap = _snapshot(cols)
    r, c = describe(cols, kinds={"r": "ordinal", "c": "numeric"})
    assert _unchanged(cols, snap)
    assert r["kind"] == "ordinal" and r["kind_source"] == "user"
    assert r["mean"] == 3.0
    assert c["kind"] == "numeric" and c["kind_source"] == "user"
    assert c["mean"] is None  # strings have no numeric statistics
    assert describe(cols)[0]["kind"] == "ordinal_candidate"


def test_describe_kinds_validation():
    with pytest.raises(ValueError):
        describe({"r": [1, 2]}, kinds={"r": "ordinal_candidate"})  # use 'ordinal'
    with pytest.raises(ValueError):
        describe({"r": [1, 2]}, kinds={"nope": "numeric"})  # unknown column


def test_describe_trimmed_statistics_with_known_outlier():
    v = describe({"v": [1, 2, 3, 4, 100]}, trim=0.2)[0]
    # floor(5 * 0.2) = 1 removed from each end: [2, 3, 4]
    assert math.isclose(v["trimmed_mean"], 3.0)
    assert math.isclose(v["trimmed_std"], 1.0)
    assert math.isclose(v["mean"], 22.0) and v["median"] == 3.0
    assert v["n_outliers_iqr"] == 1 and math.isclose(v["pct_outliers_iqr"], 20.0)


def test_describe_trim_zero_gives_plain_mean_and_std():
    v = describe({"v": [1, 2, 3, 4, 100]}, trim=0.0)[0]
    assert v["trimmed_mean"] == v["mean"] == 22.0
    assert math.isclose(v["trimmed_std"], v["std"])


def test_describe_trim_range_is_checked():
    with pytest.raises(ValueError):
        describe({"v": [1, 2]}, trim=0.5)


def test_describe_percentiles_keys_and_values():
    v = describe({"v": [1, 2, 3, 4, 100]})[0]
    # R type 7 on [1, 2, 3, 4, 100]: p05 pos 0.2, p95 pos 3.8
    assert math.isclose(v["p05"], 1.2) and math.isclose(v["p95"], 80.8)
    custom = describe({"v": [1, 2, 3, 4, 100]}, percentiles=(0.0, 1.0, 0.5))[0]
    assert custom["p00"] == 1.0 and custom["p100"] == 100.0 and custom["p50"] == 3.0
    assert "p05" not in custom


def test_describe_percentiles_must_be_whole_percents():
    with pytest.raises(ValueError):
        describe({"v": [1, 2]}, percentiles=(0.125,))
    with pytest.raises(ValueError):
        describe({"v": [1, 2]}, percentiles=(1.5,))


def test_describe_sign_counts_and_cv():
    s = describe({"s": [-2, 0, 0, 3, None]})[0]
    assert (s["n_zero"], s["n_negative"], s["n_positive"]) == (2, 1, 1)
    zero_mean = describe({"z": [-1, 1]})[0]
    assert zero_mean["cv"] is None  # mean is zero
    assert math.isclose(describe({"c": [2, 4]})[0]["cv"], math.sqrt(2) / 3)


# ----------------------------------------------------------- missing_profile


def _missing_cols():
    return {"a": [1, None, 3, None], "b": [NAN, None, 3, 4], "c": [1, 2, 3, 4]}


def test_missing_profile_hand_values():
    cols = _missing_cols()
    snap = _snapshot(cols)
    p = missing_profile(cols)
    assert _unchanged(cols, snap)
    assert (p["n_rows"], p["n_cols"]) == (4, 3)
    assert p["per_column"]["a"] == {"n_missing": 2, "pct_missing": 50.0}
    assert p["per_column"]["c"] == {"n_missing": 0, "pct_missing": 0.0}
    assert p["per_row"] == [1, 2, 0, 1]
    assert p["n_complete_rows"] == 1
    assert p["patterns"] == [
        {"columns": (), "count": 1},
        {"columns": ("a",), "count": 1},
        {"columns": ("a", "b"), "count": 1},
        {"columns": ("b",), "count": 1},
    ]
    assert p["co_missing"]["names"] == ["a", "b", "c"]
    assert p["co_missing"]["matrix"] == [[2, 1, 0], [1, 2, 0], [0, 0, 0]]


def test_missing_profile_inf_as_missing_flag():
    cols = {"a": [INF, None, 1], "b": [INF, 2, None]}
    snap = _snapshot(cols)
    default = missing_profile(cols)
    assert _unchanged(cols, snap)
    assert default["per_column"]["a"]["n_missing"] == 1
    assert default["per_row"] == [0, 1, 1]
    assert default["patterns"] == [
        {"columns": (), "count": 1},
        {"columns": ("a",), "count": 1},
        {"columns": ("b",), "count": 1},
    ]
    assert default["co_missing"]["matrix"] == [[1, 0], [0, 1]]

    flagged = missing_profile(cols, inf_as_missing=True)
    assert flagged["per_column"]["a"]["n_missing"] == 2
    assert flagged["per_row"] == [2, 1, 1] and flagged["n_complete_rows"] == 0
    assert flagged["patterns"] == [
        {"columns": ("a",), "count": 1},
        {"columns": ("a", "b"), "count": 1},
        {"columns": ("b",), "count": 1},
    ]
    assert flagged["co_missing"]["matrix"] == [[2, 1], [1, 2]]


def test_missing_profile_patterns_sorted_by_count():
    p = missing_profile({"a": [None, None, None, 1], "b": [1, 2, 3, 4]})
    assert p["patterns"][0] == {"columns": ("a",), "count": 3}


def test_missing_profile_empty_and_all_missing():
    empty = missing_profile({"a": []})
    assert empty["n_rows"] == 0 and empty["patterns"] == [] and empty["per_row"] == []
    allm = missing_profile({"a": [None, None], "b": [NAN, None]})
    assert allm["n_complete_rows"] == 0
    assert allm["co_missing"]["matrix"] == [[2, 2], [2, 2]]
    assert allm["patterns"] == [{"columns": ("a", "b"), "count": 2}]


def test_missing_profile_rejects_ragged_columns():
    with pytest.raises(ValueError):
        missing_profile({"a": [1, 2], "b": [1]})


# -------------------------------------------------------- missing_vs_target


def test_missing_vs_target_hand_values_and_excluded_target():
    cols = {"x": [1, None, 3, 4, None]}
    y = [1, 0, 0, 1, None]
    snap = _snapshot((cols, y))
    r = missing_vs_target(cols, y)
    assert _unchanged((cols, y), snap)
    x = r["x"]
    assert x["n_missing"] == 1 and x["n_observed"] == 3
    assert x["y_mean_when_missing"] == 0.0
    assert math.isclose(x["y_mean_when_observed"], 2 / 3)
    assert x["n_y_missing_excluded"] == 1


def test_missing_vs_target_inf_as_missing_flag():
    cols = {"x": [INF, 2, None]}
    y = [1, 0, 0]
    snap = _snapshot((cols, y))
    default = missing_vs_target(cols, y)["x"]
    assert _unchanged((cols, y), snap)
    assert default["n_missing"] == 1 and default["y_mean_when_missing"] == 0.0
    assert default["n_observed"] == 2 and default["y_mean_when_observed"] == 0.5
    flagged = missing_vs_target(cols, y, inf_as_missing=True)["x"]
    assert flagged["n_missing"] == 2 and flagged["y_mean_when_missing"] == 0.5
    assert flagged["n_observed"] == 1 and flagged["y_mean_when_observed"] == 0.0


def test_missing_vs_target_nan_and_empty_groups():
    r = missing_vs_target({"x": [None, None]}, [1, 0])
    assert r["x"]["y_mean_when_observed"] is None
    assert r["x"]["y_mean_when_missing"] == 0.5
    e = missing_vs_target({"x": []}, [])
    assert e["x"]["y_mean_when_missing"] is None and e["x"]["n_y_missing_excluded"] == 0
    nan_y = missing_vs_target({"x": [1.0, 2.0]}, [NAN, 2.0])
    assert nan_y["x"]["n_observed"] == 1 and nan_y["x"]["n_y_missing_excluded"] == 1


def test_missing_vs_target_rejects_non_numeric_target():
    with pytest.raises(TypeError):
        missing_vs_target({"x": [1, 2]}, ["yes", "no"])


# ------------------------------------------------------------- infer_types


def test_infer_types_hand_values():
    cols = {
        "s": ["a", "b", None],
        "f": [1.5, 2.5, None],
        "codes": [0, 1, 2, 1, 0],  # integer-valued, 3 levels: ordinal candidate
        "six": [0, 1, 2, 3, 4, 5],  # integer-valued, 6 levels: categorical
        "big": list(range(30)),  # more than 20 integer levels: numeric
    }
    snap = _snapshot(cols)
    t = infer_types(cols)
    assert _unchanged(cols, snap)
    assert t["s"] == {"kind": "categorical", "n_levels": 2, "ordinal_candidate": False}
    assert t["f"] == {"kind": "numeric", "n_levels": 2, "ordinal_candidate": False}
    assert t["codes"] == {
        "kind": "ordinal_candidate",
        "n_levels": 3,
        "ordinal_candidate": True,
    }
    assert t["six"] == {
        "kind": "categorical",
        "n_levels": 6,
        "ordinal_candidate": False,
    }
    assert t["big"] == {"kind": "numeric", "n_levels": 30, "ordinal_candidate": False}


def test_infer_types_n_category_threshold_and_kinds_override():
    assert infer_types({"big": list(range(30))}, n_category=30)["big"]["kind"] == (
        "categorical"
    )
    cols = {"codes": [0, 1, 2, 1, 0], "big": list(range(30))}
    snap = _snapshot(cols)
    t = infer_types(cols, kinds={"codes": "ordinal", "big": "categorical"})
    assert _unchanged(cols, snap)
    # the override replaces kind; ordinal_candidate still reports the data hint
    assert t["codes"] == {"kind": "ordinal", "n_levels": 3, "ordinal_candidate": True}
    assert t["big"] == {
        "kind": "categorical",
        "n_levels": 30,
        "ordinal_candidate": False,
    }


def test_infer_types_bool_and_all_missing():
    t = infer_types({"flag": [True, False, None], "e": [None, NAN]})
    assert t["flag"]["kind"] == "categorical"
    assert t["e"] == {"kind": "numeric", "n_levels": 0, "ordinal_candidate": False}
    assert infer_types({}) == {}


# ------------------------------------------------------ near_zero_variance


def test_near_zero_variance_flags_caret_rule_only():
    cols = {
        "a": [1, 1, 1, 1, 2],  # freq ratio 4: not flagged
        "b": [1] * 20 + [2],  # freq ratio 20 > 19 and 9.52% unique < 10%
        "c": [1, 2, 3, 4],
        "d": [5, 5, 5],  # zero variance
        "e": [None, None],  # no observed values: zero variance
    }
    snap = _snapshot(cols)
    flagged = near_zero_variance(cols)
    assert _unchanged(cols, snap)
    assert [f["name"] for f in flagged] == ["b", "d", "e"]
    b, d, e = flagged
    assert b["freq_ratio"] == 20.0 and b["zero_var"] is False
    assert math.isclose(b["pct_unique"], 200 / 21)
    assert d["freq_ratio"] == INF and d["zero_var"] is True
    assert math.isclose(d["pct_unique"], 100 / 3)
    assert e["pct_unique"] == 0.0 and e["zero_var"] is True


def test_near_zero_variance_empty_input():
    assert near_zero_variance({}) == []


# ------------------------------------------------------- constant_columns


def test_constant_columns_includes_all_missing():
    cols = {"a": [1, 1, None], "b": [1, 2, 3], "c": [None, None], "d": [NAN]}
    snap = _snapshot(cols)
    assert constant_columns(cols) == ["a", "c", "d"]
    assert _unchanged(cols, snap)
    assert constant_columns({}) == []


# ------------------------------------------------------------ duplicate_rows


def test_duplicate_rows_missing_equals_missing_and_int_equals_float():
    cols = {"a": [1, 2, 1.0, None, NAN], "b": ["x", "y", "x", None, None]}
    snap = _snapshot(cols)
    assert duplicate_rows(cols) == [[0, 2], [3, 4]]
    assert _unchanged(cols, snap)


def test_duplicate_rows_all_missing_and_empty():
    assert duplicate_rows({"a": [None, NAN]}) == [[0, 1]]
    assert duplicate_rows({"a": []}) == []
    assert duplicate_rows({}) == []


# --------------------------------------------------------- duplicate_columns


def test_duplicate_columns_groups_equal_columns():
    cols = {
        "a": [1, 2, None],
        "b": [1.0, 2.0, NAN],
        "c": [2, 1, None],
        "d": [1, 2, None],
    }
    snap = _snapshot(cols)
    assert duplicate_columns(cols) == [["a", "b", "d"]]
    assert _unchanged(cols, snap)


def test_duplicate_columns_all_missing_and_empty():
    assert duplicate_columns({"x": [None], "y": [NAN]}) == [["x", "y"]]
    assert duplicate_columns({}) == []


# ------------------------------------------------------------ class_balance


def test_class_balance_hand_values():
    y = [0, 1, 0, 0, None, 1, 0]
    snap = _snapshot(y)
    b = class_balance(y)
    assert _unchanged(y, snap)
    assert b["counts"] == {0: 4, 1: 2}
    assert b["proportions"] == {0: 4 / 6, 1: 2 / 6}
    assert b["imbalance_ratio"] == 2.0
    assert b["n_missing"] == 1 and b["n"] == 7


def test_class_balance_ties_and_missing_strings():
    b = class_balance(["b", "a", "b", None])
    assert list(b["counts"]) == ["b", "a"]  # count desc, so b first
    tie = class_balance(["b", "a"])
    assert list(tie["counts"]) == ["a", "b"]  # equal counts: str asc
    assert tie["imbalance_ratio"] == 1.0


def test_class_balance_single_level_and_empty():
    one = class_balance([1, 1, NAN])
    assert one["imbalance_ratio"] is None and one["n_missing"] == 1
    e = class_balance([])
    assert e == {
        "counts": {},
        "proportions": {},
        "imbalance_ratio": None,
        "n_missing": 0,
        "n": 0,
    }


# ------------------------------------------------------------ outlier_flags


def test_outlier_flags_iqr_hand_values():
    values = [1, 2, 3, 4, 100, None, INF]
    snap = _snapshot(values)
    flags = outlier_flags(values)  # q1=2, q3=4, fence [-1, 7]
    assert _unchanged(values, snap)
    assert flags == [False, False, False, False, True, None, True]


def test_outlier_flags_mad_hand_values_and_zero_mad():
    assert outlier_flags([1, 2, 3, 4, 100], method="mad") == [False] * 4 + [True]
    # MAD is zero: nothing finite is flagged, infinity still is
    assert outlier_flags([5, 5, 5, 5, 9, INF], method="mad") == [False] * 5 + [True]


def test_outlier_flags_empty_all_missing_and_errors():
    assert outlier_flags([]) == []
    assert outlier_flags([None, NAN]) == [None, None]
    with pytest.raises(TypeError):
        outlier_flags([1, "a"])
    with pytest.raises(TypeError):
        outlier_flags([True, 2])
    with pytest.raises(ValueError):
        outlier_flags([1, 2], method="zscore")


# ----------------------------------------------------- schema capture/validate


_TRAIN = {"x": [1.5, 2.5, None], "c": ["b", "a", "b"]}


def test_schema_capture_hand_values():
    s = schema_capture(_TRAIN)
    assert s["n_rows"] == 3
    assert s["columns"]["x"] == {
        "kind": "numeric",
        "kind_source": "inferred",
        "n_missing": 1,
        "min": 1.5,
        "max": 2.5,
        "levels": None,
    }
    assert s["columns"]["c"] == {
        "kind": "categorical",
        "kind_source": "inferred",
        "n_missing": 0,
        "min": None,
        "max": None,
        "levels": ["a", "b"],
    }


def test_schema_capture_ordinal_stores_levels():
    s = schema_capture({"flag": [0, 1, 1, None]})["columns"]["flag"]
    assert s["kind"] == "ordinal_candidate"
    assert s["levels"] == [0, 1] and s["min"] is None


def test_schema_capture_truncates_levels_and_leaves_input_unchanged():
    cols = {"c": [f"v{i:03d}" for i in range(250)]}
    snap = _snapshot(cols)
    entry = schema_capture(cols)["columns"]["c"]
    assert _unchanged(cols, snap)
    assert len(entry["levels"]) == 200 and entry["levels_truncated"] is True


def test_schema_validate_clean_data_and_problems():
    schema = schema_capture(_TRAIN)
    snap = _snapshot(_TRAIN)
    assert schema_validate(_TRAIN, schema) == []
    assert _unchanged(_TRAIN, snap)

    new = {"x": [0.5, None, 9.0], "c": ["a", "z", None], "extra": [1, 2, 3]}
    problems = [v["problem"] for v in schema_validate(new, schema)]
    assert problems == ["out_of_range", "unknown_level", "extra_column"]
    strict = [v["problem"] for v in schema_validate(new, schema, allow_missing=False)]
    assert strict == [
        "out_of_range",
        "unexpected_missing",
        "unknown_level",
        "unexpected_missing",
        "extra_column",
    ]


def test_schema_validate_missing_column_kind_mismatch_and_empty():
    schema = schema_capture(_TRAIN)
    v = schema_validate({"x": ["a", "b"]}, schema)
    assert {(i["column"], i["problem"]) for i in v} == {
        ("x", "kind_mismatch"),
        ("c", "missing_column"),
    }
    # an all-missing column fits any kind
    assert schema_validate({"x": [None, None], "c": [None, None]}, schema) == []
    assert schema_validate({}, schema_capture({})) == []


def test_schema_validate_hint_change_is_not_a_violation():
    # stored numeric; new integer data is ordinal_candidate but still numeric-valued
    schema = schema_capture({"x": [1.5, 2.5, 3.5]})
    assert schema_validate({"x": [2, 3, 2]}, schema) == []
    # and the other way: stored ordinal, new data is numeric floats within levels
    flag = schema_capture({"flag": [0, 1]})
    assert schema_validate({"flag": [1.0, 0.0]}, flag) == []
    assert [v["problem"] for v in schema_validate({"flag": [0, 1, 2]}, flag)] == [
        "unknown_level"
    ]


def test_schema_validate_user_declared_kind_must_match_exactly():
    schema = copy.deepcopy(schema_capture({"x": [1.5, 2.5, 3.5]}))
    schema["columns"]["x"]["kind_source"] = "user"
    problems = [v["problem"] for v in schema_validate({"x": [2, 3, 2]}, schema)]
    # stored 'numeric', data hint 'ordinal_candidate'
    assert problems == ["kind_mismatch"]


def test_schema_validate_skips_unknown_level_check_when_truncated():
    big = {"c": [f"v{i:03d}" for i in range(250)]}
    schema = schema_capture(big)
    assert schema_validate({"c": ["not-seen"]}, schema) == []
