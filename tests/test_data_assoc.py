"""Behaviour tests for mltreelib.data.assoc (standard library only)."""

from __future__ import annotations

import copy
import math

import pytest
from mltreelib.data.assoc import (
    _chi2_sf,
    correlation_filter,
    correlation_matrix,
    cramers_v_matrix,
    leakage_hints,
    vif,
)

NAN = float("nan")


def _same(a, b) -> bool:
    """NaN-safe structural equality."""
    return repr(a) == repr(b)


# ---------------------------------------------------------------- chi-square SF


def test_chi2_sf_known_values():
    assert math.isclose(_chi2_sf(3.841458820694124, 1), 0.05, abs_tol=1e-9)
    assert math.isclose(_chi2_sf(5.991464547107979, 2), 0.05, abs_tol=1e-9)
    assert math.isclose(_chi2_sf(4.0, 1), 0.04550026389635842, abs_tol=1e-9)
    assert _chi2_sf(0, 3) == 1.0


# ------------------------------------------------------------ correlation_matrix


def test_correlation_matrix_normal_case_and_skipped():
    cols = {
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],
        "c": [5, 4, 3, 2, 1],
        "s": list("uvwxy"),
    }
    res = correlation_matrix(cols)
    assert res["names"] == ["a", "b", "c"]
    assert res["skipped"] == ["s"]
    assert res["method"] == "pearson"
    m = res["matrix"]
    assert math.isclose(m[0][1], 1.0, abs_tol=1e-12)
    assert math.isclose(m[0][2], -1.0, abs_tol=1e-12)
    assert m[0][0] == 1.0 and m[2][2] == 1.0
    assert m[1][0] == m[0][1] and m[2][0] == m[0][2]  # symmetric
    assert res["n_pairs"][0][1] == 5 and res["n_pairs"][0][0] == 5


def test_correlation_matrix_spearman_with_ties():
    # ranks of x = [1.5, 1.5, 3, 4]; ranks of y = [1, 2, 3, 4]; r = sqrt(0.9)
    res = correlation_matrix({"x": [1, 1, 2, 3], "y": [1, 2, 3, 4]}, method="spearman")
    assert res["method"] == "spearman"
    assert math.isclose(res["matrix"][0][1], math.sqrt(0.9), rel_tol=1e-12)


def test_correlation_matrix_pairwise_missing_counted_and_input_unchanged():
    a = [1, 2, 3, 4, 5, 6]
    d = [1, None, 3, NAN, 5, 6]
    cols = {"a": a, "d": d}
    before = copy.deepcopy(cols)
    res = correlation_matrix(cols)
    assert res["n_pairs"][0][1] == 4  # rows 0, 2, 4, 5
    assert math.isclose(res["matrix"][0][1], 1.0, abs_tol=1e-12)
    assert res["n_pairs"][1][1] == 4  # d observed count
    assert _same(cols, before)


def test_correlation_matrix_n_below_three_and_zero_variance_give_none():
    cols = {
        "a": [1, 2, 3, 4, 5, 6],
        "e": [1, None, None, None, None, 2],  # only 2 complete pairs with a
        "z": [7, 7, 7, 7, 7, 7],
    }
    res = correlation_matrix(cols)
    i = {n: k for k, n in enumerate(res["names"])}
    assert res["matrix"][i["a"]][i["e"]] is None
    assert res["n_pairs"][i["a"]][i["e"]] == 2
    assert res["matrix"][i["a"]][i["z"]] is None
    assert res["matrix"][i["z"]][i["z"]] is None  # zero variance diagonal


def test_correlation_matrix_all_missing_and_empty():
    res = correlation_matrix({"m": [None, NAN]})
    assert res["names"] == ["m"]
    assert res["matrix"] == [[None]]
    assert res["n_pairs"] == [[0]]
    empty = correlation_matrix({})
    assert empty["names"] == [] and empty["matrix"] == []


def test_correlation_matrix_deterministic():
    cols = {"a": [3, 1, 2, 5, 4], "b": [2, 2, 1, 4, 5]}
    assert _same(
        correlation_matrix(cols, "spearman"), correlation_matrix(cols, "spearman")
    )


# ------------------------------------------------------------- correlation_filter


def test_correlation_filter_drops_larger_mean_abs():
    res = {
        "names": ["a", "b", "c"],
        "matrix": [[1.0, 0.95, 0.2], [0.95, 1.0, 0.3], [0.2, 0.3, 1.0]],
    }
    before = copy.deepcopy(res)
    out = correlation_filter(res, cutoff=0.9)
    # mean |r| of a = 0.575, of b = 0.625 -> b dropped
    assert out["drop"] == ["b"]
    assert out["kept"] == ["a", "c"]
    assert out["pairs"] == [{"a": "a", "b": "b", "corr": 0.95}]
    assert _same(res, before)


def test_correlation_filter_tie_drops_later_column():
    res = {"names": ["a", "b"], "matrix": [[1.0, -0.95], [-0.95, 1.0]]}
    out = correlation_filter(res, cutoff=0.9)
    assert out["drop"] == ["b"] and out["kept"] == ["a"]


def test_correlation_filter_ignores_none_and_empty():
    res = {"names": ["a", "b"], "matrix": [[1.0, None], [None, 1.0]]}
    out = correlation_filter(res, cutoff=0.5)
    assert out == {"drop": [], "kept": ["a", "b"], "pairs": []}
    empty = correlation_filter({"names": [], "matrix": []})
    assert empty == {"drop": [], "kept": [], "pairs": []}


# ------------------------------------------------------------------ cramers_v


def test_cramers_v_perfect_and_independent_uncorrected():
    cols = {
        "x": ["a", "a", "b", "b"],
        "y": ["u", "u", "v", "v"],
        "z": ["u", "v", "u", "v"],
    }
    out = cramers_v_matrix(cols, bias_corrected=False)
    assert out["names"] == ["x", "y", "z"]
    assert out["skipped"] == []
    assert math.isclose(out["matrix"][0][1], 1.0, abs_tol=1e-12)
    assert math.isclose(out["p_values"][0][1], 0.04550026389635842, abs_tol=1e-9)
    assert out["matrix"][0][2] == 0.0
    assert math.isclose(out["p_values"][0][2], 1.0, abs_tol=1e-12)
    assert out["matrix"][0][0] == 1.0 and out["n_pairs"][0][0] == 4


def test_cramers_v_bias_corrected_perfect_and_independent():
    cols = {
        "x": ["a", "a", "b", "b"],
        "y": ["u", "u", "v", "v"],
        "z": ["u", "v", "u", "v"],
    }
    out = cramers_v_matrix(cols)  # Bergsma correction, n = 4, r = k = 2
    assert math.isclose(out["matrix"][0][1], 1.0, abs_tol=1e-12)
    assert out["matrix"][0][2] == 0.0


def test_cramers_v_missing_excluded_and_counted_and_numeric_codes():
    cols = {
        "x": ["a", "a", "b", "b", None, "c"],
        "y": ["u", "v", "u", "v", "u", NAN],
        "w": [1, 1, 2, 2, None, None],  # integer codes -> categorical
        "f": [1.5, 2.25, 3.1, 4.7, 5.0, 6.2],  # numeric -> skipped
    }
    before = copy.deepcopy(cols)
    out = cramers_v_matrix(cols, bias_corrected=False)
    assert out["names"] == ["x", "y", "w"]
    assert out["skipped"] == ["f"]
    assert out["n_pairs"][0][1] == 4  # rows 0-3 only
    assert out["n_pairs"][0][0] == 5  # x observed count
    assert out["matrix"][0][1] == 0.0  # x and y independent on rows 0-3
    assert out["n_pairs"][0][2] == 4
    assert _same(cols, before)


def test_cramers_v_one_level_and_empty():
    out = cramers_v_matrix({"c": ["a"] * 6, "d": ["x", "y", "x", "y", "x", "y"]})
    assert out["matrix"][0][0] is None and out["p_values"][0][0] is None
    assert out["matrix"][0][1] is None
    none_out = cramers_v_matrix({})
    assert none_out["names"] == [] and none_out["matrix"] == []
    allmiss = cramers_v_matrix({"m": [None, None]})
    assert allmiss["names"] == [] and allmiss["skipped"] == ["m"]


def test_cramers_v_deterministic():
    cols = {"x": list("aabbc"), "y": list("uvuvu")}
    assert _same(cramers_v_matrix(cols), cramers_v_matrix(cols))


# ------------------------------------------------------------------------ vif


XY = {"x": [1, 2, 3, 4, 5, 6], "y": [2, 1, 4, 3, 6, 5]}  # r = 29/35


def test_vif_two_columns_equals_one_over_one_minus_r2():
    out = vif(XY)
    assert out["n_used"] == 6 and out["n_excluded"] == 0
    assert out["warning"] is None
    expected = 1 / (
        1 - (29 / 35) ** 2
    )  # r = 29/35 (cross-products sum 14.5, SS 17.5 each)
    assert math.isclose(out["vif"]["x"], expected, rel_tol=1e-9)
    assert math.isclose(out["vif"]["y"], expected, rel_tol=1e-9)


def test_vif_missing_rows_excluded_and_counted():
    cols = {"x": [1, 2, 3, 4, 5, 6], "y": [None, NAN, 4, 3, 6, 5]}
    before = copy.deepcopy(cols)
    out = vif(cols)
    assert out["n_used"] == 4 and out["n_excluded"] == 2
    # on rows 2..5: x = 3..6, y = 4,3,6,5 -> r = 0.6 -> VIF = 1 / 0.64
    assert math.isclose(out["vif"]["x"], 1 / 0.64, rel_tol=1e-9)
    assert _same(cols, before)


def test_vif_perfect_collinearity_inf_only_for_involved_columns():
    cols = {"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "z": [1, 0, 0, 1]}
    out = vif(cols)
    assert out["vif"]["x"] == math.inf and out["vif"]["y"] == math.inf
    assert math.isclose(out["vif"]["z"], 1.0, rel_tol=1e-9)


def test_vif_too_few_rows_and_constant_column():
    few = vif({"x": [1, 2], "y": [2, 1]})
    assert few["vif"] == {"x": None, "y": None}
    assert few["warning"]
    cols = {"x": XY["x"], "y": XY["y"], "c": [5, 5, 5, 5, 5, 5]}
    out = vif(cols)
    assert out["vif"]["c"] is None and out["warning"]
    assert math.isclose(out["vif"]["x"], 1225 / 384, rel_tol=1e-9)


def test_vif_skipped_empty_and_deterministic():
    out = vif({"x": XY["x"], "y": XY["y"], "s": list("abcdef")})
    assert out["skipped"] == ["s"] and out["names"] == ["x", "y"]
    empty = vif({})
    assert empty["vif"] == {} and empty["warning"]
    assert _same(vif(XY), vif(XY))


# --------------------------------------------------------------- leakage_hints


def _by_name(hints):
    return {h["name"]: h for h in hints}


def test_leakage_numeric_feature_binary_y_auc_and_flag():
    y = [0, 0, 0, 1, 1, 1]
    h = _by_name(leakage_hints({"f": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}, y))["f"]
    assert h["metric"] == "auc" and h["score"] == 1.0
    assert h["flagged"] is True and h["n_used"] == 6


def test_leakage_missing_feature_and_y_counted_not_dropped_from_input():
    cols = {"f": [1.5, None, 2.5, NAN, 3.5, 4.5]}
    y = [0, 0, 0, 1, 1, NAN]  # y missing at row 5
    before = copy.deepcopy((cols, y))
    out = leakage_hints(cols, y)
    assert len(out) == 1
    assert out[0]["n_used"] == 3  # rows 0, 2, 4
    assert out[0]["score"] == 1.0
    assert _same((cols, y), before)


def test_leakage_constant_integer_feature_is_level_rate_and_not_flagged():
    h = _by_name(leakage_hints({"k": [1, 1, 1, 1, 1, 1]}, [0, 0, 0, 1, 1, 1]))["k"]
    assert h["metric"] == "auc_level_rate"
    assert h["score"] == 0.5 and h["flagged"] is False


def test_leakage_categorical_binary_level_rate_auc():
    cols = {"c": ["a", "a", "b", "b", "c", "c"]}
    h = _by_name(leakage_hints(cols, [0, 0, 0, 1, 1, 1]))["c"]
    # level rates 0, 0, .5, .5, 1, 1 vs labels: AUC = 8.5 / 9
    assert h["metric"] == "auc_level_rate"
    assert math.isclose(h["score"], 8.5 / 9, rel_tol=1e-12)
    assert h["flagged"] is False


def test_leakage_numeric_y_r2_and_eta2():
    yn = [1, 2, 3, 4, 5, 6]
    cols = {"f": [2.5, 4.5, 6.5, 8.5, 10.5, 12.5], "c": ["a", "a", "b", "b", "c", "c"]}
    out = _by_name(leakage_hints(cols, yn))
    assert out["f"]["metric"] == "r2" and math.isclose(
        out["f"]["score"], 1.0, rel_tol=1e-12
    )
    assert out["f"]["flagged"] is True
    assert out["c"]["metric"] == "eta2"
    assert math.isclose(out["c"]["score"], 16 / 17.5, rel_tol=1e-12)
    assert out["c"]["flagged"] is False


def test_leakage_degenerate_all_missing_empty_and_deterministic():
    const_y = leakage_hints({"f": [1, 2, 3]}, [1, 1, 1])
    assert const_y[0]["score"] is None and const_y[0]["flagged"] is False
    allmiss = leakage_hints({"m": [None, NAN, None]}, [0, 1, 0])[0]
    assert allmiss["score"] is None and allmiss["n_used"] == 0
    assert leakage_hints({}, []) == []
    cols = {"f": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}
    y = [0, 0, 0, 1, 1, 1]
    assert _same(leakage_hints(cols, y), leakage_hints(cols, y))


def test_leakage_rejects_bad_y_length_and_multiclass_strings():
    with pytest.raises(ValueError):
        leakage_hints({"f": [1, 2]}, [0, 1, 2])
    with pytest.raises(ValueError):
        leakage_hints({"f": [1.5, 2.5, 3.5]}, ["a", "b", "c"])
