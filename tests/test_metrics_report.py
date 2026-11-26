"""Behaviour tests for MetricsReport (one-shot printable report over Metrics)."""

from __future__ import annotations

import copy
import json
import math

import pytest
from mltreelib import Metrics, MetricsReport, uplift_threshold
from mltreelib.metrics_report import metrics_report

NAN = math.nan


def _same(a, b) -> bool:
    """Equality that treats NaN as equal to NaN and compares nested containers."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


REG_TRUE = [3.0, 5.0, 2.5, 7.0]
REG_PRED = [2.5, 5.0, 3.0, 8.0]
CLS_TRUE = [1, 0, 1, 1, 0, 0, 1, 0]
CLS_PRED = [1, 0, 0, 1, 0, 1, 1, 0]
CLS_PROBA = [0.9, 0.2, 0.4, 0.8, 0.1, 0.6, 0.7, 0.3]


# -- per-task overview equals the direct Metrics calls -----------------------


def test_regression_overview_hand_checked_and_equal_to_metrics():
    rep = MetricsReport("regression", y_true=REG_TRUE, y_pred=REG_PRED)
    direct = Metrics().regression(REG_TRUE, REG_PRED)
    # Errors are 0.5, 0, -0.5, -1: MAE = 2/4, RMSE = sqrt(1.5/4).
    assert rep["n"] == 4
    assert rep["mae"] == 0.5
    assert math.isclose(rep["rmse"], math.sqrt(0.375), rel_tol=1e-12)
    for key, value in direct.items():
        if key != "deciles":
            assert _same(rep.scalars[key], value), key
    assert rep.tables["error_deciles"] == direct["deciles"]


def test_regression_groups_table_matches_by_group_report():
    groups = ["a", "a", "b", "b"]
    rep = MetricsReport("regression", y_true=REG_TRUE, y_pred=REG_PRED, groups=groups)
    direct = Metrics().regression_by_group(groups, REG_TRUE, REG_PRED)
    rows = {row["group"]: row for row in rep.tables["by_group"]}
    assert rows.keys() == direct.keys() == {"a", "b"}
    for key in ("n", "mae", "rmse", "r2"):
        assert _same(rows["a"][key], direct["a"][key])


def test_classification_overview_hand_checked():
    rep = MetricsReport("classification", y_true=CLS_TRUE, y_pred=CLS_PRED)
    direct = Metrics().classification(CLS_TRUE, CLS_PRED)
    assert rep["accuracy"] == 0.75  # 6 of 8 agree
    assert rep["n_classes"] == 2
    assert "auc" not in rep.scalars  # not computed without probabilities
    assert rep.tables["confusion"][0] == {"actual": 0, "pred 0": 3, "pred 1": 1}
    assert rep.tables["per_class"][0]["support"] == 4
    assert rep["accuracy"] == direct["accuracy"]
    assert (
        rep["balanced_accuracy"]
        == Metrics().classification_scores(CLS_TRUE, CLS_PRED)["balanced_accuracy"]
    )


def test_classification_binary_proba_matches_metrics():
    rep = MetricsReport(
        "classification", y_true=CLS_TRUE, y_pred=CLS_PRED, proba=CLS_PROBA
    )
    m = Metrics()
    direct = m.classification(CLS_TRUE, CLS_PRED, probabilities=CLS_PROBA)
    assert _same(rep["auc"], direct["auc"])
    assert _same(rep["logloss"], direct["logloss"])
    assert _same(rep["brier"], direct["brier"])
    assert (
        rep["average_precision"] == m.pr_curve(CLS_TRUE, CLS_PROBA)["average_precision"]
    )
    assert rep["ks"] == m.ks_statistic(CLS_TRUE, CLS_PROBA)["ks"]
    assert rep.tables["decile"] == m.decile_table(CLS_TRUE, CLS_PROBA)
    assert rep.tables["calibration"] == m.calibration(CLS_TRUE, CLS_PROBA)["bins"]


def test_classification_threshold_derives_labels():
    derived = MetricsReport(
        "classification", y_true=CLS_TRUE, proba=CLS_PROBA, threshold=0.5
    )
    given = MetricsReport(
        "classification",
        y_true=CLS_TRUE,
        y_pred=[1 if p >= 0.5 else 0 for p in CLS_PROBA],
        proba=CLS_PROBA,
    )
    assert derived["threshold"] == 0.5
    assert derived["accuracy"] == given["accuracy"]


def test_classification_multiclass_one_vs_rest():
    y_true = [0, 1, 2, 2, 1, 0]
    y_pred = [0, 1, 2, 1, 1, 0]
    proba = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.1, 0.5, 0.4],
        [0.2, 0.7, 0.1],
        [0.6, 0.3, 0.1],
    ]
    rep = MetricsReport("classification", y_true=y_true, y_pred=y_pred, proba=proba)
    rows = rep.tables["one_vs_rest"]
    assert [row["class"] for row in rows] == [0, 1, 2]
    assert rows[2]["n_positive"] == 2
    ind = [float(t == 2) for t in y_true]
    assert rows[2]["auc"] == Metrics().roc_curve(ind, [p[2] for p in proba])["auc"]
    assert "logloss" not in rep.scalars
    assert any("binary probabilities only" in note for note in rep._notes)


def test_time_series_overview_matches_metrics():
    y = [10.0, 12.0, 11.0, 13.0]
    p = [10.5, 11.5, 11.0, 12.5]
    train = [9.0, 10.0, 11.0, 10.0]
    rep = MetricsReport(
        "time_series", y_true=y, y_pred=p, y_train=train, seasonal_period=2
    )
    direct = Metrics().time_series(y, p, y_train=train, seasonal_period=2)
    assert rep["mae"] == 0.375
    for key, value in direct.items():
        assert _same(rep.scalars[key], value), key


def test_survival_overview_matches_metrics_and_horizon_table():
    dur = [5.0, 8.0, 3.0, 9.0, 6.0, 2.0]
    ev = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    risk = [0.8, 0.3, 0.9, 0.2, 0.5, 0.95]
    times = [3.0, 6.0]
    rep = MetricsReport("survival", durations=dur, events=ev, risk=risk, times=times)
    direct = Metrics().survival(dur, ev, risk, times=times)
    assert rep["n_samples"] == 6 and rep["n_events"] == 4
    for key, value in direct.items():
        if key != "brier_at_horizons":
            assert _same(rep.scalars[key], value), key
    assert [row["time"] for row in rep.tables["brier_at_horizons"]] == [
        t for t, _ in direct["brier_at_horizons"]
    ]


def test_uplift_overview_matches_metrics():
    y = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    t = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    tau = [0.5, 0.1, 0.3, 0.6, 0.2, 0.4, 0.1, 0.0]
    rep = MetricsReport("uplift", y_true=y, treatment=t, tau_pred=tau)
    direct = Metrics().uplift(y, t, tau)
    assert rep["n_treated"] == 4 and rep["n_control"] == 4
    for key, value in direct.items():
        if key not in ("qini_curve", "cumulative_gain"):
            assert _same(rep.scalars[key], value), key
    assert rep.tables["qini_curve"][1] == {
        "population_fraction": 0.125,
        "qini": direct["qini_curve"][1][1],
    }


def test_uplift_policy_table_and_threshold_hand_checked():
    # 12 rows: top 6 (tau >= 0.4) have +1.0 uplift; bottom 6 (tau <= 0.1) have -1.0 uplift.
    y = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    w = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    tau = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4]
    y_orig, w_orig, tau_orig = list(y), list(w), list(tau)

    thr = uplift_threshold(y, w, tau)
    assert y == y_orig and w == w_orig and tau == tau_orig
    assert thr == 0.1

    rep = MetricsReport("uplift", y_true=y, treatment=w, tau_pred=tau, threshold=thr)
    policy = {row["policy"]: row for row in rep.tables["policy"]}
    assert set(policy) == {"selected", "treat_all", "treat_none"}

    sel = policy["selected"]
    # Top 6 rows: 3 treated (all y=1 -> mean 1.0), 3 control (all y=0 -> mean 0.0), q = 6/12 = 0.5 -> value = 0.5
    assert sel["n_selected"] == 6
    assert sel["n_treated"] == 3
    assert sel["n_control"] == 3
    assert math.isclose(sel["value"], 0.5, rel_tol=1e-12)
    assert sel["ci_low"] == sel["ci_high"] == 0.5

    all_row = policy["treat_all"]
    assert all_row["n_selected"] == 12
    assert all_row["n_treated"] == 6
    assert all_row["n_control"] == 6
    assert math.isclose(all_row["value"], 0.0, abs_tol=1e-12)
    assert all_row["ci_low"] < 0.0 < all_row["ci_high"]

    none_row = policy["treat_none"]
    assert none_row["n_selected"] == 0
    assert none_row["value"] == 0.0
    assert none_row["ci_low"] is None and none_row["ci_high"] is None


def test_uplift_threshold_cost_and_min_arm():
    y = [1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
    w = [1, 1, 0, 0, 1, 1, 0, 0]
    tau = [0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.0]
    # With high cost (1.5 > max uplift 1.0), treat-none is optimal.
    assert uplift_threshold(y, w, tau, cost=1.5) == max(tau)
    # With cost=0.25 and threshold=0.3: selected=4 rows, q=0.5, diff=1.0, net=0.5*(1.0-0.25)=0.375
    rep = MetricsReport(
        "uplift", y_true=y, treatment=w, tau_pred=tau, threshold=0.3, cost=0.25
    )
    sel = next(r for r in rep.tables["policy"] if r["policy"] == "selected")
    assert math.isclose(sel["value"], 0.375, rel_tol=1e-12)
    assert rep["cost"] == 0.25

    with pytest.raises(ValueError, match="min_arm must be a positive integer"):
        uplift_threshold(y, w, tau, min_arm=0)
    with pytest.raises(ValueError, match="cost must be a finite number"):
        uplift_threshold(y, w, tau, cost=math.inf)


def test_uplift_by_decile_table_has_ten_deciles_when_n_ge_10():
    n = 20
    y = [1.0 if (i < 10 and i % 2 == 0) or (i >= 10 and i % 2 == 1) else 0.0 for i in range(n)]
    w = [1 if i % 2 == 0 else 0 for i in range(n)]
    tau = [float(n - i) for i in range(n)]
    rep = MetricsReport("uplift", y_true=y, treatment=w, tau_pred=tau)
    deciles = rep.tables["uplift_by_decile"]
    assert len(deciles) == 10
    assert [d["decile"] for d in deciles] == list(range(1, 11))
    assert sum(d["n"] for d in deciles) == n
    assert deciles[0]["mean_tau_pred"] == 19.5
    assert deciles[0]["uplift"] == 1.0
    assert deciles[-1]["uplift"] == -1.0
    assert "Uplift by decile" in rep.to_text()
    assert "## Policy" in rep.to_markdown()
    json.loads(rep.to_json())


def test_uplift_policy_handles_inf_without_nan_ci():
    rep = MetricsReport(
        "uplift",
        y_true=[math.inf, 0.0, 1.0, 0.0],
        treatment=[1.0, 0.0, 1.0, 0.0],
        tau_pred=[0.5, 0.1, 0.3, 0.2],
    )
    sel = next(r for r in rep.tables["policy"] if r["policy"] == "selected")
    assert sel["ci_low"] is None and sel["ci_high"] is None
    json.loads(rep.to_json())


# -- rendering ---------------------------------------------------------------


def test_text_and_markdown_show_headings_and_numbers():
    rep = MetricsReport(
        "regression", y_true=REG_TRUE, y_pred=REG_PRED, groups=["a", "a", "b", "b"]
    )
    text = rep.to_text()
    md = rep.to_markdown()
    for heading in (
        "Metrics report: regression",
        "Inputs",
        "Overview",
        "Error by decile",
        "By group",
    ):
        assert heading in text
    assert "rmse" in text and "0.6124" in text
    assert "## Overview" in md and "| rmse | 0.6124 |" in md
    assert str(rep) == text
    assert repr(rep) == "MetricsReport(task='regression', rows=4)"


def test_to_text_snapshot_tiny_regression():
    rep = MetricsReport("regression", y_true=[1.0, 2.0, 4.0], y_pred=[1.5, 2.0, 3.0])
    assert rep.to_text() == (
        "Metrics report: regression\n"
        "==========================\n"
        "\n"
        "Inputs\n"
        "------\n"
        "Rows per argument (n_missing = None/NaN):\n"
        "  argument  n  n_missing\n"
        "  y_true    3          0\n"
        "  y_pred    3          0\n"
        "\n"
        "Overview\n"
        "--------\n"
        "  n                        3\n"
        "  n_missing                0\n"
        "  mean_error          0.1667\n"
        "  bias                0.9286\n"
        "  mae                    0.5\n"
        "  mad                 0.5556\n"
        "  mape                    25\n"
        "  rmse                0.6455\n"
        "  r2                  0.7321\n"
        "  max_error                1\n"
        "  explained_variance    0.75\n"
        "\n"
        "Error by decile\n"
        "---------------\n"
        "Error profile by target decile:\n"
        "  decile  n  mean_error  mae  mape\n"
        "       1  1        -0.5  0.5    50\n"
        "       2  1        -0.5  0.5    50\n"
        "       3  1        -0.5  0.5    50\n"
        "       4  1        -0.5  0.5    50\n"
        "       5  1           0    0     0\n"
        "       6  1           0    0     0\n"
        "       7  1           0    0     0\n"
        "       8  1           1    1    25\n"
        "       9  1           1    1    25\n"
        "      10  1           1    1    25\n"
    )


def test_to_dict_round_trips_through_json():
    rep = MetricsReport(
        "classification", y_true=CLS_TRUE, y_pred=CLS_PRED
    )  # auc etc. absent
    data = rep.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert json.loads(rep.to_json()) == data
    assert data["scalars"]["accuracy"] == 0.75


def test_to_dict_maps_nan_and_inf_to_json_safe_values():
    rep = MetricsReport(
        "regression", y_true=[1.0, 2.0, 3.0], y_pred=[1.0, math.inf, 3.0]
    )
    data = rep.to_dict()
    assert data["scalars"]["rmse"] == "inf"
    json.dumps(data, allow_nan=False)


# -- missing values ----------------------------------------------------------


def test_regression_missing_counted_not_dropped_or_imputed():
    y_true = [3.0, 5.0, 2.5, 7.0, NAN]
    y_pred = [2.5, NAN, 3.0, 8.0, 1.0]
    y_copy, p_copy = copy.deepcopy(y_true), copy.deepcopy(y_pred)
    rep = MetricsReport("regression", y_true=y_true, y_pred=y_pred)
    assert rep["n_missing"] == 2
    assert rep["n"] == 3  # rows 0, 2 and 3 are complete pairs
    rows = {row["argument"]: row for row in rep.tables["inputs"]}
    assert rows["y_true"] == {"argument": "y_true", "n": 5, "n_missing": 1}
    assert rows["y_pred"] == {"argument": "y_pred", "n": 5, "n_missing": 1}
    assert _same(y_true, y_copy) and _same(y_pred, p_copy)


def test_none_and_nan_give_identical_reports():
    with_none = MetricsReport(
        "regression", y_true=[1.0, None, 3.0], y_pred=[1.0, 2.0, 2.0]
    )
    with_nan = MetricsReport(
        "regression", y_true=[1.0, NAN, 3.0], y_pred=[1.0, 2.0, 2.0]
    )
    assert with_none.to_dict() == with_nan.to_dict()


def test_classification_missing_label_counted():
    y_true = [1, NAN, 0, 1]
    y_pred = [1, 0, 0, None]
    rep = MetricsReport("classification", y_true=y_true, y_pred=y_pred)
    assert rep["n_missing"] == 2 and rep["n"] == 2
    assert any("classification_scores" in note for note in rep._notes)
    assert y_true[1] != y_true[1]  # the caller's NaN is still there
    assert y_pred[3] is None


def test_survival_missing_duration_refused_naming_argument():
    with pytest.raises(ValueError, match="durations has 1 missing value"):
        MetricsReport(
            "survival",
            durations=[5.0, NAN, 3.0],
            events=[1.0, 0.0, 1.0],
            risk=[0.8, 0.3, 0.9],
        )


def test_proba_missing_refused_naming_argument():
    with pytest.raises(ValueError, match="proba has 1 missing value"):
        MetricsReport(
            "classification",
            y_true=[1, 0, 1, 0],
            y_pred=[1, 0, 1, 0],
            proba=[0.9, None, 0.8, 0.1],
        )


def test_uplift_missing_treatment_refused():
    with pytest.raises(ValueError, match="treatment has 1 missing value"):
        MetricsReport(
            "uplift",
            y_true=[1.0, 0.0, 1.0, 0.0],
            treatment=[1.0, None, 0.0, 1.0],
            tau_pred=[0.5, 0.1, 0.3, 0.6],
        )


def test_all_pairs_missing_raises_naming_both_arguments():
    with pytest.raises(ValueError, match="no row has both y_true and y_pred observed"):
        MetricsReport("regression", y_true=[NAN, None], y_pred=[1.0, NAN])


def test_undefined_ranking_metric_is_noted_not_fatal():
    rep = MetricsReport(
        "classification", y_true=[1, 1, 1], y_pred=[1, 1, 0], proba=[0.9, 0.8, 0.4]
    )
    assert rep["ks"] is None
    assert any("ks_statistic is undefined" in note for note in rep._notes)


# -- arguments and errors ----------------------------------------------------


def test_task_name_is_case_insensitive():
    assert (
        MetricsReport("Regression", y_true=REG_TRUE, y_pred=REG_PRED).task
        == "regression"
    )


def test_unknown_task_lists_valid_tasks():
    with pytest.raises(
        ValueError, match="regression, classification, time_series, survival, uplift"
    ):
        MetricsReport("ranking", y_true=[1.0], y_pred=[1.0])


def test_length_mismatch_names_argument():
    with pytest.raises(ValueError, match="y_pred has 1 rows but y_true has 2"):
        MetricsReport("regression", y_true=[1.0, 2.0], y_pred=[1.0])


def test_missing_required_argument_named():
    with pytest.raises(ValueError, match="requires y_pred"):
        MetricsReport("regression", y_true=[1.0])
    with pytest.raises(ValueError, match="requires y_pred \\(labels\\) or proba"):
        MetricsReport("classification", y_true=[1, 0])


def test_argument_not_for_task_rejected():
    with pytest.raises(TypeError, match="survival report does not take y_true"):
        MetricsReport(
            "survival", y_true=[1.0], durations=[1.0], events=[1.0], risk=[1.0]
        )


def test_non_numeric_value_names_argument():
    with pytest.raises(TypeError, match=r"y_true\[0\] must be a number"):
        MetricsReport("regression", y_true=["x"], y_pred=[1.0])


def test_classification_labels_must_be_integers():
    with pytest.raises(TypeError, match=r"y_true\[0\] must be an integer class label"):
        MetricsReport("classification", y_true=[0.5, 1], y_pred=[1, 0])


def test_binary_proba_needs_labels_zero_and_one():
    with pytest.raises(ValueError, match="labels with binary proba"):
        MetricsReport(
            "classification", y_true=[0, 1, 2], y_pred=[0, 1, 2], proba=[0.1, 0.5, 0.9]
        )


def test_threshold_rules():
    with pytest.raises(ValueError, match="threshold only applies with proba"):
        MetricsReport("classification", y_true=[1, 0], y_pred=[1, 0], threshold=0.3)
    with pytest.raises(ValueError, match="not both"):
        MetricsReport(
            "classification",
            y_true=[1, 0],
            y_pred=[1, 0],
            proba=[0.9, 0.1],
            threshold=0.3,
        )


def test_empty_input_raises_clear_error():
    with pytest.raises(ValueError, match="regression report: y_true is empty"):
        MetricsReport("regression", y_true=[], y_pred=[])


def test_getitem_unknown_key_is_key_error():
    rep = MetricsReport("regression", y_true=REG_TRUE, y_pred=REG_PRED)
    with pytest.raises(KeyError, match="no scalar 'zzz'"):
        rep["zzz"]


def test_module_function_is_alias_of_class():
    a = metrics_report("regression", y_true=REG_TRUE, y_pred=REG_PRED)
    b = MetricsReport("regression", y_true=REG_TRUE, y_pred=REG_PRED)
    assert a.to_json() == b.to_json()


def test_report_is_deterministic():
    first = MetricsReport(
        "classification", y_true=CLS_TRUE, y_pred=CLS_PRED, proba=CLS_PROBA
    )
    second = MetricsReport(
        "classification", y_true=CLS_TRUE, y_pred=CLS_PRED, proba=CLS_PROBA
    )
    assert first.to_text() == second.to_text()
