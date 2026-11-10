"""Tests for model explanation and evaluation surface.

Covers:
- break_down (DALEX-style decision path decomposition)
- exact TreeSHAP attribution and efficiency invariant
- permutation importance and unified feature_importance(method=...)
- evaluation curves in Metrics
- plotting integration with coherent theme
"""

from __future__ import annotations

import pytest
from mltreelib import Metrics, ModelTree


def test_break_down_telescoping_python():
    X = [[float(i % 10), float((i * 3) % 7)] for i in range(100)]
    y = [float(2 * (i % 10) + (10.0 if (i % 10) > 5 else 0.0)) for i in range(100)]

    model = ModelTree(max_depth=3, min_rows=5, leaf="mean")
    model.fit(X, y)

    row = [3.0, 4.0]
    bd = model.break_down(row)

    assert "baseline" in bd
    assert "steps" in bd
    assert "residual" in bd
    assert "prediction" in bd

    step_sum = sum(contrib for _, contrib in bd["steps"])
    assert abs(bd["baseline"] + step_sum + bd["residual"] - bd["prediction"]) < 1e-7


def test_treeshap_efficiency_and_importance():
    X = [[float(i % 10), float((i * 3) % 7), 1.0] for i in range(100)]
    y = [float(5 * (i % 10)) for i in range(100)]

    model = ModelTree(max_depth=3, min_rows=5, leaf="mean")
    model.fit(X, y)

    row = [6.0, 2.0, 1.0]
    sh = model.shapley(row)

    assert "base_value" in sh
    assert "values" in sh
    assert "prediction" in sh

    values_dict = dict(sh["values"])
    total_phi = sum(values_dict.values())
    pred = model.predict([row])[0]

    # Efficiency: base_value + sum(values) == prediction
    assert abs(sh["base_value"] + total_phi - pred) < 1e-7

    # Dummy feature (column x2 is constant 1.0) must receive 0 attribution
    assert abs(values_dict["x2"]) < 1e-7

    # Dataset Shapley importance
    shap_imp = model.feature_importance(method="shapley", x=X)
    assert len(shap_imp) == 3
    assert shap_imp[0][0] == "x0"  # x0 carries all signal
    assert dict(shap_imp)["x2"] < 1e-7


def test_permutation_importance_and_unified_api():
    X = [[float(i % 10), float((i * 3) % 7)] for i in range(100)]
    y = [float(4 * (i % 10)) for i in range(100)]

    model = ModelTree(max_depth=3, min_rows=5, leaf="mean")
    model.fit(X, y)

    # Direct permutation_importance call
    perm_imp = model.permutation_importance(X, y, metric="rmse", n_repeats=3, seed=42)
    assert len(perm_imp) == 2
    assert perm_imp[0][0] == "x0"
    assert perm_imp[0][1] > 0.0

    # Unified feature_importance accessor
    gain_imp = model.feature_importance(method="gain")
    assert gain_imp[0][0] == "x0"

    perm_unified = model.feature_importance(method="permutation", x=X, y=y, metric="rmse")
    assert perm_unified[0][0] == "x0"

    with pytest.raises(Exception):
        model.feature_importance(method="unknown_method")


def test_plot_breakdown_and_importance():
    from mltreelib import plots

    X = [[float(i % 10), float((i * 3) % 7)] for i in range(100)]
    y = [float(2 * (i % 10)) for i in range(100)]

    model = ModelTree(max_depth=2, min_rows=5, leaf="mean")
    model.fit(X, y)

    bd = model.break_down([4.0, 1.0])
    ax_bd = plots.plot_breakdown(bd)
    assert ax_bd is not None

    imp = model.feature_importance(method="gain")
    ax_imp = plots.plot_importance(imp)
    assert ax_imp is not None


def test_evaluation_curves_in_metrics_and_plots():
    from mltreelib import plots

    y_true = [0.0, 0.0, 1.0, 1.0] * 25
    scores = [0.1, 0.2, 0.8, 0.9] * 25

    metrics = Metrics()

    # ROC
    roc = metrics.roc_curve(y_true, scores)
    assert "points" in roc and "auc" in roc
    assert abs(roc["auc"] - 1.0) < 1e-7
    ax_roc = plots.plot_roc(y_true, scores)
    assert ax_roc is not None

    # Precision-Recall
    pr = metrics.pr_curve(y_true, scores)
    assert "points" in pr and "average_precision" in pr
    assert abs(pr["average_precision"] - 1.0) < 1e-7
    ax_pr = plots.plot_pr(y_true, scores)
    assert ax_pr is not None

    # Calibration
    calib = metrics.calibration(y_true, scores, n_bins=5)
    assert "bins" in calib
    ax_calib = plots.plot_calibration(y_true, scores, n_bins=5)
    assert ax_calib is not None

    # Cumulative Gain & Lift
    gain = metrics.gain_curve(y_true, scores)
    assert "points" in gain
    ax_gain = plots.plot_cumulative_gain(y_true, scores)
    assert ax_gain is not None

    lift = metrics.lift_curve(y_true, scores)
    assert "points" in lift
    ax_lift = plots.plot_lift(y_true, scores)
    assert ax_lift is not None
