"""Comprehensive tests for Spec 19: Frontier Survival Trees, AFT Leaves, Cure Models, Splines, and Calibration Metrics.

Validates:
1. Frontier Splitting Criteria:
   - gain="pseudo_value" (Jackknife pseudo-values for RMST/survival)
   - gain="rmst" (Restricted Mean Survival Time area difference)
   - gain="aft_deviance" (AFT parametric likelihood reduction)
2. Parametric AFT Leaves:
   - leaf_family="aft" with aft_dist in {"weibull", "lognormal", "loglogistic"}
   - smooth continuous survival and cumulative hazard curves
   - leaf="linear" yielding covariate-dependent smooth survival curves within leaves
3. Mixture Cure Rate Leaves:
   - leaf_family="cure"
   - survival plateau behavior: S(t) bounded below by cure fraction as t increases
4. Restricted Cubic Splines Leaves:
   - leaf_family="spline" with flexible knots
   - valid survival monotonicity S(t) in [0, 1]
5. Frontier Calibration Metrics:
   - D-Calibration (Haider et al. JMLR 2020) uniformity test
   - Integrated Calibration Index (Austin et al. 2020) ICI, E50, E90, Emax
   - 1-Calibration slope and intercept
   - Competing risks calibration (Alberge et al. 2026)
"""

import math
import pytest
from mltreelib import (
    ModelTree,
    Metrics,
    d_calibration,
    integrated_calibration_index,
    one_calibration,
    competing_risks_calibration,
)


@pytest.fixture
def synthetic_survival_data():
    """Synthetic right-censored survival dataset with planted covariate effects."""
    X = []
    durations = []
    events = []

    # 120 samples: 60 low-risk (group 0, long survival), 60 high-risk (group 1, short survival)
    for i in range(120):
        if i < 60:
            X.append([0.0, float(i % 10)])
            # Weibull / lognormal style duration around 40-80
            durations.append(40.0 + float(i) * 0.7)
            events.append(1.0 if i % 4 != 0 else 0.0)  # 75% event rate
        else:
            X.append([1.0, float(i % 10)])
            # High risk duration around 5-25
            durations.append(5.0 + float(i - 60) * 0.3)
            events.append(1.0 if i % 5 != 0 else 0.0)  # 80% event rate

    return X, durations, events


def test_frontier_splitting_criteria(synthetic_survival_data):
    X, durations, events = synthetic_survival_data
    eval_times = [10.0, 25.0, 45.0, 70.0]

    for gain_criterion in ["pseudo_value", "rmst", "aft_deviance"]:
        tree = ModelTree(max_depth=2, min_rows=10, gain=gain_criterion)
        tree.fit(X, (durations, events))

        # Check survival predictions
        surv = tree.predict(X, kind="survival", times=eval_times)
        assert len(surv) == len(X)
        for row in surv:
            assert len(row) == len(eval_times)
            # S(t) must be non-increasing
            for k in range(len(row) - 1):
                assert row[k] >= row[k + 1] - 1e-6
            assert 0.0 <= row[-1] <= 1.0

        # Check hazard predictions
        haz = tree.predict(X, kind="hazard", times=eval_times)
        assert len(haz) == len(X)
        for row in haz:
            # Lambda(t) non-decreasing
            for k in range(len(row) - 1):
                assert row[k] <= row[k + 1] + 1e-6

        # Check risk ranking: group 1 (index 90) must have higher risk than group 0 (index 20)
        risk = tree.predict(X, kind="risk")
        assert len(risk) == len(X)
        assert risk[90] > risk[20]


def test_parametric_aft_leaves(synthetic_survival_data):
    X, durations, events = synthetic_survival_data
    eval_times = [5.0, 15.0, 30.0, 50.0, 75.0, 100.0]

    for dist in ["weibull", "lognormal", "loglogistic"]:
        # Test AFT constant leaf
        tree_const = ModelTree(
            max_depth=2,
            min_rows=10,
            leaf_family="aft",
            aft_dist=dist,
            leaf="mean",
        )
        tree_const.fit(X, (durations, events))

        surv_const = tree_const.predict(X, kind="survival", times=eval_times)
        assert len(surv_const) == len(X)
        for row in surv_const:
            assert len(row) == len(eval_times)
            for k in range(len(row) - 1):
                assert row[k] >= row[k + 1] - 1e-6
            assert 0.0 <= row[-1] <= 1.0

        # Test AFT linear model tree leaf (smooth continuous variation within leaf)
        tree_linear = ModelTree(
            max_depth=2,
            min_rows=10,
            leaf_family="aft",
            aft_dist=dist,
            leaf="linear",
        )
        tree_linear.fit(X, (durations, events))

        surv_linear = tree_linear.predict(X, kind="survival", times=eval_times)
        assert len(surv_linear) == len(X)
        # Verify continuous curves vary across rows with different covariate values
        assert surv_linear[0] != surv_linear[5]

        haz_linear = tree_linear.predict(X, kind="hazard", times=eval_times)
        assert len(haz_linear) == len(X)
        for row in haz_linear:
            for k in range(len(row) - 1):
                assert row[k] <= row[k + 1] + 1e-6


def test_mixture_cure_rate_leaves(synthetic_survival_data):
    X, durations, events = synthetic_survival_data
    # In cure models, long-term survival plateaus at the cure fraction pi
    cure_frac = 0.25
    tree = ModelTree(
        max_depth=2,
        min_rows=10,
        leaf_family="cure",
        cure_fraction=cure_frac,
    )
    tree.fit(X, (durations, events))

    long_times = [10.0, 50.0, 200.0, 500.0, 1000.0]
    surv = tree.predict(X, kind="survival", times=long_times)

    for row in surv:
        # Monotonically non-increasing
        for k in range(len(row) - 1):
            assert row[k] >= row[k + 1] - 1e-6
        # Plateau: long-term survival S(t) does not drop below cure_fraction
        assert row[-1] >= cure_frac - 1e-3


def test_spline_survival_leaves(synthetic_survival_data):
    X, durations, events = synthetic_survival_data
    tree = ModelTree(
        max_depth=2,
        min_rows=10,
        leaf_family="spline",
        spline_knots=4,
    )
    tree.fit(X, (durations, events))

    eval_times = [5.0, 15.0, 35.0, 60.0, 90.0]
    surv = tree.predict(X, kind="survival", times=eval_times)
    assert len(surv) == len(X)
    for row in surv:
        assert len(row) == len(eval_times)
        for k in range(len(row) - 1):
            assert row[k] >= row[k + 1] - 1e-6
        assert 0.0 <= row[-1] <= 1.0


def test_frontier_calibration_metrics(synthetic_survival_data):
    X, durations, events = synthetic_survival_data

    # Fit a model to obtain survival predictions
    tree = ModelTree(max_depth=2, min_rows=10, leaf_family="aft", aft_dist="weibull")
    tree.fit(X, (durations, events))

    horizon = 30.0
    surv_at_horizon = [row[0] for row in tree.predict(X, kind="survival", times=[horizon])]

    # 1. D-Calibration test
    d_rep = d_calibration(durations, events, surv_at_horizon, n_bins=5)
    assert "chi2_stat" in d_rep
    assert "p_value" in d_rep
    assert "is_calibrated" in d_rep
    assert len(d_rep["bin_counts"]) == 5
    assert d_rep["n_samples"] == len(durations)

    # Via Metrics class method
    metrics = Metrics()
    d_rep_m = metrics.d_calibration(durations, events, surv_at_horizon, n_bins=5)
    assert d_rep_m["chi2_stat"] == d_rep["chi2_stat"]

    # 2. Integrated Calibration Index (ICI)
    ici_rep = integrated_calibration_index(durations, events, surv_at_horizon, horizon=horizon)
    assert "ici" in ici_rep
    assert "e50" in ici_rep
    assert "e90" in ici_rep
    assert "emax" in ici_rep
    assert ici_rep["ici"] >= 0.0
    assert ici_rep["emax"] >= ici_rep["e90"] >= ici_rep["e50"]

    # Alias check via Metrics
    ici_rep_m = metrics.ici(durations, events, surv_at_horizon, horizon=horizon)
    assert ici_rep_m["ici"] == ici_rep["ici"]

    # 3. 1-Calibration slope and intercept
    one_rep = one_calibration(durations, events, surv_at_horizon, horizon=horizon)
    assert "slope" in one_rep
    assert "intercept" in one_rep
    assert math.isfinite(one_rep["slope"])
    assert math.isfinite(one_rep["intercept"])

    # 4. Competing Risks Calibration
    cif_preds = {
        1: [0.3 * (1.0 - s) for s in surv_at_horizon],
        2: [0.7 * (1.0 - s) for s in surv_at_horizon],
    }
    cr_rep = competing_risks_calibration(durations, events, cif_preds, horizon=horizon)
    assert "cause_ici" in cr_rep
    assert "cause_brier" in cr_rep
    assert 1 in cr_rep["cause_ici"]
    assert 2 in cr_rep["cause_ici"]
    assert cr_rep["cause_ici"][1] >= 0.0
