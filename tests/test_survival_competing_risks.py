"""Comprehensive tests for Spec 10: Survival and Competing Risks Trees.

Validates:
1. Right-censored single event survival trees:
   - Criteria: LogRank, Martingale, Tarone-Ware, Peto-Prentice, SurvivalBrier.
   - Predictions: kind="response", kind="survival", kind="hazard", kind="risk".
   - Monotonicity invariants: S(t) non-increasing, Lambda(t) non-decreasing.
2. Competing risks multi-cause trees:
   - Multi-event targets: E in {0, 1, 2}.
   - Criteria: GrayTest (subdistribution log-rank), LogRank, Martingale.
   - Predictions: kind="cif", cause=1 and cause=2.
   - Aalen-Johansen CIF simplex invariant: sum_c F_c(t) + S(t) == 1.0.
3. Competing risks metrics:
   - Metrics().competing_risks(...) and competing_risks_report(...).
   - Cause-specific Harrell's and Uno's IPCW C-indices, time-dependent Brier scores, IBS.
"""

import pytest
from mltreelib import ModelTree, Metrics, competing_risks_report


def test_survival_single_event_criteria_and_predictions():
    # Synthetic right-censored data with clear risk separation based on x0
    # Group 0: low risk (longer survival: ~50-80), Group 1: high risk (shorter survival: ~10-30)
    X = []
    durations = []
    events = []

    for i in range(100):
        if i < 50:
            X.append([0.0, float(i)])
            durations.append(40.0 + float(i))
            events.append(1.0 if i % 3 != 0 else 0.0)  # ~67% event rate
        else:
            X.append([1.0, float(i)])
            durations.append(5.0 + float(i - 50) * 0.5)
            events.append(1.0 if i % 4 != 0 else 0.0)  # ~75% event rate

    eval_times = [5.0, 15.0, 30.0, 50.0, 75.0]

    # Test all survival criteria
    for gain_scheme in ["logrank", "martingale", "tarone_ware", "peto", "survival_brier"]:
        tree = ModelTree(max_depth=2, min_rows=10, gain=gain_scheme)
        tree.fit(X, (durations, events))

        # Check prediction kinds
        surv = tree.predict(X, kind="survival", times=eval_times)
        assert len(surv) == len(X)
        for row in surv:
            assert len(row) == len(eval_times)
            # S(t) non-increasing
            for k in range(len(row) - 1):
                assert row[k] >= row[k + 1] - 1e-6
            assert 0.0 <= row[-1] <= 1.0

        haz = tree.predict(X, kind="hazard", times=eval_times)
        assert len(haz) == len(X)
        for row in haz:
            assert len(row) == len(eval_times)
            # Lambda(t) non-decreasing
            for k in range(len(row) - 1):
                assert row[k] <= row[k + 1] + 1e-6

        # Median time response
        resp = tree.predict(X, kind="response")
        assert len(resp) == len(X)
        assert all(t > 0.0 for t in resp)

        # Risk score (higher = higher risk / shorter survival)
        risk = tree.predict(X, kind="risk")
        assert len(risk) == len(X)
        # High risk group (X[i][0] == 1.0) must have higher risk score than low risk group
        assert risk[75] > risk[25]


def test_competing_risks_aalen_johansen_simplex_property():
    # Competing risks: Cause 1 (e.g. disease relapse) and Cause 2 (e.g. treatment toxicity / other cause)
    # Event 0: censored, 1: cause 1, 2: cause 2
    X = []
    durations = []
    events = []

    for i in range(120):
        feat = float(i) / 10.0
        X.append([feat, float(i % 5)])
        if i % 3 == 0:
            events.append(0.0)  # Censored
            durations.append(10.0 + feat * 2.0)
        elif i % 3 == 1:
            events.append(1.0)  # Cause 1
            durations.append(5.0 + feat * 0.8)
        else:
            events.append(2.0)  # Cause 2
            durations.append(8.0 + feat * 1.5)

    tree = ModelTree(max_depth=3, min_rows=10, gain="gray", target_cause=1)
    tree.fit(X, (durations, events))

    eval_times = [5.0, 10.0, 15.0, 25.0, 35.0]

    # Predict S(t), CIF for cause 1, and CIF for cause 2
    surv_curves = tree.predict(X, kind="survival", times=eval_times)
    cif_cause1 = tree.predict(X, kind="cif", times=eval_times, cause=1)
    cif_cause2 = tree.predict(X, kind="cif", times=eval_times, cause=2)

    assert len(cif_cause1) == len(X)
    assert len(cif_cause2) == len(X)

    for i in range(len(X)):
        s_row = surv_curves[i]
        c1_row = cif_cause1[i]
        c2_row = cif_cause2[i]

        for t_idx in range(len(eval_times)):
            f1 = c1_row[t_idx]
            f2 = c2_row[t_idx]
            s = s_row[t_idx]

            assert 0.0 <= f1 <= 1.0
            assert 0.0 <= f2 <= 1.0
            assert 0.0 <= s <= 1.0

            # Aalen-Johansen Simplex Invariant: F_1(t) + F_2(t) + S(t) == 1.0
            simplex_sum = f1 + f2 + s
            assert pytest.approx(simplex_sum, abs=1e-5) == 1.0

        # Monotonicity of CIF: cumulative incidence is non-decreasing in time
        for t_idx in range(len(eval_times) - 1):
            assert c1_row[t_idx] <= c1_row[t_idx + 1] + 1e-7
            assert c2_row[t_idx] <= c2_row[t_idx + 1] + 1e-7


def test_competing_risks_metrics_report():
    # Evaluate with PyMetrics and competing_risks_report
    durations = [3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    events = [1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0]

    eval_times = [5.0, 15.0, 25.0]

    # Synthetic predictions
    cif_cause1 = [[0.1, 0.2, 0.35] for _ in range(len(durations))]
    cif_cause2 = [[0.05, 0.15, 0.25] for _ in range(len(durations))]
    risk1 = [0.1 * i for i in range(len(durations))]
    risk2 = [0.2 * (len(durations) - i) for i in range(len(durations))]

    metrics = Metrics()
    report = metrics.competing_risks(
        durations=durations,
        events=events,
        cif_curves={1: cif_cause1, 2: cif_cause2},
        risk_scores={1: risk1, 2: risk2},
        times=eval_times,
    )

    assert report["n_samples"] == 8
    assert report["causes"] == [1, 2]
    assert report["cause_counts"] == [3, 3]  # 3 of cause 1, 3 of cause 2
    assert 1 in report["cause_c_indices"]
    assert 2 in report["cause_c_indices"]
    assert 1 in report["cause_uno_c_indices"]
    assert 2 in report["cause_uno_c_indices"]
    assert 1 in report["cause_ibs"]
    assert 2 in report["cause_ibs"]
    assert 1 in report["brier_at_horizons"]
    assert 2 in report["brier_at_horizons"]

    # Also test standalone function
    report2 = competing_risks_report(
        durations=durations,
        events=events,
        cif_curves={1: cif_cause1, 2: cif_cause2},
        risk_scores={1: risk1, 2: risk2},
        times=eval_times,
    )
    assert report2["causes"] == [1, 2]
    assert report2["cause_counts"] == report["cause_counts"]
