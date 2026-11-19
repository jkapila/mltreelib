import pytest
from mltreelib import Metrics

def test_metrics_uplift_report():
    m = Metrics()
    # 10 units: 5 treated, 5 control
    # When treated, outcomes are higher for high predicted uplift
    y_true = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    treatment = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    tau_pred = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]

    rep = m.uplift(y_true, treatment, tau_pred)
    assert rep["n_total"] == 10
    assert rep["n_treated"] == 5
    assert rep["n_control"] == 5
    assert "ate" in rep
    assert "qini_coef" in rep
    assert "qini_norm" in rep
    assert "auuc" in rep
    assert "top_decile_uplift" in rep
    assert "top_quintile_uplift" in rep
    assert len(rep["qini_curve"]) == 11
    assert len(rep["cumulative_gain"]) == 11

def test_metrics_survival_report():
    m = Metrics()
    durations = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    events = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    # Higher risk score -> fails earlier (perfect concordance)
    risk_scores = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    rep = m.survival(durations, events, risk_scores)
    assert rep["n_samples"] == 8
    assert rep["n_events"] == 5
    assert rep["harrell_c"] >= 0.8
    assert rep["uno_c"] >= 0.8
    assert "ibs" in rep
    assert len(rep["brier_at_horizons"]) > 0

def test_metrics_time_series_report():
    m = Metrics()
    y_train = [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]
    y_true = [14.0, 16.0, 15.0, 17.0]
    y_pred = [14.2, 15.8, 15.1, 16.9]
    std_pred = [0.5, 0.5, 0.5, 0.5]

    rep = m.time_series(y_true, y_pred, y_train=y_train, seasonal_period=1, std_pred=std_pred)
    assert rep["n_periods"] == 4
    assert rep["mae"] < 0.3
    assert rep["rmse"] < 0.3
    assert rep["mase"] is not None
    assert rep["mase"] < 0.5
    assert rep["wape"] < 0.05
    assert rep["smape"] < 5.0
    assert rep["mda"] == 1.0
    assert rep["crps"] is not None
    assert rep["crps"] > 0.0
