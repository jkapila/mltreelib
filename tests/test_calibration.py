from random import Random

import pytest
from mltreelib import Metrics
from mltreelib.calibration import IsotonicCalibrator


def test_isotonic_reduces_brier_score_on_independent_predictions():
    def sample(seed):
        rng = Random(seed)
        probabilities = [rng.choice((0.05, 0.15, 0.85, 0.95)) for _ in range(2_000)]
        events = [float(rng.random() < 0.2 + 0.6 * p) for p in probabilities]
        return events, probabilities

    y_cal, p_cal = sample(42)
    y_test, p_test = sample(43)
    calibrated = IsotonicCalibrator.fit(y_cal, p_cal).predict(p_test)
    metrics = Metrics()
    assert metrics.brier_score(y_test, calibrated) < metrics.brier_score(y_test, p_test)


def test_calibrator_checks_probability_range():
    model = IsotonicCalibrator.fit([0, 1], [0.25, 0.75])
    assert model.predict([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="probabilities"):
        model.predict([float("nan")])
