"""Real-data checks for the generic optimizer surface.

The TabArena Parquet files are not tracked in git. These tests skip when the
suite is not downloaded locally.
"""

from math import sqrt

import pytest
from mltreelib.datasets import load_dataset
from mltreelib.solvers import optimize


def load_airfoil():
    """Read the airfoil self-noise table as one numeric feature and one target.

    The dataset is read from the repository folder or the local cache. The test
    skips when the suite is not available, because the files are not tracked in
    git and a test run never downloads them.
    """
    try:
        data = load_dataset("airfoil_self_noise", auto_download=False)
    except (ImportError, ValueError) as error:
        pytest.skip(f"airfoil dataset is not available locally: {error}")
    columns = data["columns"]
    rows = [
        (float(x), float(y))
        for x, y in zip(columns["frequency"], columns["scaled-sound-pressure"])
    ]
    if not rows:
        pytest.skip("airfoil dataset has no rows")
    return rows


def standardize(values):
    """Return values shifted to zero mean and scaled to unit standard deviation."""
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    spread = sqrt(variance)
    return [(value - mean) / spread for value in values]


def ordinary_least_squares(x, y):
    """Fit an intercept and slope to standardized data."""
    count = len(x)
    mean_x = sum(x) / count
    mean_y = sum(y) / count
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    variance = sum((xi - mean_x) ** 2 for xi in x)
    slope = covariance / variance
    return mean_y - slope * mean_x, slope


def sum_of_squares(x, y, intercept, slope):
    """Residual sum of squares for one intercept and slope."""
    return sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))


def test_random_search_matches_the_least_squares_optimum_on_real_data():
    rows = load_airfoil()
    x = standardize([row[0] for row in rows])
    y = standardize([row[1] for row in rows])
    best_intercept, best_slope = ordinary_least_squares(x, y)
    optimum = sum_of_squares(x, y, best_intercept, best_slope)

    def objective(parameters):
        return sum_of_squares(x, y, parameters[0], parameters[1])

    result = optimize(
        objective,
        [(-5.0, 5.0), (-5.0, 5.0)],
        method="random_search",
        samples=2000,
        seed=17,
    )

    assert result.status == "converged"
    assert result.evaluations == 2000
    assert result.best_value <= optimum * 1.01
    assert abs(result.best_x[0] - best_intercept) < 0.1
    assert abs(result.best_x[1] - best_slope) < 0.1


def test_latin_hypercube_matches_the_least_squares_optimum_on_real_data():
    rows = load_airfoil()
    x = standardize([row[0] for row in rows])
    y = standardize([row[1] for row in rows])
    best_intercept, best_slope = ordinary_least_squares(x, y)
    optimum = sum_of_squares(x, y, best_intercept, best_slope)

    def objective(parameters):
        return sum_of_squares(x, y, parameters[0], parameters[1])

    result = optimize(
        objective,
        [(-5.0, 5.0), (-5.0, 5.0)],
        method="latin_hypercube",
        samples=2000,
        seed=17,
    )

    assert result.status == "converged"
    assert result.evaluations == 2000
    # Stratified sampling pairs strata across dimensions, so the nearest joint
    # sample sits a little further from the optimum than uniform draws do.
    assert result.best_value <= optimum * 1.05
    assert abs(result.best_x[1] - best_slope) < 0.25
