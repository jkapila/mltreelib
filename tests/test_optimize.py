"""Behavior tests for the generic optimizer Python surface.

One file holds the whole surface: the shared contracts first, then the real
data checks, then one section per method.
"""

import math
from math import sqrt

import pytest
from mltreelib.datasets import load_dataset
from mltreelib.solvers import OptimizationResult, optimize


def sphere(point):
    return sum(value * value for value in point)


# -- Shared surface -------------------------------------------------------


def test_optimize_returns_a_typed_result():
    result = optimize(sphere, [(-2.0, 2.0), (-2.0, 2.0)], samples=200, seed=3)
    assert isinstance(result, OptimizationResult)
    assert result.status == "converged"
    assert result.converged is True
    assert result.evaluations == 200
    assert len(result.best_x) == 2
    assert result.best_value == pytest.approx(sphere(result.best_x))
    assert result.history is None
    assert result.max_constraint_violation is None


def test_optimize_keeps_candidates_inside_bounds():
    result = optimize(sphere, [(-1.0, 1.0), (0.0, 3.0)], samples=100, seed=5)
    assert all(-1.0 <= value <= 1.0 for value in result.best_x[:1])
    assert all(0.0 <= value <= 3.0 for value in result.best_x[1:])
    assert math.isfinite(result.best_value)


def test_optimize_is_reproducible_for_one_seed():
    first = optimize(sphere, [(-2.0, 2.0)], samples=50, seed=11)
    second = optimize(sphere, [(-2.0, 2.0)], samples=50, seed=11)
    third = optimize(sphere, [(-2.0, 2.0)], samples=50, seed=12)
    assert first.best_x == second.best_x
    assert first.best_value == second.best_value
    assert first.best_x != third.best_x


def test_optimize_reports_the_evaluation_budget():
    result = optimize(sphere, [(-2.0, 2.0)], samples=100, max_evaluations=25, seed=1)
    assert result.status == "evaluation_limit"
    assert result.converged is False
    assert result.evaluations == 25


def test_latin_hypercube_is_reproducible():
    first = optimize(
        sphere, [(-1.0, 1.0)], method="latin_hypercube", samples=30, seed=4
    )
    second = optimize(
        sphere, [(-1.0, 1.0)], method="latin_hypercube", samples=30, seed=4
    )
    assert first.best_x == second.best_x
    assert first.evaluations == 30


def test_optimize_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        optimize(sphere, [(-1.0, 1.0)], method="nope")


def test_optimize_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        optimize(sphere, [(1.0, -1.0)])


def test_optimize_surfaces_callback_failures():
    def broken(point):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        optimize(broken, [(-1.0, 1.0)], samples=5)


def test_optimize_reports_non_finite_objective_values():
    def broken(point):
        return float("nan")

    result = optimize(broken, [(-1.0, 1.0)], samples=5)
    assert result.status == "invalid_objective"
    assert result.converged is False


# -- Real data ------------------------------------------------------------


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


# -- Methods --------------------------------------------------------------


def quadratic(point):
    """Weighted quadratic with its minimum at (0.3, -1.5)."""
    return (point[0] - 0.3) ** 2 + 2.0 * (point[1] + 1.5) ** 2


def gradient(point):
    """Exact gradient of the quadratic objective."""
    return [2.0 * (point[0] - 0.3), 4.0 * (point[1] + 1.5)]


BOUNDS = [(-2.0, 2.0), (-2.0, 2.0)]


def run(method, **kwargs):
    """Run one method on the quadratic objective."""
    return optimize(quadratic, BOUNDS, method=method, **kwargs)


@pytest.mark.parametrize("method", ["random_search", "latin_hypercube"])
def test_sampling_methods_find_the_optimum(method):
    result = run(method, samples=2000, seed=5)
    assert result.status == "converged"
    assert result.best_value < 0.2, f"value was {result.best_value}"


def test_lbfgs_b_finds_the_optimum_with_a_gradient():
    result = run("lbfgs_b", start=[1.5, 1.5], gradient=gradient, max_evaluations=200)
    assert result.status == "converged"
    assert abs(result.best_x[0] - 0.3) < 1e-3, f"point was {result.best_x}"
    assert abs(result.best_x[1] + 1.5) < 1e-3, f"point was {result.best_x}"


def test_lbfgs_b_falls_back_to_finite_differences():
    result = run("lbfgs_b", start=[1.5, 1.5], max_evaluations=400)
    assert result.status == "converged"
    assert abs(result.best_x[1] + 1.5) < 1e-2, f"point was {result.best_x}"


@pytest.mark.parametrize("method", ["nelder_mead", "powell"])
def test_derivative_free_methods_find_the_optimum(method):
    result = run(method, start=[1.5, 1.5], max_evaluations=5000)
    assert result.status == "converged"
    assert abs(result.best_x[0] - 0.3) < 1e-2, f"point was {result.best_x}"
    assert abs(result.best_x[1] + 1.5) < 1e-2, f"point was {result.best_x}"


def test_simulated_annealing_finds_the_optimum():
    result = run("simulated_annealing", start=[0.0, 0.0], max_evaluations=5000, seed=3)
    assert result.status in {"converged", "evaluation_limit"}
    assert result.best_value < 1.0, f"value was {result.best_value}"


def test_basin_hopping_finds_the_optimum():
    result = run("basin_hopping", start=[1.5, 1.5], max_evaluations=5000, seed=3)
    assert result.status in {"converged", "evaluation_limit"}
    assert result.best_value < 0.5, f"value was {result.best_value}"


def test_unbounded_methods_reject_box_bounds():
    for method in ("bfgs", "nonlinear_cg"):
        result = optimize(quadratic, [(-10.0, 10.0), (-10.0, 10.0)], method=method)
        assert result.status == "unsupported"


def test_bfgs_and_conjugate_gradient_find_the_unbounded_optimum():
    bounds = [(float("-inf"), float("inf")), (float("-inf"), float("inf"))]
    for method in ("bfgs", "nonlinear_cg"):
        result = optimize(
            quadratic,
            bounds,
            method=method,
            start=[1.0, 1.0],
            gradient=gradient,
            max_evaluations=500,
        )
        assert result.status == "converged", f"{method} gave {result.status}"
        assert abs(result.best_x[0] - 0.3) < 1e-3, f"point was {result.best_x}"
        assert abs(result.best_x[1] + 1.5) < 1e-3, f"point was {result.best_x}"


# -- COBYLA ---------------------------------------------------------------


def constrained_objective(point):
    """Weighted quadratic with its minimum at (0.3, -1.5)."""
    return (point[0] - 0.3) ** 2 + 2.0 * (point[1] + 1.5) ** 2


def sum_constraint(point):
    """Inequality constraint that keeps the sum of the parameters under -1.5."""
    return [-1.5 - point[0] - point[1]]


def run_cobyla(constraint=sum_constraint, **kwargs):
    """Run COBYLA on the constrained quadratic."""
    return optimize(
        constrained_objective,
        [(-5.0, 5.0), (-5.0, 5.0)],
        method="cobyla",
        start=[-1.0, -1.0],
        constraints=constraint,
        max_evaluations=2000,
        **kwargs,
    )


def test_cobyla_solves_a_constrained_problem():
    result = run_cobyla()
    assert result.status == "converged", f"status was {result.status}"
    assert result.best_value < 0.0601, f"value was {result.best_value}"
    assert abs(result.best_x[0] - 0.1) < 1e-2, f"point was {result.best_x}"
    assert abs(result.best_x[1] + 1.6) < 1e-2, f"point was {result.best_x}"


def test_cobyla_reports_the_constraint_violation():
    result = run_cobyla()
    assert result.max_constraint_violation is not None
    assert result.max_constraint_violation <= 1e-6, (
        f"violation was {result.max_constraint_violation}"
    )


def test_cobyla_rejects_equality_constraints():
    result = run_cobyla(constraint_kind="equality")
    assert result.status == "unsupported"
    assert "equality" in (result.message or "")


def test_cobyla_reports_an_infeasible_problem():
    result = run_cobyla(constraint=lambda point: [-50.0 - point[0] - point[1]])
    assert result.status == "infeasible"
    assert result.max_constraint_violation > 0.0


def test_cobyla_counts_its_evaluations():
    result = run_cobyla()
    assert result.evaluations > 0
    assert result.evaluations <= 2000
    assert result.iterations > 0


def test_cobyla_preserves_the_incumbent_when_initialization_spends_the_budget():
    result = optimize(
        sphere,
        [(-1.0, 1.0)],
        method="cobyla",
        start=[0.5],
        constraints=lambda point: [point[0]],
        max_evaluations=1,
    )
    assert result.status == "evaluation_limit"
    assert result.evaluations == 1
    assert result.best_x == [0.5]
    assert result.best_value == 0.25
    assert result.max_constraint_violation == 0.0


@pytest.mark.parametrize(
    ("constraint", "message"),
    [
        (lambda point: [0.0] if point[0] == -1.0 else [0.0, 0.0], "number of constraints"),
        (lambda _point: [float("nan")], "non-finite"),
    ],
)
def test_cobyla_rejects_invalid_constraint_values(constraint, message):
    result = run_cobyla(constraint)
    assert result.status == "hard_error"
    assert message in result.message
    assert result.evaluations > 0


def test_unconstrained_method_rejects_constraints_before_evaluating():
    def unexpected(_point):
        pytest.fail("unsupported requests must not evaluate callbacks")

    result = optimize(unexpected, [(-1.0, 1.0)], constraints=unexpected)
    assert result.status == "unsupported"
    assert result.evaluations == 0


@pytest.mark.parametrize("method", ["cmaes", "shade", "nes", "ga"])
def test_evolutionary_methods_optimize_the_user_objective(method):
    values = []

    def objective(point):
        assert all(-2.0 <= value <= 2.0 for value in point)
        values.append(quadratic(point))
        return values[-1]

    result = optimize(
        objective, BOUNDS, method=method, start=[1.0, 1.0], max_evaluations=3000
    )
    assert result.evaluations == len(values)
    assert result.best_value == min(values)
    assert result.best_value == pytest.approx(quadratic(result.best_x))
    assert result.best_value < 0.3


@pytest.mark.parametrize("local_method", ["lbfgs_b", "bfgs", "nonlinear_cg", "nelder_mead", "powell"])
def test_basin_hopping_selects_a_local_solver(local_method):
    bounds = (
        [(-math.inf, math.inf)] * 2
        if local_method in {"bfgs", "nonlinear_cg"}
        else BOUNDS
    )
    result = optimize(
        quadratic,
        bounds,
        method="basin_hopping",
        local_method=local_method,
        start=[1.0, 1.0],
        max_evaluations=250,
    )
    assert result.best_value < 1e-4


def test_lbfgs_b_keeps_the_evaluated_start_when_the_gradient_exceeds_the_budget():
    result = optimize(sphere, [(-1.0, 1.0)], method="lbfgs_b", start=[0.5], max_evaluations=1)
    assert result.status == "evaluation_limit"
    assert result.evaluations == 1
    assert result.best_x == [0.5]
    assert result.best_value == 0.25
