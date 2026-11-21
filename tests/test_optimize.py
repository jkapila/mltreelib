"""Behavior tests for the generic optimizer Python surface."""

import math

import pytest
from mltreelib.solvers import OptimizationResult, optimize


def sphere(point):
    return sum(value * value for value in point)


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
