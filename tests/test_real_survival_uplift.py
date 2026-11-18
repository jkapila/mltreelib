"""Check censoring weights and randomized policy arithmetic."""
from math import isclose
from pathlib import Path
from runpy import run_path

example = run_path(str(Path(__file__).parents[1] / "examples/real_survival_uplift.py"))


def test_censoring_km_and_brier_ties():
    g = example["censor_km"]([1., 2., 2., 3.], [0., 1., 0., 1.])
    assert isclose(g(1., left=True), 1.)
    assert isclose(g(1.), .75)
    assert isclose(g(2., left=True), .75)
    assert isclose(g(2.), .5)
    score = example["brier"]([2., 3.], [1., 0.], [.5, .8], 2., g)
    assert isclose(score, (.25 / .75 + .04 / .5) / 2)


def test_policy_value_on_randomized_groups():
    value, n, nt, nc, interval = example["group_value"]([0., 1., 0., 1., 0., 1., 0., 1.], [0., 1., 0., 1., 0., 1., 0., 1.], [True, True, True, True, False, False, False, False])
    assert (value, n, nt, nc, interval) == (0.5, 4, 2, 2, (0.5, 0.5))
