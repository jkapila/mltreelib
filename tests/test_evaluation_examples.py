"""Check the math used by the survival and uplift evaluation examples."""
from pathlib import Path
from runpy import run_path
from math import isclose

example = run_path(str(Path(__file__).parents[1] / "examples/survival_uplift_evaluation.py"))
ipcw_brier = example["ipcw_brier"]
policy_value = example["policy_value"]


def test_brier_ignores_censoring_before_horizon_and_weights_events():
    # An event at t=1, a censored case at t=1, and a survivor at t=3.
    result = ipcw_brier([1., 1., 3.], [1., 0., 0.], [.5, .2, .8], 2., lambda _: .5)
    assert isclose(result, (.5 ** 2 / .5 + .2 ** 2 / .5) / 3)


def test_randomized_policy_value_scales_effect_and_treatment_cost():
    # Two selected people differ by 4 units. The other two are not selected.
    value, selected, treated, control = policy_value([2., 6., 100., -100.], [0., 1., 0., 1.], [True, True, False, False], 1.)
    assert (value, selected, treated, control) == (1.5, 2, 1, 1)
    assert policy_value([2., 6.], [0., 1.], [False, False], 1.) == (0., 0, 0, 0)
