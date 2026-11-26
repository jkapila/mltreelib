"""Tests for mltreelib.data.scaling (log1p, power, mean/median difference, Box-Cox, Yeo-Johnson)."""

from __future__ import annotations

import copy
import math

import pytest
from mltreelib.data.scaling import (
    FeatureTransformer,
    GroupedFeatureTransformer,
    boxcox_lambda,
    log1p,
    yeojohnson_lambda,
)

NAN = math.nan
INF = math.inf
ALL_KINDS = ["log1p", "power", "mean_diff", "median_diff", "boxcox", "yeojohnson"]


def _same(a, b) -> bool:
    """NaN-safe equality for lists of plain values."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None or y is None:
            if x is not y:
                return False
        elif (
            isinstance(x, float)
            and isinstance(y, float)
            and math.isnan(x)
            and math.isnan(y)
        ):
            continue
        elif x != y:
            return False
    return True


def _positive_data() -> list[float]:
    return [0.5, 1.0, 2.0, 4.0, 8.0, 0.25]


def _mixed_data() -> list[float]:
    return [-3.0, -1.0, -0.2, 0.0, 0.5, 2.0, 7.0, -0.7, 1.5]


def _fit_data_for(kind: str) -> list[float]:
    return _positive_data() if kind in ("boxcox", "log1p", "power") else _mixed_data()


def _loglik_boxcox(xs: list[float], lam: float) -> float:
    """Independent Box-Cox profile log-likelihood (plain formulas, population variance)."""
    ys = [math.log(x) if abs(lam) < 1e-12 else (x**lam - 1.0) / lam for x in xs]
    n = len(xs)
    m = sum(ys) / n
    var = sum((y - m) ** 2 for y in ys) / n
    return -0.5 * n * math.log(var) + (lam - 1.0) * sum(math.log(x) for x in xs)


def _loglik_yeojohnson(xs: list[float], lam: float) -> float:
    """Independent Yeo-Johnson profile log-likelihood (plain formulas)."""

    def psi(x: float) -> float:
        if x >= 0:
            return math.log(x + 1) if abs(lam) < 1e-12 else ((x + 1) ** lam - 1) / lam
        if abs(lam - 2) < 1e-12:
            return -math.log(1 - x)
        return -((1 - x) ** (2 - lam) - 1) / (2 - lam)

    ys = [psi(x) for x in xs]
    n = len(xs)
    m = sum(ys) / n
    var = sum((y - m) ** 2 for y in ys) / n
    jac = sum((1 if x >= 0 else -1) * math.log(abs(x) + 1) for x in xs)
    return -0.5 * n * math.log(var) + (lam - 1.0) * jac


def _grid_argmax(
    loglik, lo: float = -5.0, hi: float = 5.0, step: float = 0.01
) -> float:
    k = round((hi - lo) / step)
    best_lam, best_ll = lo, -math.inf
    for i in range(k + 1):
        lam = lo + i * step
        ll = loglik(lam)
        if ll > best_ll:
            best_lam, best_ll = lam, ll
    return best_lam


# ---------- closed-form checks ----------


def test_log1p_closed_form_and_inverse():
    t = FeatureTransformer("log1p")
    out = t.fit_transform([0.0, 1.0, 3.0])
    assert math.isclose(out[0], 0.0, abs_tol=1e-15)
    assert math.isclose(out[1], math.log(2.0), rel_tol=1e-15)
    assert math.isclose(out[2], math.log(4.0), rel_tol=1e-15)
    back = t.inverse_transform(out)
    assert all(
        math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-15)
        for a, b in zip(back, [0.0, 1.0, 3.0])
    )
    assert t.n_fit_ == 3 and t.n_missing_ == 0 and t.n_inf_ == 0


def test_log1p_function_convenience():
    assert log1p([0.0, None]) == [0.0, None]
    assert math.isclose(log1p([3.0])[0], math.log(4.0))
    values = [0.0, None, NAN, INF]
    snapshot = copy.deepcopy(values)
    out = log1p(values)
    assert out[1] is None and math.isnan(out[2]) and out[3] == INF
    assert _same(values, snapshot)
    assert log1p([]) == []


def test_power_closed_form_and_inverse():
    sq = FeatureTransformer("power", p=2.0).fit([0.0, 1.0])
    assert sq.transform([0.0, 1.0, 3.0]) == [0.0, 1.0, 9.0]
    assert sq.inverse_transform([9.0]) == [3.0]

    half = FeatureTransformer("power", p=0.5)
    assert half.fit_transform([4.0, 9.0]) == [2.0, 3.0]
    assert half.inverse_transform([2.0, 3.0]) == [4.0, 9.0]


def test_power_odd_integer_allows_negatives_and_even_rejects_them():
    cube = FeatureTransformer("power", p=3.0)
    assert cube.fit_transform([-2.0, 2.0]) == [-8.0, 8.0]
    assert cube.inverse_transform([-8.0, 8.0]) == [-2.0, 2.0]

    with pytest.raises(ValueError, match="index 0"):
        FeatureTransformer("power", p=2.0).fit([-2.0, 1.0])
    with pytest.raises(ValueError):
        FeatureTransformer("power", p=0.5).fit([-1.0])


def test_power_rejects_zero_exponent():
    with pytest.raises(ValueError):
        FeatureTransformer("power", p=0.0)


def test_mean_diff_closed_form_counts_and_inverse():
    t = FeatureTransformer("mean_diff").fit([1.0, 2.0, 3.0, None, NAN, INF])
    assert t.params_ == {"mean": 2.0}
    assert (t.n_fit_, t.n_missing_, t.n_inf_) == (3, 2, 1)
    assert t.transform([4.0, None]) == [2.0, None]
    assert t.inverse_transform([2.0]) == [4.0]


def test_median_diff_closed_form():
    t = FeatureTransformer("median_diff").fit([1.0, 2.0, 10.0, None])
    assert t.params_ == {"median": 2.0}
    assert t.transform([5.0]) == [3.0]
    assert t.inverse_transform([3.0]) == [5.0]


def test_boxcox_fixed_lambda_closed_forms():
    assert FeatureTransformer("boxcox", lmbda=0.0).fit([1.0]).transform([math.e]) == [
        pytest.approx(1.0, rel=1e-15)
    ]
    sq = FeatureTransformer("boxcox", lmbda=2.0).fit([1.0])
    assert sq.transform([3.0]) == [pytest.approx(4.0)]  # (3^2 - 1) / 2
    half = FeatureTransformer("boxcox", lmbda=0.5).fit([1.0])
    assert half.transform([4.0]) == [pytest.approx(2.0)]  # (sqrt(4) - 1) / 0.5


def test_yeojohnson_fixed_lambda_closed_forms():
    zero = FeatureTransformer("yeojohnson", lmbda=0.0).fit([1.0])
    assert zero.transform([1.0]) == [pytest.approx(math.log(2.0))]
    assert zero.transform([-1.0]) == [pytest.approx(-1.5)]  # -((1+1)^2 - 1) / 2
    two = FeatureTransformer("yeojohnson", lmbda=2.0).fit([1.0])
    assert two.transform([-1.0]) == [pytest.approx(-math.log(2.0))]


# ---------- lambda fitting ----------


def test_boxcox_lambda_matches_bruteforce_grid_on_exponential_like_data():
    n = 100
    xs = [-math.log(1.0 - (i + 0.5) / n) for i in range(n)]  # exponential quantiles
    lam_fit = boxcox_lambda(xs)
    lam_grid = _grid_argmax(lambda lam: _loglik_boxcox(xs, lam))
    assert abs(lam_fit - lam_grid) <= 0.01 + 1e-9
    assert _loglik_boxcox(xs, lam_fit) >= _loglik_boxcox(xs, lam_grid) - 1e-9


def test_yeojohnson_lambda_matches_bruteforce_grid_on_mixed_sign_data():
    xs = [-3.0 + 0.1 * i for i in range(60)] + [0.5 * i for i in range(1, 40)]
    lam_fit = yeojohnson_lambda(xs)
    lam_grid = _grid_argmax(lambda lam: _loglik_yeojohnson(xs, lam))
    assert abs(lam_fit - lam_grid) <= 0.01 + 1e-9
    assert _loglik_yeojohnson(xs, lam_fit) >= _loglik_yeojohnson(xs, lam_grid) - 1e-9


def test_lambda_fit_handles_missing_and_empty_without_raising():
    xs = [1.0, 2.0, 4.0, None, NAN, INF]
    snapshot = copy.deepcopy(xs)
    assert math.isfinite(boxcox_lambda(xs))
    assert math.isfinite(yeojohnson_lambda(xs))
    assert _same(xs, snapshot)
    assert math.isnan(boxcox_lambda([None, NAN]))
    assert math.isnan(yeojohnson_lambda([]))
    assert boxcox_lambda([5.0, 5.0]) == 1.0  # constant values: lambda defaults to 1


def test_yeojohnson_fitted_lambda_round_trip_over_negative_and_positive_values():
    xs = _mixed_data() + [10.0, -10.0]
    t = FeatureTransformer("yeojohnson").fit(xs)
    back = t.inverse_transform(t.transform(xs))
    for a, b in zip(back, xs):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


@pytest.mark.parametrize("lam", [-1.5, -0.3, 0.0, 1e-13, 0.5, 2.0, 2.0 + 1e-13, 3.0])
def test_yeojohnson_fixed_lambda_round_trip(lam):
    xs = _mixed_data()
    t = FeatureTransformer("yeojohnson", lmbda=lam).fit(xs)
    back = t.inverse_transform(t.transform(xs))
    for a, b in zip(back, xs):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def test_boxcox_round_trip_fitted_and_fixed():
    xs = _positive_data()
    fitted = FeatureTransformer("boxcox").fit(xs)
    fixed = FeatureTransformer("boxcox", lmbda=0.3).fit(xs)
    for t in (fitted, fixed):
        back = t.inverse_transform(t.transform(xs))
        for a, b in zip(back, xs):
            assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


# ---------- missing, NaN and infinity pass-through ----------


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_missing_and_infinite_pass_through_exactly(kind):
    base = _fit_data_for(kind)
    values = [base[0], None, NAN, INF, -INF, base[1]]
    snapshot = copy.deepcopy(values)
    t = FeatureTransformer(kind)
    t.fit(values)
    assert (t.n_missing_, t.n_inf_) == (2, 2)
    assert t.n_fit_ == 2

    out = t.transform(values)
    back = t.inverse_transform(out)
    for res in (out, back):
        assert len(res) == len(values)
        assert res[1] is None
        assert math.isnan(res[2])
        assert res[3] == INF
        assert res[4] == values[4]
    assert _same(values, snapshot)


def test_nan_is_returned_as_nan_object_not_none():
    t = FeatureTransformer("mean_diff").fit([1.0, 3.0])
    out = t.transform([NAN])
    assert isinstance(out[0], float) and math.isnan(out[0])


# ---------- empty and all-missing edge cases ----------


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_empty_and_all_missing_columns_do_not_raise(kind):
    t = FeatureTransformer(kind).fit([])
    assert t.n_fit_ == 0
    assert t.transform([]) == []

    t2 = FeatureTransformer(kind).fit([None, NAN])
    assert t2.n_fit_ == 0 and t2.n_missing_ == 2
    assert _same(t2.transform([None, NAN]), [None, NAN])
    assert _same(t2.inverse_transform([None, NAN]), [None, NAN])


def test_data_dependent_transform_of_observed_value_after_empty_fit_raises():
    with pytest.raises(ValueError, match="no observed values"):
        FeatureTransformer("mean_diff").fit([None]).transform([1.0])


# ---------- errors ----------


def test_transform_before_fit_raises_runtime_error():
    with pytest.raises(RuntimeError):
        FeatureTransformer("log1p").transform([1.0])
    with pytest.raises(RuntimeError):
        GroupedFeatureTransformer("mean_diff").transform([1.0], ["a"])


def test_log1p_rejects_values_at_or_below_minus_one_with_index():
    with pytest.raises(ValueError, match="index 2"):
        FeatureTransformer("log1p").fit([0.0, 1.0, -1.0, -5.0])
    t = FeatureTransformer("log1p").fit([0.0, 1.0])
    with pytest.raises(ValueError, match="index 1"):
        t.transform([0.0, -2.0])
    assert t.transform([-1.0 + 1e-9])  # just inside the domain is fine


def test_boxcox_rejects_nonpositive_values_with_index():
    with pytest.raises(ValueError, match="index 1"):
        FeatureTransformer("boxcox").fit([1.0, 0.0, 2.0])
    t = FeatureTransformer("boxcox").fit([1.0, 2.0])
    with pytest.raises(ValueError, match="index 0"):
        t.transform([-1.0])


def test_unknown_kind_and_parameter_errors():
    with pytest.raises(ValueError):
        FeatureTransformer("zscore")
    with pytest.raises(TypeError):
        FeatureTransformer("log1p", p=2.0)
    with pytest.raises(TypeError):
        FeatureTransformer("boxcox").fit(["a"])
    with pytest.raises(TypeError):
        FeatureTransformer("mean_diff").fit([True, 1.0])


# ---------- grouped transformer ----------


def test_grouped_mean_diff_one_parameter_set_per_group():
    values = [1.0, 3.0, 10.0, 20.0]
    groups = ["a", "a", "b", "b"]
    g = GroupedFeatureTransformer("mean_diff")
    out = g.fit_transform(values, groups)
    assert out == [-1.0, 1.0, -5.0, 5.0]
    assert g.transformers_["a"].params_ == {"mean": 2.0}
    assert g.transformers_["b"].params_ == {"mean": 15.0}
    assert g.inverse_transform(out, groups) == pytest.approx(values)


def test_grouped_none_is_its_own_group_and_missing_passes_through():
    values = [1.0, None, 4.0, NAN, 10.0]
    groups = [None, "a", None, "a", None]
    snapshot = copy.deepcopy(values)
    g = GroupedFeatureTransformer("median_diff").fit(values, groups)
    assert g.transformers_[None].params_ == {"median": 4.0}
    assert g.transformers_["a"].n_fit_ == 0
    out = g.transform(values, groups)
    assert out[0] == -3.0 and out[2] == 0.0 and out[4] == 6.0
    assert out[1] is None and math.isnan(out[3])
    assert _same(values, snapshot)


def test_grouped_unseen_group_raises_keyerror_with_name_or_uses_global_fallback():
    values = [1.0, 3.0, 10.0]
    groups = ["a", "a", "b"]
    strict = GroupedFeatureTransformer("mean_diff").fit(values, groups)
    with pytest.raises(KeyError, match="c"):
        strict.transform([1.0], ["c"])

    lenient = GroupedFeatureTransformer("mean_diff", fallback="global").fit(
        values, groups
    )
    global_mean = (1.0 + 3.0 + 10.0) / 3.0
    assert lenient.transform([1.0], ["c"]) == [pytest.approx(1.0 - global_mean)]


def test_grouped_empty_and_all_missing_and_deterministic():
    g = GroupedFeatureTransformer("boxcox").fit([], [])
    assert g.transform([], []) == []
    g2 = GroupedFeatureTransformer("yeojohnson").fit([None, NAN], ["x", "x"])
    assert _same(g2.transform([None, NAN], ["x", "x"]), [None, NAN])

    values = _positive_data() + [2.0, 3.0]
    groups = ["p", "q", "p", "q", "p", "q", "p", "q"][: len(values)]
    first = GroupedFeatureTransformer("boxcox").fit_transform(values, groups)
    second = GroupedFeatureTransformer("boxcox").fit_transform(values, groups)
    assert first == second


def test_grouped_domain_error_reports_input_row_not_group_position():
    with pytest.raises(ValueError, match="index 2"):
        GroupedFeatureTransformer("log1p").fit([0.0, 1.0, -2.0], ["a", "b", "b"])
    g = GroupedFeatureTransformer("boxcox").fit([1.0, 2.0, 3.0], ["a", "b", "b"])
    with pytest.raises(ValueError, match="index 3"):
        g.transform([1.0, 2.0, 1.0, 0.0], ["a", "b", "a", "b"])


def test_grouped_rejects_bad_fallback_and_length_mismatch():
    with pytest.raises(ValueError):
        GroupedFeatureTransformer("log1p", fallback="mean")
    with pytest.raises(ValueError):
        GroupedFeatureTransformer("log1p").fit([1.0, 2.0], ["a"])


# ---------- determinism and convenience functions ----------


@pytest.mark.parametrize("kind", ["boxcox", "yeojohnson"])
def test_lambda_fit_is_deterministic(kind):
    xs = _positive_data() if kind == "boxcox" else _mixed_data()
    a = FeatureTransformer(kind).fit(xs)
    b = FeatureTransformer(kind).fit(list(xs))
    assert a.params_ == b.params_
    assert a.transform(xs) == b.transform(xs)


def test_convenience_lambda_functions_match_class():
    xs = _positive_data()
    assert boxcox_lambda(xs) == FeatureTransformer("boxcox").fit(xs).params_["lambda"]
    ys = _mixed_data()
    assert (
        yeojohnson_lambda(ys)
        == FeatureTransformer("yeojohnson").fit(ys).params_["lambda"]
    )
    assert isinstance(boxcox_lambda(xs), float)


def test_inputs_are_never_modified_by_fit_transform():
    xs = [1.0, None, NAN, 2.0, INF]
    snapshot = copy.deepcopy(xs)
    for kind in ("log1p", "mean_diff", "median_diff", "yeojohnson"):
        FeatureTransformer(kind).fit_transform(xs)
        assert _same(xs, snapshot)
