"""Response-scale predictions and fitted distributions through ModelTree's public API."""

import math

import pytest

from mltreelib import ModelTree


@pytest.mark.parametrize(
    "family, targets, weights, expected",
    [
        ("binomial", [0.0, 1.0, 0.0, 1.0], [3.0, 1.0, 1.0, 3.0], [0.25, 0.75]),
        ("poisson", [0.0, 2.0, 1.0, 3.0], [1.0, 3.0, 1.0, 3.0], [1.5, 2.5]),
    ],
)
def test_weighted_linear_leaf_recovers_two_group_mle(family, targets, weights, expected):
    # An intercept and one indicator saturate the two groups. Their weighted
    # event rates/count means are exact MLEs, not ordinary unweighted means.
    model = ModelTree(leaf_family=family, gain="deviance", max_depth=0, l2=0.0)
    model.fit(
        [[0], [0.0], [1], [1.0], [None], [float("nan")]],
        targets + [None, float("nan")],
        w=weights + [1000.0, 1000.0],
    )
    assert model.predict([[0], [1]]) == pytest.approx(expected, abs=1e-6)
    if family == "binomial":
        assert model.predict_proba([[0], [1]]) == pytest.approx(expected, abs=1e-6)
        assert model.cdf([[0], [1]], 0.5) == pytest.approx([0.75, 0.25], abs=1e-6)
    else:
        with pytest.raises(ValueError, match="[Bb]inomial"):
            model.predict_proba([[0]])
        assert model.cdf([[0], [1]], 0.0) == pytest.approx(
            [math.exp(-1.5), math.exp(-2.5)], abs=1e-6,
        )


@pytest.mark.parametrize(
    "family, targets, weights, expected",
    [
        ("binomial", [0.0, 1.0, 0.0, 1.0], [3.0, 1.0, 1.0, 3.0], [0.25, 0.75]),
        ("poisson", [0.0, 2.0, 1.0, 3.0], [1.0, 3.0, 1.0, 3.0], [1.5, 2.5]),
    ],
)
def test_categorical_mean_leaves_fit_weighted_distributions(family, targets, weights, expected):
    model = ModelTree(
        leaf_family=family, leaf="mean", gain="deviance", max_depth=1,
        min_rows=1, minprob=0.0, prune="none",
    )
    model.fit(
        [["low"], ["low"], ["high"], ["high"], [None]],
        targets + [None], w=weights + [1000.0], dtypes=["categorical"],
    )
    assert model.n_leaves() == 2
    assert model.predict([["low"], ["high"]]) == pytest.approx(expected, abs=1e-6)
    # The fitted CDF must consume the same raw string vocabulary as predict.
    if family == "binomial":
        assert model.predict_proba([["low"], ["high"]]) == pytest.approx(expected)
        assert model.cdf([["low"], ["high"]], 0.0) == pytest.approx([0.75, 0.25])
    else:
        # For a Poisson variable, P(Y <= 1) = exp(-mean) * (1 + mean).
        assert model.cdf([["low"], ["high"]], 1.5) == pytest.approx(
            [2.5 * math.exp(-1.5), 3.5 * math.exp(-2.5)], abs=1e-6,
        )
    raw_missing = [[None], [float("nan")], ["unseen"]]
    predictions = model.predict(raw_missing)
    assert len(predictions) == 3 and all(math.isfinite(p) for p in predictions)
    assert len(model.cdf(raw_missing, 0.0)) == 3
    if family == "binomial":
        assert model.predict_proba(raw_missing) == pytest.approx(predictions)


@pytest.mark.parametrize("family, targets", [("binomial", [0.0, 1.0, 1.0]), ("poisson", [0.0, 2.0, 4.0])])
def test_observed_targets_with_missing_numeric_predictors_train(family, targets):
    model = ModelTree(leaf_family=family, gain="deviance", max_depth=0, l2=0.1)
    model.fit([[0.0], [None], [float("nan")], [1.0]], targets + [None])
    predictions = model.predict([[None], [float("nan")], [1.0]])
    assert len(predictions) == 3 and all(math.isfinite(p) for p in predictions)
    if family == "binomial":
        assert all(0.0 <= p <= 1.0 for p in predictions)
    else:
        assert all(p >= 0.0 for p in predictions)


@pytest.mark.parametrize("family, target", [("binomial", -1.0), ("binomial", 0.5), ("binomial", 2.0), ("poisson", -1.0), ("binomial", float("inf")), ("poisson", float("inf"))])
def test_invalid_observed_targets_are_rejected(family, target):
    model = ModelTree(leaf_family=family, max_depth=0)
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0.0, target])


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="leaf_family"):
        ModelTree(leaf_family="gamma")


@pytest.mark.parametrize("family", ["binomial", "poisson"])
@pytest.mark.parametrize("leaf", ["median", "quantile", "huber"])
def test_non_gaussian_unsupported_leaf_shapes_are_rejected(family, leaf):
    with pytest.raises(ValueError):
        model = ModelTree(leaf_family=family, leaf=leaf)
        model.fit([[0], [1]], [0.0, 1.0])


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_non_gaussian_chow_gate_is_rejected(family):
    with pytest.raises(ValueError):
        model = ModelTree(leaf_family=family, split_test="chow")
        model.fit([[0], [1]], [0.0, 1.0])


def test_gaussian_default_stays_response_scale_and_rejects_distribution_helpers():
    model = ModelTree(leaf="mean", max_depth=0)
    model.fit([[0], [1]], [2.0, 4.0])
    assert model.predict([[0], [None]]) == pytest.approx([3.0, 3.0])
    with pytest.raises(ValueError, match="[Bb]inomial"):
        model.predict_proba([[0]])
    with pytest.raises(ValueError):
        model.cdf([[0]], 0.0)


@pytest.mark.parametrize("family, targets, expected", [("binomial", [1.0, 1.0], 1.0), ("poisson", [0.0, 0.0], 0.0)])
def test_boundary_targets_have_finite_predictions(family, targets, expected):
    model = ModelTree(leaf_family=family, max_depth=0, l2=0.0)
    model.fit([[0], [1]], targets)
    predictions = model.predict([[0], [1], [None]])
    assert all(math.isfinite(p) for p in predictions)
    assert predictions == pytest.approx([expected] * 3, abs=1e-6)


def test_fitted_cdf_is_distinct_from_empirical_quantile():
    model = ModelTree(leaf_family="poisson", leaf="mean", max_depth=0)
    model.fit([[0], [0]], [0.0, 4.0])
    # The empirical leaf has no mass at one, but Poisson(mean=2) does.
    assert model.cdf([[0]], 1.0) == pytest.approx([3.0 * math.exp(-2.0)])
    assert model.quantile([[0]], 0.0) == pytest.approx([0.0])
    assert model.quantile([[0]], 1.0) == pytest.approx([4.0])


def test_poisson_accepts_fractional_nonnegative_rates():
    model = ModelTree(leaf_family="poisson", leaf="mean", max_depth=0)
    model.fit([[0], [1]], [0.5, 1.5])
    assert model.predict([[0], [1]]) == pytest.approx([1.0, 1.0])


def test_leaves_reports_link_coefficients_and_response_constants_separately():
    model = ModelTree(leaf_family="binomial", leaf="mean", max_depth=0)
    model.fit([[0], [0], [0], [1]], [0.0, 0.0, 0.0, 1.0])
    leaf = model.leaves()[0]
    assert leaf["intercept"] is None
    assert leaf["response_constant"] == pytest.approx(0.25)
    assert leaf["mean"] == pytest.approx(0.25)

    linear = ModelTree(leaf_family="binomial", gain="deviance", max_depth=0, l2=0.0)
    linear.fit([[0], [0], [1], [1]], [0.0, 1.0, 0.0, 1.0])
    leaf = linear.leaves()[0]
    assert math.isfinite(leaf["intercept"])
    assert leaf["response_constant"] == pytest.approx(0.5)


def test_invalid_public_targets_are_not_perfect_losses():
    with pytest.raises(ValueError):
        ModelTree(leaf_family="poisson", max_depth=0).fit([[0]], [-1.0])
    with pytest.raises(ValueError):
        ModelTree(leaf_family="binomial", max_depth=0).fit([[0]], [2.0])


def test_gaussian_lean_leaf_keeps_quantile_stats_contract():
    model = ModelTree(leaf="mean", max_depth=0, leaf_digest=False)
    model.fit([[0], [0]], [0.0, 2.0])
    assert model.predict([[0]]) == pytest.approx([1.0])
    assert math.isnan(model.quantile([[0]], 0.5)[0])
