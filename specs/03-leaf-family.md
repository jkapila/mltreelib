# [LEAF-FAMILY] Distributional leaves: logit, probit, poisson, then tobit/tweedie/softmax

## Problem Statement

The classification and distribution world is untouched: the tree fits
constants or linear least-squares leaves only. The maintainer wants leaves
that fit distributions — logit/probit for binary targets, poisson for
counts, later tobit (censored), tweedie (insurance-style) and softmax
(multiclass) — and probability outputs (predict_proba), so the single tree
can serve classification and distributional regression without leaving the
library.

## Solution

A `leaf_family` parameter generalizes the leaf fit: the leaf models a
distribution's parameters (working-chart/GLM style, IRLS solves on the
existing faer stack) instead of a bare mean. Family = one knob, many
variants: gaussian (today's behavior), binomial/logit first, poisson next,
probit/tobit/tweedie/softmax as variants of the same machinery.
`predict_proba` lands with the binomial family. The design follows the
saved NGBoost deep-dive: leaves fit distribution parameters directly in v1
(single tree), with boosting-into-a-distribution explicitly deferred.

## User Stories

1. As a maintainer, I want `leaf_family="binomial"` leaves, so that 0/1 targets get logistic leaf fits (probabilities by contract).
2. As a maintainer, I want `predict_proba(x)`, so that classification outputs are probabilities, not hard values.
3. As a maintainer, I want `leaf_family="poisson"` leaves with a log link, so that count/rate targets are modeled with the right mean-variance shape.
4. As a maintainer, I want `leaf_family="gaussian"` to remain exactly today's behavior, so that nothing moves without a round.
5. As a maintainer, I want probit as a link variant of binomial, so that the choice is a knob, not a fork.
6. As a maintainer, I want tobit leaves later, so that censored targets (limits, detection thresholds) are expressible.
7. As a maintainer, I want tweedie leaves later, so that insurance-style zero-inflated continuous targets fit.
8. As a maintainer, I want softmax/multinomial later, so that multiclass targets have a first-class path.
9. As a maintainer, I want the leaf family to work with linear and constant leaf shapes, so that logistic-linear leaves (GLM on path columns) are possible.
10. As a maintainer, I want the IRLS/Newton solves to run on the faer stack, so that per-leaf GLM fitting stays fast in the split scan.
11. As a maintainer, I want the distributional outputs (mean, quantiles, CDF) exposed from leaves, so that uncertainty is queryable like the existing t-digest quantiles.
12. As a maintainer, I want the family to compose with the purity gate and honesty, so that the knobs batch and this one multiply rather than conflict.
13. As a maintainer, I want family-appropriate criteria defaults suggested but not forced (binomial ↔ entropy/logloss family), so that the gain knob stays the user's.
14. As a maintainer, I want every family default-off/neutral until a round measures it, per the standing rule.

## Implementation Decisions

- `leaf_family: "gaussian"|"binomial"|"poisson"|...` is orthogonal to
  `leaf` (fit shape: linear/mean/median/quantile/huber). v1 = constant and
  linear shapes with gaussian/binomial/poisson families.
- Parameterization follows the NGBoost working-chart convention (log link
  for poisson, logit for binomial) rather than canonical natural
  parameters; the distinction is documented in the API page.
- Leaf fitting: IRLS per leaf, each iteration a weighted least-squares
  solve on the faer path (the elastic-net scout's findings — ndarray-glm's
  weighted IRLS — are the reference implementation).
- `predict_proba` returns P(class 1) rows for binomial families; for other
  families it errors with a clear message (or returns the mean CDF at 0.5 —
  to decide at implementation, documented).
- Distribution outputs ride the existing leaf stats: the t-digest stays the
  quantile engine; family CDFs/means come from the fitted parameters.
- The boosting-into-a-distribution variant (NGBoost proper) is explicitly
  out; the saved deep-dive is the design reference when it is attempted.

## Testing Decisions

- External behavior only: fitted leaf parameters produce the right mean/
  probability predictions on known regimes; predict_proba rows sum to one
  and separate a planted 0/1 structure; poisson leaves beat gaussian leaves
  on planted count data.
- Single seam: the existing core integration tests file (same seam as the
  leaf-kind and criteria tests — prior art: the median/huber leaf tests).
- Exact-algorithm pins (logistic MLE on a tiny hand-computed fixture) live
  in the same file, mirroring how the rpart-parity fixture pins exactness.

## Out of Scope

- Boosting/dist-ensembles (NGBoost-style); forests; the knobs batch
  (separate issue); R-side glmTree comparisons; calibration tooling.

## Further Notes

- The user's phrase "leaf_family is the right approach we can generalize it
  to many variants" is the design mandate; the family list order is
  binomial → poisson → probit → tobit → tweedie → softmax.

## Status (2026-10-31)

First batch **landed** (`de9ec9c`): binomial/logit and poisson/log with
constant + linear shapes, `gain="deviance"`, weighted IRLS on faer,
`predict_proba`, fitted `cdf`, family-aware pruning and `leaves()` output.
Gaussian behavior unchanged. Remaining for follow-up rounds: probit, tobit,
tweedie, softmax; distributional outputs beyond `cdf`; family/gain default
suggestion surfaced to the user (not auto-switched).
