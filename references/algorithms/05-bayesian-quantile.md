# 05 — Bayesian trees and quantile methods

## BART — Bayesian additive regression trees (Chipman, George, McCulloch)

**Source:** `extracted/BART_2.9.10/` (R `BART`), `extracted/BayesTree_0.3-1.5/`
(`BayesTree`), `extracted/bartMachine_1.4.2/` (`bartMachine`) — three
independent implementations of the same idea.
**Idea:** sum of many small regression trees with a **sparsity-inducing
prior** on tree shape and leaf values; MCMC samples the posterior → predictions
come with **uncertainty** for free.

```text
init: f(x) = Σ_{m=1..M} g_m(x), g_m = stump, leaves ~ N(0, sigma_mu)
repeat for many MCMC draws:
    for m in 1..M:
        partial residual r_i = y_i - Σ_{j != m} g_j(x_i)
        propose tree change (grow / prune / change) with Metropolis-Hastings:
            accept ∝ P(r | new tree) * P(new tree | alpha, beta) / (proposal ratio)
        draw leaf values mu ~ N(posterior mean, posterior var)
    draw sigma^2 from inverse-gamma conditional
prediction = posterior mean over draws; intervals from the draw spread
```

**Knobs:** ntree (M ≈ 200 typical), k (leaf prior spread), power/base (tree
prior), nu (error prior), ndpost/nskip (MCMC draws), num_cut (cutpoint grid),
sigma_hat.
**Variants:** `BayesTree` adds DART-style variable selection / Dirichlet
sparsity for p >> n; `bartMachine` is a Java-backed implementation with
conformal calibration hooks.
**Note:** the UQ baseline; "trees + honest uncertainty" is what foundation
models claim — BART has done it since 2010.

## Quantile Regression Forests (Meinshausen 2006) — the quantile one

**Source:** `extracted/quantregForest_1.4-0/` (R `quantregForest`).
**Idea:** keep the forest's **full leaf distributions**: predict quantiles
from the weighted empirical distribution of training targets, where the
weights count how often the query lands in the same leaf as each training row.

```text
train: grow a Random Forest as usual (keep all trees + training y)
predict_quantile(x, q):
    w_i = (1/B) Σ_b [ x_i and x fall in the same leaf of tree b ]
    return weighted quantile_q of {y_i} under weights w
```

**Knobs:** ntree, nodesize, keep.inbag (for unbiased weights), nthreads.
**Note:** distribution-free quantiles from any forest — the cheap alternative
to NGBoost heads; compare both with `Metrics` + interval scores later.

## Quantile GBT recap

See 03 (#23) for the boosting-side quantile method (pinball loss); QRF is the
forest-side counterpart. The practical split: QRF adapts to any forest and
gives full distributions; Quantile-GBT optimizes the quantile loss directly
and is usually sharper per quantile.
