# Spec 32: Bayesian Tree Ensembles

Status: proposed

Family: Bayesian. Distinct from the bagging, boosting, and meta families: the primary output is a **distribution over functions**, not a point prediction.

Scope boundary: this specification owns the Bayesian tree family. Spec 28 owns `ModelForest`, Spec 29 owns `ModelBoost`, Spec 30 owns `ModelMeta`.

## Problem Statement

`mltreelib` has no Bayesian inference layer. It has no MCMC, no posterior draws, no credible intervals, no posterior over model structure.

### 1. The family is not a fourth `kind`

The three existing ensemble families all produce a point prediction plus diagnostics:

| | Output | Uncertainty |
|---|---|---|
| `ModelForest` | Point prediction | Diagnostics only |
| `ModelBoost` | Point prediction | Diagnostics only |
| `ModelMeta` | Point prediction | Diagnostics only |
| **Bayesian** | **Posterior over functions** | **Credible intervals and bands** |

BART is neither bagging nor boosting. It fits no bootstrap samples, so it is not bagging. It samples all trees jointly rather than greedily and sequentially, so it is not boosting. It is a Bayesian **sum-of-trees** model fitted by backfitting MCMC.

Because the output contract differs fundamentally, a Bayesian family cannot be a `kind` value inside an existing facade. It is a **fourth task facade**, in the same pattern as Spec 25's `AnomalyModel`, `ClusterModel`, `RuleModel`, and `ComponentModel`.

### 2. But most of BART's *value* is reachable without MCMC

This is the finding that makes the family worth starting.

BART's three priors are all **growth-criterion regularisers**:

- The **tree-structure prior**, $P(\text{nonterminal} \mid d) = \alpha(1+d)^{-\beta}$ with $\alpha = 0.95$, $\beta = 2$, controls tree size and depth.
- The **leaf-shrinkage prior**, $\mu_{ij} \sim N(0, \sigma_\mu^2)$ with $\sigma_\mu = 0.5/(k\sqrt{m})$ and $k = 2$, controls how far a leaf mean can stray from zero.
- The **error-variance prior**, $\sigma^2 \sim \nu\lambda/\chi_\nu^2$ with $(\nu, q) = (3, 0.90)$, controls how much noise the model attributes to the residual.

None of these requires a sampler. They are split-scoring adjustments over the existing tree engine. Adopting them yields BART-*shaped* deterministic trees: properly regularised, less prone to overfitting small leaves than a plain CART, with a documented and tunable prior story.

**That is the first wave.** The posterior machinery is later, and it is a different project.

### 3. The local benchmark says the family matters

On the library's own 56-dataset Elo arena:

| Model | Overall Elo | Regression Elo | Classification Elo | Mean fit time |
|---|---:|---:|---:|---:|
| CatBoost(200) | 1457.8 | 1498.5 | 1372.4 | 1.01 s |
| perpetual (Rust) | 1400.1 | 1505.9 | 1314.7 | 7.19 s |
| **bart** | **1313.9** | **1243.3** | **1228.6** | **25.49 s** |
| ranger | 1298.2 | 1700.6 | 1212.8 | 17.78 s |
| ModelTree(all) | 1290.1 | 715.9 | 1204.7 | 4.01 s |

`bart` is seventh of 47 entrants. It beats ranger, `ModelTree(all)`, sklearn GBM, HistGradientBoosting, LightGBM(200), ExtraTrees, and `forust (Rust)` on overall Elo. It is the only entrant whose output is a posterior.

It also has the second-worst fit time measured, at 25.49 s against EBM's 66.59 s. That is the MCMC cost, and it is why the deterministic-prior wave comes first.

### 4. Monotone BART's semantics already match this library

mBART constrains **leaf values**, not split rules. A leaf value may not exceed the minimum above-neighbour value, nor fall below the maximum below-neighbour value, for each constrained coordinate. Every tree being constrained makes the sum monotone.

`mltreelib` already has `tree_monotone` for split direction and `leaf_monotone` for leaf coefficients. So the mBART semantics is an **audit** of an existing capability, not new work: the check is that a monotone model satisfies the constraint on the *fitted* prediction, not merely on each tree in isolation.

This is the same finding Spec 29 reached from the boosting side, and the two together mean the library's monotone machinery needs one careful audit rather than two separate features.

### 5. Six citations that do not exist

Before any implementation, the bibliography must be fixed. These are widely cited and do not exist as described:

- "Bayesian boosting," Geurst, van der Vaart, van der Velden, *AoS* 2021.
- "Safe Bayesian boosting," Simin, Grinsztajn, *JRSS-B* 2022.
- Friedman, Hastie & Tibshirani, JCGS 2001, "Bayesian backfitting."
- JASA 2001, "Additive Bayesian Regression."
- Scheibler, Carvalho, Tewari & Boyd, spike-and-slab BART, JMLR 2019.
- Kim & Oh, "EBART," JMLR 2024.

Verified substitutes are in `references/research/2026-11-22-bayesian-tree-ensembles-landscape.md` §2 and §15.

## Solution

One facade, `ModelBayes`, with three levels of capability.

### 1. The three levels

| Level | What it is | Requires MCMC | Verdict |
|---|---|---|---|
| **L1: Priors as configuration** | BART's tree prior, leaf-shrinkage prior, and error-variance prior adopted as growth-criterion configuration over the existing tree engine | No | **First wave** |
| **L2: Posterior over the sum** | Gibbs sampler with Metropolis-Hastings structure moves, leaf draws, and $\sigma$ draws | Yes | Later, experimental |
| **L3: Derived posterior products** | Credible intervals, credible bands, posterior predictive intervals, posterior variable-inclusion probabilities, interaction diagnostics | Yes | Later, experimental |

### 2. Level 1: what exactly changes

Three new growth-criterion parameters, all optional and all off by default so existing fits are bit-identical.

**The tree-structure prior.** At depth $d$, the probability that a node is nonterminal is $s_d = \alpha(1+d)^{-\beta}$, with $\alpha \in (0,1)$ and $\beta \in [0, \infty)$. A candidate split's gain is scaled by $\frac{s_d}{1 - s_d}$, the ratio of the prior probability of splitting to the prior probability of not splitting at that depth. This makes deep trees progressively harder to justify, which is the deterministic equivalent of BART's size control.

**The leaf-shrinkage prior.** A leaf mean is drawn toward zero with prior variance $\sigma_\mu^2 = \frac{0.5^2}{k^2 m}$, where $m$ is the number of trees and $k$ is the prior-SD count. In a deterministic fit this becomes a ridge-style penalty on the leaf intercept, with the penalty scaled by $\sigma_\mu^2$ rather than by the leaf sample size. The posterior mean of a BART leaf is

$$\hat\mu_j = \frac{S_j/\sigma^2}{n_j/\sigma^2 + 1/\sigma_\mu^2},$$

so the deterministic analogue is to fit the leaf with a precision-weighted prior of variance $\sigma_\mu^2$ added to the data precision $n_j/\sigma^2$. That is a shrinkage toward zero with a data-independent floor, and it is exactly what a prior-regularised leaf fit should be.

**The error-variance prior.** $\sigma^2 \sim IG(\nu/2, \nu\lambda/2)$, with $\nu$ the degrees of freedom and $\lambda$ the scale, calibrated from an empirical $\hat\sigma$ so that $P(\sigma < \hat\sigma) = q$. In a deterministic fit this sets the scale at which the leaf shrinkage prior and the leaf precision are compared. The paper's defaults are $(\nu, q) = (3, 0.90)$, and $\nu < 3$ is rejected because it places too much mass near $\sigma = 0$.

**The consequence, stated plainly.** With these three, a single tree grown under BART priors is a *different estimator* from a CART tree, and it is not the BART posterior. The report must say so. A model fitted this way is "BART-prior CART", not "BART".

### 3. Level 2: the sampler, when it lands

The full sampler is an outer Gibbs sweep with three steps per tree:

1. $T_j \mid R_j, \sigma$ via Metropolis-Hastings with the leaf means integrated out.
2. $M_j \mid T_j, R_j, \sigma$ via independent Gaussian leaf draws.
3. $\sigma \mid \cdots, y$ from its inverse-gamma full conditional.

The move set is four moves, not three: grow a terminal node at 0.25, prune a pair of terminal nodes at 0.25, change a nonterminal split rule at 0.40, and swap parent and child rules at 0.10.

**The move set is the blocker.** The library's tree builder is greedy top-down with a deterministic split scan. A Metropolis-Hastings sampler needs to propose a *single structural change* to one tree and evaluate the marginal likelihood of the whole tree under that change. That is a different growth mechanism, not a configuration of the existing one.

**Decision:** Level 2 is a separate growth engine, not a mode of the existing one. It lands under `mltreelib.experimental.bayes` and does not touch the deterministic path.

### 4. Level 3: the posterior products

Each needs the sampler, and each has a distinct implementation cost.

| Product | Mechanism | Cost |
|---|---|---|
| Credible interval for $f(x)$ | Posterior quantiles of the draw vector at $x$ | Cheap once draws exist |
| Credible band for the function | Envelope over draws | Cheap once draws exist |
| Posterior predictive interval | Integrate $\sigma$ and new-error noise | Cheap once draws exist |
| Posterior variable-inclusion probability | Per-draw split counts, then inclusion proportion | Cheap; the permutation-calibrated threshold is the work |
| Posterior interaction diagnostics | Conditional inclusion and conditional marginal difference | Moderate |
| Convergence diagnostics | Geweke, plus R-hat and effective sample size | Moderate; no package computes R-hat or ESS by default, so the library would ship its own |

### 5. The facade

```text
ModelBayes
  kind: BayesKind        # prior_cart | posterior_sum
  mode: ModelMode        # regression | classification
```

| Trait | `prior_cart` | `posterior_sum` |
|---|---|---|
| `supports_posterior` | no | yes |
| `supports_credible_interval` | no | yes |
| `supports_credible_band` | no | yes |
| `supports_predictive_interval` | no | yes |
| `supports_posterior_importance` | no | yes |
| `supports_honesty` | no | no |
| `supports_inference` | no | asymptotic only, and only for the posterior products |

`supports_honesty` is false for both kinds. Bayesian sample splitting exists in the literature, but it is not part of BART and adding it would be a separate design.

`supports_inference` is true only for `posterior_sum`, and only in the sense of the posterior products themselves. The credible intervals are **exact** under the model, up to Monte Carlo error. That is a different and stronger guarantee than GRF's asymptotic intervals, and the report must state it as such.

### 6. Public surface

```python
from mltreelib.bayes import ModelBayes

bayes = ModelBayes(
    kind="prior_cart",
    mode="regression",
    alpha=0.95,              # tree prior base
    beta=2.0,                # tree prior exponent
    k=2.0,                   # leaf prior SD count
    nu=3.0,                  # error variance prior df
    q=0.90,                  # error variance prior quantile
    sigma_hat=None,          # empirical error scale; None estimates it
    max_depth=None,
    min_samples_leaf=1,
    num_trees=1,
    random_state=None,
)

bayes.fit(X, y, weights=None)
bayes.predict(X)
bayes.predict_proba(X)          # classification
bayes.credible_interval(X, level=0.95)   # posterior_sum only
bayes.credible_band(X, level=0.95)       # posterior_sum only
bayes.posterior_importance()              # posterior_sum only
bayes.report()
bayes.config()
bayes.save(path); ModelBayes.load(path)
```

`get_params` and `set_params` are absent, per Spec 25 Decision 11. There is no sklearn import or dependency.

### 7. What is deliberately not here

- **No adaptive variable selection.** The Dirichlet split-weight prior and the ABC spike-and-forest both need split-prior plumbing across the 30 split criteria. Later.
- **No GP leaves.** GP-BART replaces the constant leaf with a Gaussian process per leaf, which needs a different leaf model and a different sampler. Later, and only if extrapolation outside the training support is a requirement.
- **No Dirichlet-process error model.** DPMBART replaces the Gaussian error with a DP mixture. Later.
- **No causal or survival BART.** Both need posterior draws of nuisance models and, for mediation, a structural-equation identification step. The deterministic causal and survival layers belong to the forest and boosting families.
- **No port of any R, Python, or Java package.** The algorithms are covered above; the packages are oracles.

## User Stories

### Level 1: Priors as Configuration

1. As a user, I want BART's tree-structure prior as a growth-criterion parameter, so that my trees are properly size-regularised.
2. As a user, I want the leaf-shrinkage prior as a leaf-fit parameter, so that leaf means shrink toward zero with a data-independent floor.
3. As a user, I want the error-variance prior configurable, so that the scale at which shrinkage is applied is explicit rather than implicit.
4. As a user, I want all three off by default, so that existing fits are bit-identical after this change.
5. As a user, I want the report to state that a prior-regularised tree is not the BART posterior, so that the model is not oversold.
6. As a user, I want the prior parameters reported alongside the other controller settings, so that the fit is reproducible.
7. As a user, I want the empirical error scale reported when it is estimated, so that the prior calibration is visible.
8. As a library maintainer, I want the prior parameters to reach the existing split engine through typed controls, so that no new split-criterion variant is needed.

### Monotone Audit

9. As a user, I want the monotone constraint verified on the fitted prediction rather than per tree, so that the constraint means what it says.
10. As a user, I want to know when a monotone constraint was active and how often it bound, so that I can see the constraint's effect.

### Level 2 and 3: The Posterior

11. As a user, I want a posterior over the regression function, so that I get credible intervals rather than point predictions with error bars.
12. As a user, I want credible bands for the whole function, so that I can see where the model is uncertain.
13. As a user, I want posterior predictive intervals that separate aleatoric from epistemic uncertainty, so that I can tell noise from ignorance.
14. As a user, I want posterior variable-inclusion probabilities, so that I can see which predictors the data support.
15. As a user, I want convergence diagnostics, so that I know the chain has mixed before I read any interval.
16. As a user, I want the sampler under an experimental namespace, so that its slower fits and different guarantees are clearly separated.
17. As a user, I want the monotone-constrained posterior variant, so that monotone constraints and posterior inference compose.

### Naming and Framing

18. As a user, I want the family named separately from the bagging, boosting, and meta families, so that I do not confuse a posterior with an averaged point prediction.
19. As a library maintainer, I want the family documented as neither bagging nor boosting, so that the comparison is accurate.

## Implementation Decisions

### Decision 1: A Fourth Task Facade, Not a Fourth `kind`

`ModelBayes` is a separate facade, matching the pattern of Spec 25's four task facades.

The reason is the output contract. The other three families return a point prediction and attach diagnostics. The Bayesian family returns a distribution. A `kind` value cannot change that, and pretending otherwise would produce an API where `predict_interval` means two different things depending on the kind, which is exactly the confusion the three-family split was created to avoid.

### Decision 2: Priors as Configuration First, Sampler Later

Level 1 ships without MCMC.

The reasoning is cost and risk. Level 1 is a contained change to split scoring and leaf fitting, it touches the existing deterministic path, and it is independently useful. Level 2 is a new growth engine, a new sampler, and a new convergence-diagnostics surface. Doing Level 1 first produces something usable and gives the library BART's regularisation story without committing to a sampler design.

### Decision 3: The Deterministic Result Is Named Honestly

A tree grown under BART priors is **not** the BART posterior, and the report says so in those words.

The temptation is to call it "BART" because the priors are the same. It is not the same estimator. BART's posterior integrates over tree structures; a deterministic prior-regularised tree picks one structure greedily under a size prior. The credible intervals, the posterior over structure, and the variable-inclusion posterior are all absent. The report's Learner section states which prior was used and which posterior products are unavailable.

### Decision 4: The Tree-Structure Prior Enters Through Gain Scaling

$P(\text{nonterminal} \mid d) = \alpha(1+d)^{-\beta}$ enters the split scan as a multiplier on the candidate's gain: $\frac{s_d}{1-s_d}$, with $s_d = \alpha(1+d)^{-\beta}$.

This is the standard way a size prior enters a deterministic greedy grower, and it requires no new split-criterion variant. It is a per-node scalar, computed once per node from the depth.

**The limitation is stated.** The gain scaling is an approximation to the prior's effect on the posterior, because a greedy grower never sees the alternatives a Bayesian posterior would average over. The report states that the prior is applied greedily.

### Decision 5: The Leaf-Shrinkage Prior Enters as a Prior Precision

The posterior leaf mean is

$$\hat\mu_j = \frac{S_j/\sigma^2}{n_j/\sigma^2 + 1/\sigma_\mu^2},$$

so the deterministic analogue is a leaf fit with prior precision $1/\sigma_\mu^2$ added to the data precision $n_j/\sigma^2$, shrinking the leaf intercept toward zero.

This is a precision-weighted fit, not a penalty on the coefficient, and the distinction matters: the shrinkage target is zero (the mean of the rescaled response), not the column mean. The leaf fit therefore needs a prior-precision term on the intercept only, not on the slope coefficients.

### Decision 6: The Error-Variance Prior Sets the Comparison Scale

$\sigma^2 \sim IG(\nu/2, \nu\lambda/2)$ with $\nu$ and $\lambda$ calibrated from an empirical $\hat\sigma$ and a quantile $q$.

In the deterministic fit, $\sigma^2$ enters only through the ratio between the data precision $n_j/\sigma^2$ and the prior precision $1/\sigma_\mu^2$, which is what makes the leaf shrinkage depth-independent. The prior therefore sets the scale at which shrinkage operates, and the report states the calibration.

**$\nu < 3$ is rejected.** The BART authors reject it because too much prior mass near $\sigma = 0$ encourages overfitting. The library's control validates the same bound.

### Decision 7: Monotone Constraints Are Audited at the Fitted Level

The check is on the fitted prediction of the whole model, not on each tree's contribution.

This is mBART's semantics, and it matches what a user means by "this model is monotone". The library's existing `leaf_monotone` enforces the constraint inside a leaf fit; the audit adds the check that the *ensemble* or *fitted* prediction is monotone on the test data, and reports when the constraint was active and how often it bound.

### Decision 8: The Sampler Is a Separate Growth Engine

Level 2's Metropolis-Hastings move set proposes a single structural change to one tree and evaluates the tree's marginal likelihood with the leaf means integrated out. The library's builder is greedy top-down with a deterministic scan.

A single tree's marginal likelihood cannot be evaluated from the existing statistics, which are sufficient for a greedy gain comparison but not for a marginal likelihood over all structures of that tree. So Level 2 is a new engine, lives under `mltreelib.experimental.bayes`, and leaves the deterministic path untouched.

### Decision 9: The Posterior Products Are Exact, and the Report Says So

Under the model, the posterior credible intervals are exact up to Monte Carlo error. That is a **stronger** guarantee than GRF's asymptotic intervals, and it is a different kind of guarantee from a bootstrap interval.

The report labels the provenance of every interval: posterior, bootstrap, conformal, or asymptotic. Spec 28 produces conformal intervals, Spec 29 produces no intervals, and the Bayesian family produces posterior intervals. Three different provenances under three different facades, each labelled.

### Decision 10: No Adaptive Variable Selection in the First Wave

The Dirichlet split-weight prior and the ABC spike-and-forest both need split-probability weights threaded through all 30 split criteria, plus a different sampler story for the deterministic case.

A deterministic approximation exists: weight the split candidates by the learned weight, which needs no MCMC. It is BUILD_LATER, after the weights have somewhere to live.

### Decision 11: One Multi-Core Job at a Time

A posterior sampler is inherently sequential in its chain structure. It can be parallel across chains, but a chain cannot be parallelised without changing its semantics. Chains therefore run one at a time, or in a separate process with its own memory budget.

## Exact Build Sequence

### Phase 0 — Prior Parameters and Controls

1. Add `alpha`, `beta`, `k`, `nu`, `q`, and `sigma_hat` to the growth controls, all defaulting to the inactive state.
2. Add the tree-structure prior as a per-node gain multiplier in the split scan.
3. Add the leaf prior precision to the leaf fit, on the intercept only.
4. Add the error-variance calibration, with the empirical scale estimated when not supplied.
5. Validate the parameter ranges: $\alpha \in (0,1)$, $\beta \ge 0$, $k > 0$, $\nu \ge 3$, $q \in (0,1)$.
6. Add Rust tests that a fit with all priors off is bit-identical to a fit before this change, that deeper trees are harder to justify as $\beta$ rises, and that leaf shrinkage is stronger for smaller $\sigma_\mu$.
7. Add Python tests for the controls, their validation, and the report's prior section.

Gate: `cargo test` and Python tests pass. A default fit is unchanged.

### Phase 1 — Report and the Honest Label

1. Add the bayes-specific report sections: the priors used, their calibration, the empirical error scale, and an explicit statement of which posterior products are unavailable.
2. Add the monotone fitted-level audit and the constraint-binding frequency.
3. Add the `prior_cart` and `posterior_sum` kinds to the facade, with `posterior_sum` failing clearly until Phase 2 lands.
4. Add Rust tests for report section presence and for the monotone audit.
5. Add Python tests for the report and the unavailable-product error.

Gate: `cargo test` and Python tests pass. The report says "not the BART posterior" in the Learner section.

### Phase 2 — Posterior Sampler

1. Add the marginal-likelihood evaluation for a single tree with the leaf means integrated out.
2. Add the four-move Metropolis-Hastings proposal: grow, prune, change rule, swap rules, at 0.25, 0.25, 0.40, 0.10.
3. Add the outer Gibbs sweep over trees, then the leaf draws, then the $\sigma$ draw.
4. Add the burn-in, retained-draw, and thinning controls, with the paper's defaults of 200 and 1000 and the packages' 100 and 1000 both documented.
5. Add the chain storage and the draw indexing.
6. Add the experimental namespace entry.
7. Add Rust tests asserting the acceptance-ratio construction, that the chain mixes on a small known problem, and that the posterior mean matches a closed-form solution where one exists.
8. Add Python tests for the sampler controls and the draw storage.

Gate: `cargo test` and Python tests pass. The posterior mean on a simple problem matches the analytic value within Monte Carlo error.

### Phase 3 — Posterior Products

1. Add credible intervals for $f(x)$ and credible bands for the function.
2. Add posterior predictive intervals integrating $\sigma$ and new-error noise.
3. Add posterior variable-inclusion probabilities and the permutation-calibrated threshold.
4. Add convergence diagnostics: Geweke, split R-hat, and effective sample size.
5. Add the monotone-constrained posterior variant.
6. Add Rust tests asserting interval coverage on a synthetic problem with known noise, and that the diagnostics detect a deliberately non-mixing chain.
7. Add Python tests for each product and for the diagnostics.

Gate: `cargo test` and Python tests pass. Interval coverage meets the nominal level within Monte Carlo error.

## Testing Decisions

- **Good tests assert external behaviour.** Tests verify recovered parameters, interval coverage, prior effects, report contents, trait errors, and reproducibility. They do not assert private sampler internals.
- **Prior-effect seam.** Assert that with all priors off the fit is identical to the pre-change fit. Assert that raising $\beta$ produces shallower trees on a dataset where depth is not useful. Assert that a smaller $k$ shrinks leaf means further toward zero.
- **Leaf-prior seam.** Assert that the leaf fit's prior term applies to the intercept only, and that a slope coefficient is not shrunk toward zero.
- **Monotone audit seam.** Assert that a model fitted with monotone constraints is monotone on the *fitted* prediction, and that a deliberately unconstrained fit fails the audit rather than passing silently.
- **Sampler seam.** Assert the acceptance ratio against a hand-computed value on a small tree. Assert that the posterior mean on a linear problem with known slope matches within Monte Carlo error. Assert that the chain's $\sigma$ draw matches the inverse-gamma full conditional.
- **Mixing seam.** Assert that a deliberately ill-conditioned chain fails the convergence diagnostics.
- **Interval seam.** Assert empirical coverage of the credible interval on synthetic data with known noise, within Monte Carlo error.
- **Report seam.** Assert the priors and their calibration appear in the report, and that the unavailable-posterior-products statement appears for `prior_cart`.
- **Trait seam.** Assert every posterior-only method raises a typed error under `prior_cart`.
- **Determinism seam.** Fit twice with the same seed and assert identical draws. Fit with different seeds and assert the draws differ but the posterior means agree within Monte Carlo error.
- **Benchmark smoke test.** Fit `prior_cart` on one deterministic dataset. Verify output schema, counts, statuses, and artifact creation.
- **Prior art.** Reuse the existing deterministic-RNG tests and the weighted-fit test conventions in `crates/mltreelib-core/tests/`.

## Out of Scope

- The bagging family. Spec 28.
- The boosting family. Spec 29.
- The meta-estimator family. Spec 30.
- Bayesian causal forests and mediation analysis. They need posterior draws of nuisance models and a structural-equation identification step.
- Bayesian survival analysis with a posterior over the survival curve.
- GP leaves, treed GP, and Dirichlet-process error models.
- Adaptive variable selection through the Dirichlet split-weight prior or spike-and-forest priors.
- Porting `BayesTree`, `BART`, `dbarts`, `bartMachine`, `XBART`, `stochtree`, or `bartz`. They are oracles.
- A GPU port. The official XBART has no GPU implementation.
- Any sklearn import, dependency, or `BaseEstimator` compatibility.

## Further Notes

- Research basis for the model specification, priors, sampler mechanics, package defaults, XBART's status, the variable-selection mechanisms, the GP hybrids, and the citation corrections: `references/research/2026-11-22-bayesian-tree-ensembles-landscape.md`.
- The selection framework that decides when a Bayesian estimator is the right choice rather than a forest, a booster, or a stack: Spec 31.
- Spec 25 owns the common report contract and the sklearn compatibility policy. This specification contributes the prior-configuration report section and the posterior-product sections.
- The BART posterior is **exact** under the model, up to Monte Carlo error. That is a different guarantee class from GRF's asymptotic intervals and from conformal intervals, and the report must label it.
- The local Elo board puts `bart` seventh at 1313.9 overall with a 25.49 s mean fit time. The fit time is the MCMC cost, and it is why the deterministic-prior wave comes first.
- One multi-core job at a time. A chain is sequential; parallel chains must not run concurrently with another multi-core fit.
