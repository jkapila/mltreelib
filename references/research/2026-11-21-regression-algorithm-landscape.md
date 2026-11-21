# Regression Algorithm Landscape for `mltreelib`

**Date:** 2026-11-21
**Scope:** Regression algorithms beyond the OLS, ridge, lasso, and elastic-net core, assessed for use as standalone regressors and as terminal leaf models.
**Decision informed:** Which regression families belong in the Rust core, which belong behind optional interfaces, and which stay external.
**Confidence:** High for classical algorithm mechanics and repository fit. Medium for recent methods where evidence comes from small studies or single domains.

**Implementation spec:** `specs/27-regression-model-expansion.md` holds the exact build sequence, module layout, contracts, and phase gates for this note. It adds the group metadata substrate that group lasso, panel entity grouping, and hierarchical pooling depend on.

---

## 1. Executive Summary

The package has a solid penalized-linear core and a broad but shallow set of leaf families. The gaps are not exotic algorithms. They are well-established regression families with clear implementation paths.

Current regression core:

- Exact Gaussian OLS and Ridge through normal equations and Cholesky.
- Lasso and Elastic Net through cyclic coordinate descent.
- L-BFGS for smooth generalized linear deviance.
- Generalized linear families: Gaussian, Binomial, Probit, Poisson, Gamma, Tweedie, Tobit.
- Leaf-level location estimators: weighted mean, weighted median, weighted quantile, Huber M-location.
- MARS spline leaves, KNN leaves, softmax leaves, survival, AFT, cure, and spline-survival leaves.
- Post-hoc isotonic calibration for probabilities.

Recommended additions, in order:

1. **Quantile regression as a full linear model.** Current quantile support is a 0D location estimate. Multi-predictor quantile regression with pinball loss is the single largest missing regression family. It adds heteroscedasticity awareness, asymmetric-loss modeling, and distributional leaf output.
2. **Huber regression as a full linear model with scale estimation.** Current Huber support is also 0D location only. Multivariate Huber with scale calibration is a small extension and directly complements the median and quantile leaves.
3. **Expectile regression.** Asymmetric squared loss. One extra parameter, smooth objective, fits the existing L-BFGS path, and recovers the mean at 0.5 and the median at 0.5 under symmetry.
4. **Non-negative least squares.** A constrained OLS variant, trivial on top of the existing Cholesky path, and directly useful for count and exposure models.
5. **Group and sparse-group lasso.** The package thesis is grouped and hierarchical structure. Current CCD supports only scalar L1 and L2 penalties. Group penalties are the natural fit.
6. **Generalized additive models.** MARS covers piecewise-linear bases. A penalized-spline GAM with backfitting gives smoother, lower-variance shape functions. Build it only if it beats MARS.
7. **Negative binomial regression.** Poisson assumes equidispersion. Negative binomial with an estimated dispersion parameter is a small extension to the existing family machinery and fixes a real modeling gap.
8. **RANSAC and Theil-Sen.** Cheap outlier-resistant estimators. Useful as external benchmark baselines rather than core models.

Not recommended for the core:

- Kernel ridge, SVR, and Gaussian process regression. Poor per-leaf scaling, and the package already has KNN and MARS for nonparametric structure.
- BART and other Bayesian tree ensembles. Monte Carlo cost and poor fit with a deterministic tree package.
- Deep neural regressors. Autodiff, GPU, and explainability assumptions conflict with the package stance.
- Zero-inflated and hurdle models. Model-specific two-part likelihoods with poor small-sample behavior. Defer until count data demand is proven.

---

## 2. Current Repository State

### 2.1 Solver core

| Module | Contents |
|---|---|
| `solver/normal.rs` | Exact Gaussian OLS and Ridge, rank-revealing fallback, coefficient inference statistics |
| `solver/ccd.rs` | Cyclic coordinate descent with L1 soft thresholding, Elastic Net, monotone coefficient projection |
| `solver/lbfgs.rs` | L-BFGS for smooth GLM deviance, Armijo line search, L2 penalty |
| `solver/dense.rs` | Dense weighted Gaussian and GLM fitting through the shared kernels |
| `solver/family.rs` | GLM families: Gaussian, Binomial, Probit, Poisson, Gamma, Tweedie, Tobit |
| `solver/summary.rs` | R-style and statsmodels-style coefficient statistics |

`SolverOptions` exposes `family`, `l1`, `l2`, `monotone`, `max_iter`, `tol`, `verbose`. There are no bounds, no weight matrices, no sample weights beyond one scalar per row, and no per-observation influence options.

### 2.2 Leaf families

`LeafKind` in `crates/mltreelib-core/src/leaf.rs`:

| Leaf | Estimator | Optimization |
|---|---|---|
| Linear | Penalized GLM deviance | IRLS, L-BFGS, CCD, evolutionary |
| Mean | Weighted mean | Closed form |
| Median | Weighted median | Closed form, 0D |
| Quantile(tau) | Weighted quantile | Closed form, 0D |
| Huber | Huber M-location | Closed form, 0D |
| MARS | Piecewise-linear basis expansion | Internal fitting |
| Prob, Odds, LogOdits | Closed-form transforms of fitted mean | None |
| KNN | Neighbor index plus weighted response | None |

`LeafFamily` extends the GLM set with Survival, AFT, Cure, and Spline.

### 2.3 The critical distinction

`LeafKind::Quantile`, `LeafKind::Median`, and `LeafKind::Huber` are **0D location estimators**. They ignore every predictor column. They are not regression models in the multi-predictor sense.

`stats.rs` confirms this: `w_median`, `w_quantile`, and `huber_location` each consume `&[(f64, f64)]` of value-weight pairs and return one scalar.

This is the single biggest framing error in the current leaf portfolio. A user asking for "quantile regression" wants conditional quantiles $\hat Q_\tau(y \mid x)$, not a constant.

### 2.4 Structural gap analysis

| Regression family | Core support | Gap |
|---|---|---|
| OLS / Ridge | Full | None |
| Lasso / Elastic Net | Full, Gaussian only | No non-Gaussian L1 |
| Huber regression | 0D only | No multivariate version, no scale estimation |
| Quantile regression | 0D only | No multivariate version, no asymmetric loss |
| Expectile regression | None | Missing entirely |
| Non-negative least squares | None | Missing |
| Group / sparse-group lasso | None | Missing, despite grouped-structure thesis |
| GAM | MARS partial | No penalized smooth basis, no backfitting |
| Negative binomial | None | Poisson only, assumes equidispersion |
| Beta regression | None | Missing for bounded continuous outcomes |
| Ordinal regression | None | Missing, already covered in the leaf gap note |
| Partial least squares | None | Covered by the component-analysis note |
| Kernel ridge / SVR | None | Deliberately out of scope |
| Gaussian process | None | Deliberately out of scope |
| BART | None | Deliberately out of scope |

---

## 3. Quantile Regression

### 3.1 Mechanics

Quantile regression minimizes the pinball (quantile) loss:

\[
\rho_\tau(r) = r\left(\tau - \mathbb{1}(r < 0)\right), \qquad r = y - x^\top\beta
\]

Equivalently, it solves the linear programming formulation

\[
\min_\beta \sum_i \left[\tau (y_i - x_i^\top\beta)_+ + (1-\tau)(x_i^\top\beta - y_i)_+\right]
\]

Koenker and Bassett established the estimator. Koenker's annual review covers four decades of development.

Gradient form for smooth optimization:

\[
\frac{\partial}{\partial\beta} \sum_i \rho_\tau(y_i - x_i^\top\beta) = \sum_i \left(\tau - \mathbb{1}(r_i < 0)\right) x_i
\]

The loss is convex but nondifferentiable at zero residual. Standard approaches are iteratively reweighted least squares with a smoothed weight, or linear programming via a simplex or interior-point method.

### 3.2 Why it belongs here

- Heteroscedastic and skewed outcomes are common in tabular data. A mean model gives one number; quantile regression gives the whole conditional distribution.
- The loss is convex. A proximal or active-set method fits the existing solver architecture.
- It reuses the same design matrix, weight, and intercept conventions as the current GLM path.
- Quantile forests and distributional regression trees are active research areas. Recent work builds efficient distributional regression trees on weighted interval score and continuous ranked probability score losses for calibrated probabilistic forecasting.
- Penalized quantile regression with lasso and elastic-net penalties is now packaged in R through `rqPen`, with a 2025 release.

### 3.3 Package fit

**Tier 1. Highest-value single addition.**

Implementation paths, in order of practicality:

1. **Proximal or smoothed- IRLS.** The pinball loss decomposes into a smooth part plus an L1-type asymmetric term. FISTA with asymmetric soft thresholding, or proximal gradient, handles it using the generic solver seam planned in Spec 26.
2. **IRLS with smoothed weights.** Weight $w_i = \tau$ when $r_i > 0$ and $1-\tau$ otherwise, smoothed near zero. Simple, converges well, but the smoothing parameter is a numerical artifact that must be documented.
3. **Linear programming.** Exact but slow and needs an LP solver. The package already rejected a heavyweight LP dependency in the COBYLA decision.

Recommended surface: `LeafKind::Quantile(tau)` becomes a genuine multivariate quantile regression when predictors are present, keeping the 0D weighted-quantile behavior as the degenerate no-predictor case. Report the pinball loss, the fitted tau, and convergence status.

### 3.4 Risks

- Small leaves produce unstable quantile fits. A minimum-support rule is required.
- Quantile crossing between multiple taus. If several taus are fitted jointly, enforce monotonicity or report crossings.
- The asymmetric loss breaks the exchangeability of residuals that many inference statistics assume. Coefficient-level inference must not be reported without a quantile-appropriate method.

---

## 4. Huber Regression with Scale Estimation

### 4.1 Mechanics

Huber regression minimizes

\[
\sum_i \rho_\delta\left(\frac{y_i - x_i^\top\beta}{\hat\sigma}\right)
\]

where the Huber loss is quadratic below the threshold $\delta$ and linear above it:

\[
\rho_\delta(r) =
\begin{cases}
\frac{1}{2} r^2, & |r| \le \delta, \\
\delta |r| - \frac{1}{2}\delta^2, & |r| > \delta
\end{cases}
\]

The scale $\hat\sigma$ must be estimated. A common route is a robust scale estimate such as MAD scaled by a consistency factor, updated iteratively.

Huber (1964) established the M-estimator framework. Bellec and colleagues (2025, JMLR) give recent error estimation and adaptive tuning for unregularized robust M-estimators.

### 4.2 Why it belongs here

- The package already has a 0D Huber location estimator. Extending to multivariate regression is a natural step.
- Huber keeps statistical efficiency near the Gaussian model while resisting outliers. That is the right default for a tree leaf, where outliers often concentrate in small leaves.
- The objective is convex and smooth except at the transition. It fits L-BFGS with a line search.
- It pairs directly with the existing median leaf, which is the fully-inefficient limit.

### 4.3 Package fit

**Tier 1. Small extension.**

Implementation:

1. Robust scale estimate per leaf, from weighted MAD of initial residuals, scaled for consistency under normality.
2. Huber loss and gradient in the family or leaf layer.
3. IRLS or L-BFGS solve with the scale updated between outer iterations.
4. Expose the tuning constant $\delta$ and the final scale in the result.

This also supplies a **robust regression baseline** for benchmarks, since no core multivariate robust estimator exists today.

### 4.4 Risks

- Scale and location estimates can diverge. The outer iteration needs a documented convergence and fallback rule.
- The tuning constant $\delta$ changes the estimator from near-OLS to near-LAD. It must be explicit, not hidden.
- The current 0D Huber estimator must keep working unchanged.

---

## 5. Expectile Regression

### 5.1 Mechanics

Expectile regression minimizes an asymmetric squared loss:

\[
\min_\beta \sum_i \rho_\tau^{\exp}\left(y_i - x_i^\top\beta\right), \qquad
\rho_\tau^{\exp}(r) =
\begin{cases}
\tau r^2, & r \ge 0, \\
(1-\tau) r^2, & r < 0
\end{cases}
\]

Expectiles interpolate between the mean at $\tau = 0.5$ and the median at $\tau = 0.5$ under symmetric errors, depending on parameterization convention.

### 5.2 Why it belongs here

- The loss is smooth. It fits the existing L-BFGS path with no proximal machinery.
- It gives an asymmetric loss family that complements quantile regression: smooth, cheap, and interpolating the mean.
- Expectile regression is the standard companion to quantile regression for loss-based learning, and R's `expectreg` package documents current practice as of 2026.
- Hybrid quantile-expectile methods remain active research, with a 2025 arXiv preprint on efficient combinations.

### 5.3 Package fit

**Tier 1, but after quantile regression.**

Quantile regression gets priority because it targets conditional quantiles, which users ask for by name. Expectiles add a smooth asymmetric option at low cost once the loss framework exists.

Expectile loss and gradient live beside the quantile loss. Both share the same asymmetric-weight structure, so a single implementation can serve both with a different weight function.

### 5.4 Risks

- Expectiles are less widely understood than quantiles. Documentation must show the mapping and the mean-recovery property.
- A user asking for "asymmetric regression" must not confuse expectiles with quantiles.

---

## 6. Non-Negative Least Squares

### 6.1 Mechanics

NNLS solves

\[
\min_{\beta \ge 0} \frac{1}{2}\|X\beta - y\|_2^2
\]

The active-set algorithm of Lawson and Hanson selects a subset of variables, solves an unconstrained least-squares problem on that subset, and adjusts the active set until all coefficients are nonnegative.

scikit-learn exposes this as `LinearRegression(positive=True)`.

### 6.2 Why it belongs here

- Natural for count rates, exposures, prices, and other nonnegative quantities.
- The package already supports monotone coefficient projection inside CCD. NNLS is the same idea applied to the unpenalized OLS path.
- Small implementation on top of the existing Cholesky machinery.
- Directly relevant to the package's count and exposure modeling.

### 6.3 Package fit

**Tier 1, low cost.**

Non-negativity is a solver capability, not a new leaf family. Under the Spec 26 solver-capability model, NNLS becomes a bound-constrained variant of the direct Gaussian path: the same Cholesky solve with an active-set loop.

It also provides the simplest test case for the bounds work in Spec 26, since it needs no iterative solver.

### 6.4 Risks

- Degenerate and rank-deficient designs need careful pivoting.
- The intercept is usually unpenalized but should also be constrained to be nonnegative, which is a semantic decision to document.

---

## 7. Group and Sparse-Group Lasso

### 7.1 Mechanics

Group lasso applies an L2 norm penalty over groups of coefficients:

\[
\min_\beta \frac{1}{2n}\|X\beta - y\|_2^2 + \alpha \sum_g \sqrt{\beta_g^\top\beta_g}
\]

It selects whole groups. Yuan and Lin (2006) established the estimator. The sparse-group lasso adds an additional scalar L1 penalty, selecting groups and members within groups simultaneously. Simon et al. (2013) cover nonconvex grouped selection.

### 7.2 Why it belongs here

- The package thesis is grouped and hierarchical structure. Current CCD supports scalar L1 and L2 only.
- Group penalties match feature groups, interaction groups, hierarchical variables, and multi-column encodings.
- Group lasso is a natural extension of the existing coordinate descent: group soft thresholding replaces coordinate soft thresholding.
- The group structure already exists in the package as `interaction_groups` and `leaf_cols` concepts.

### 7.3 Package fit

**Tier 1 for the package thesis, Tier 2 for general value.**

Group lasso is the most distinctive addition relative to standard libraries. A generic library user might not need it; a tree-library user working with grouped columns will.

Implementation path:

1. Group metadata on the design view: a list of coefficient-index groups.
2. Group soft thresholding in the CCD update, scaling each group by its L2 norm.
3. Sparse-group lasso as group thresholding plus per-coordinate thresholding.
4. Group-aware coordinate ordering for deterministic results.

### 7.4 Risks

- Group definition must be explicit and validated. An accidental group of one changes the estimator.
- Non-overlapping groups simplify the math. Overlapping groups need a different algorithm.
- Group selection consistency is an asymptotic property, not a finite-sample guarantee.

---

## 8. Generalized Additive Models

### 8.1 Mechanics

A GAM models

\[
g(E[Y \mid x]) = \beta_0 + \sum_j f_j(x_j)
\]

with smooth univariate functions $f_j$, fitted by penalized least squares with backfitting. The R `mgcv` package uses thin-plate or cubic regression splines with a roughness penalty.

Shape-constrained variants constrain the direction of each shape function. `scam` provides these in R, with a 2026 release.

### 8.2 Why it belongs here

- Captures smooth nonlinear effects without dense interactions.
- Interpretable through shape functions and partial dependence.
- The package already has MARS, which is a piecewise-linear basis expansion with interaction detection.

### 8.3 Package fit

**Tier 2. Conditional on evidence.**

The MARS overlap is substantial. A GAM leaf must earn its place by offering smoother, lower-variance shape functions than MARS at comparable interpretability.

Recommended sequence:

1. Improve MARS first: better knot placement, smoother basis options, and explicit regularization strength.
2. Benchmark MARS against a penalized-spline GAM on package datasets.
3. Add a GAM leaf only if it wins on accuracy or stability at equal interpretability cost.

A pure-Rust GAM exists as the `solow-gam` crate (BSD-3-Clause), with penalized B-splines, cyclic cubic splines, and smoothing-parameter selection by generalized cross-validation. Download volume is very low at roughly 219 total, so treat it as a reference rather than a dependency.

### 8.4 Risks

- Duplicating MARS under a second name is the main failure mode.
- Smoothing-parameter selection adds a hyperparameter that must be deterministic.

---

## 9. Negative Binomial Regression

### 9.1 Mechanics

The negative binomial GLM has variance

\[
V(\mu) = \mu + \alpha \mu^2
\]

with an extra dispersion parameter $\alpha$. Two parameterizations exist: NB1 keeps the mean as the modeled quantity, NB2 models the dispersion. Fitting usually alternates between estimating $\beta$ given $\alpha$ and estimating $\alpha$ given $\beta$.

Poisson assumes equidispersion. Negative binomial fixes overdispersion from population heterogeneity, clustering, or omitted covariates.

### 9.2 Why it belongs here

- The package already has Poisson. Negative binomial is a small extension to the family machinery.
- Count data with overdispersion is a common real case. Current Poisson leaves give biased standard errors there.
- A 2026 R vignette on `fastglm` documents negative binomial, hurdle, and zero-inflation as the three main count-model extensions beyond family-driven GLMs.

### 9.3 Package fit

**Tier 1, small extension.**

- Add a `NegativeBinomial` variant to `GlmFamily` with an extra dispersion parameter.
- Extend the variance function and deviance.
- Fit with alternating estimation: IRLS for $\beta$ given $\alpha$, moment or maximum-likelihood update for $\alpha$.
- Report the dispersion parameter and a dispersion test against Poisson.

### 9.4 Risks

- The dispersion parameter can approach zero, degenerating to Poisson. Handle that boundary explicitly.
- Leaf-level dispersion estimates are unstable in small leaves.
- Zero inflation is a separate model family. Do not conflate the two.

---

## 10. Other GLM Family Extensions

### 10.1 Beta regression

For outcomes bounded strictly in $(0,1)$, beta regression models the mean through a logit link with a beta distribution. It handles bounded continuous data that a Gaussian model or a binomial model handles poorly.

**Package fit:** Tier 2. Clean fit with the existing family machinery. Add after negative binomial.

### 10.2 Zero-inflated and hurdle models

Two-part models for excess zeros. Each needs its own likelihood and structural-zero submodel.

**Package fit:** Defer. Model-specific, poor small-sample behavior, and no proof of demand yet. Zero-inflated negative binomial and hurdle Poisson are the common forms.

### 10.3 Ordinal regression

Cumulative logit and probit for ordered categorical targets.

**Package fit:** Tier 1 by another route. Already analyzed in `2026-11-21-core-ml-leaf-family-gap-analysis.md`. Not repeated here.

### 10.4 Inverse Gaussian, beta-binomial, and quasi-family

Lower-demand GLM extensions. Add when a concrete dataset needs one.

---

## 11. Robust and Resilient Estimators as Benchmarks

### 11.1 RANSAC

Randomly samples minimal subsets, fits, and counts consensus. Handles heavy outlier contamination. Deterministic with a fixed seed.

**Package fit:** External benchmark baseline. The generic random-search machinery in Spec 26 could support it, but RANSAC is not a leaf model and does not belong in the solver core.

### 11.2 Theil-Sen

Median of pairwise slopes. Robust with a high breakdown point. Cost grows quadratically with sample size.

**Package fit:** External benchmark baseline. Same reasoning as RANSAC.

### 11.3 Least absolute deviations

Minimizes the sum of absolute residuals. The multivariate LAD estimator.

**Package fit:** Derivative-free through the Spec 26 seam, but the convex quantile-style solver handles it. Defer as a distinct method.

### 11.4 Why these matter even when not implemented

They provide the **independent oracle** for the robust estimators the package does implement. Huber regression needs a reference for its breakdown behavior. Without RANSAC or Theil-Sen in the benchmark suite, a Huber regression improvement cannot be distinguished from a generic least-squares change.

---

## 12. Deliberately Out of Scope

| Family | Decision | Reason |
|---|---|---|
| Kernel ridge regression, SVR | Reject for core | Per-leaf kernel matrices do not scale. KNN and MARS cover nonparametric structure. |
| Gaussian process regression | Reject for core | Requires per-leaf solve or variational approximation. Poor fit for small leaves. Recent sparse-variational work remains research-scale. |
| BART and Bayesian tree ensembles | Reject for core | Monte Carlo posterior sampling conflicts with the deterministic package stance. |
| Deep neural regressors, tabPFN regressors | External benchmark only | Autodiff and GPU assumptions. Package already benchmarks these externally. |
| Explainable Boosting Machine | External benchmark only | Gradient boosting with interaction detection is an ensemble strategy, not a leaf family. It belongs beside the tree ensembles in Spec 14 and 15. |
| Partial least squares | Defer to component analysis | Covered by `2026-11-21-component-analysis-and-unified-model-reports.md`, not a leaf family. |
| Mixed-effects and hierarchical regression | Defer | Requires random-effect design and sparse Cholesky. Package has grouped structure but no mixed-model machinery. |

---

## 13. Rust Ecosystem Reuse

| Crate | Coverage | License | Downloads | Role |
|---|---|---|---|---|
| `solow` | Broad stack: regression, GLMs, mixed effects, Bayesian, survival, SVM, kernels, tree ensembles | BSD-3-Clause | roughly 155 total | Too broad and too small to adopt. Reference only. |
| `solow-gam` | GAM with penalized B-splines, cyclic cubic splines, GCV smoothing selection | BSD-3-Clause | roughly 219 total | Reference for a GAM implementation. |
| `linfa-linear` | Pure-Rust linear regression family | MIT | established | Reference only. Package already owns this surface. |
| `linreg-core` | OLS, Ridge, Lasso, Elastic Net, and more | — | established | Reference only. |
| `anofox-regression` | Robust statistics regression library | MIT | growing | Reference for robust estimator implementations. |

Decision: **no external regression crate becomes a runtime dependency.** The package owns its numerical semantics: missing-data handling, weight conventions, monotone projection, intercept treatment, and deterministic ordering. A general regression crate would require an adapter over every call site, exactly as decided for the solver crates in Spec 26.

Use these crates as dev-dependencies and benchmark oracles.

---

## 14. Implementation-Fit Matrix

| Method | New dependency | Runtime shape | Memory | Mixed types | Interpretability | Tier |
|---|---:|---:|---:|---:|---:|---:|
| Quantile regression | None | $O(NP)$ per iteration | $O(P)$ | Yes | High | 1 |
| Huber regression with scale | None | $O(NP)$ per iteration | $O(P)$ | Yes | High | 1 |
| Expectile regression | None | $O(NP)$ per iteration | $O(P)$ | Yes | High | 1 |
| Non-negative least squares | None | $O(NP^2)$ active set | $O(P^2)$ | Yes | High | 1 |
| Group lasso | None | $O(NP)$ per group sweep | $O(P)$ | Yes | High | 1 |
| Sparse-group lasso | None | $O(NP)$ per group sweep | $O(P)$ | Yes | High | 1 |
| Negative binomial GLM | None | $O(NP)$ per iteration | $O(P)$ | Yes | High | 1 |
| Beta regression | None | $O(NP)$ per iteration | $O(P)$ | Yes | High | 2 |
| GAM | None | $O(NP)$ per backfit sweep | $O(P + B)$ | Yes | High | 2 |
| Partial least squares | None | $O(NPB)$ | $O(PB)$ | Numeric | Medium | 2 |
| RANSAC | None | $O(K \cdot N)$ | $O(P)$ | Yes | Low | 3 |
| Theil-Sen | None | $O(N^2)$ | $O(N^2)$ or streamed | Yes | Low | 3 |
| Kernel ridge / SVR | Linear algebra | $O(N^3)$ or $O(N^2)$ | $O(N^2)$ | Extension-dependent | Low | 4 |
| Gaussian process | Linear algebra | $O(N^3)$ exact | $O(N^2)$ | Via kernel | Medium | 4 |

---

## 15. Recommended Build Sequence

### Phase 1 — Quantile regression

1. Add the pinball loss and gradient to the family or leaf layer.
2. Fit through the Spec 26 generic solver seam using a proximal or smoothed-IRLS path.
3. Promote `LeafKind::Quantile(tau)` from a 0D location estimator to a multivariate model when predictors are present.
4. Add expected sign loss as a reported metric.
5. Add Rust and Python tests on synthetic data with known conditional quantiles.

### Phase 2 — Huber regression with scale

1. Add robust scale estimation per leaf.
2. Add the Huber loss and gradient.
3. Add the multivariate Huber leaf and the robust regression benchmark baseline.
4. Compare against RANSAC and Theil-Sen oracles in the benchmark suite.

### Phase 3 — Expectile regression

1. Add the asymmetric squared loss.
2. Reuse the asymmetric-weight structure from quantile regression.
3. Add the expectile leaf with mean-recovery and median-recovery tests.

### Phase 4 — Non-negative least squares

1. Add the active-set NNLS solver over the existing Cholesky path.
2. Register it as a bound capability under the Spec 26 solver capability table.
3. Add it as the simplest bounds test case.

### Phase 5 — Group and sparse-group lasso

1. Add group metadata to the design view.
2. Add group soft thresholding to CCD.
3. Add sparse-group lasso.
4. Validate group definitions and add grouped-structure tests.

### Phase 6 — Negative binomial and beta regression

1. Add `NegativeBinomial` to the GLM family set with dispersion estimation.
2. Add `Beta` to the family set for bounded continuous outcomes.
3. Add dispersion reporting and a Poisson-dispersion test.

### Phase 7 — Conditional GAM work

1. Improve MARS knot placement and regularization.
2. Benchmark against a penalized-spline prototype.
3. Add a GAM leaf only if it wins on accuracy or stability at equal interpretability.

---

## 16. Evaluation Requirements

Every regression addition needs:

- **Correctness against a reference.** Compare against an independent implementation on fixed data, not against a hand-computed single case.
- **Recovered structure.** Synthetic data with known conditional quantiles must recover them.
- **Weight semantics.** Case weights must scale both loss and gradient identically, and the weighting convention must be documented per method.
- **Missing data.** Every method follows the package NA policy. No silent imputation.
- **Determinism.** Same seed, same input, same output.
- **Small-leaf behavior.** Report what happens when a leaf has fewer observations than parameters. Prefer an explicit failure or a documented fallback over a silent singular solve.
- **Held-out performance.** A lower training loss must not hide overfitting. Report held-out RMSE, pinball loss, or deviance as appropriate.

---

## 17. Key Risks

1. **Quantile crossing.** Multiple fitted taus can cross. Enforce monotonicity or report crossings.
2. **Small-leaf instability.** Quantile, Huber scale, and dispersion estimates all degrade in small leaves. Minimum-support rules are mandatory.
3. **Asymmetric-loss inference.** Standard OLS coefficient errors do not apply to quantile or expectile fits. Do not report them.
4. **Duplicate families.** A GAM that only renames MARS adds maintenance cost and no capability.
5. **Group-definition ambiguity.** A wrong group structure silently changes the estimator.
6. **Family-parameter confusion.** Negative binomial NB1 and NB2 parameterizations must not be mixed.
7. **Memory in NNLS.** The active-set path needs $O(P^2)$ storage. Fine for leaves, not for wide standalone designs.
8. **Benchmark oracles missing.** Without RANSAC, Theil-Sen, and external references, robust-regression improvements cannot be validated.

---

## 18. Primary Sources

1. Koenker, R. and Bassett, G. (1978). "Regression Quantiles." *Econometrica* 46(1), 33-50. [DOI](https://doi.org/10.2307/1913643).
2. Koenker, R. (2017). "Quantile Regression: 40 Years On." *Annual Review of Statistics and Its Application* 5, 985-1011. [DOI](https://doi.org/10.1146/annurev-economics-063016-103651).
3. Koenker, R. and Hallock, K. F. (2001). "Quantile Regression." *Journal of Economic Perspectives* 15(4), 143-156. [DOI](https://doi.org/10.1257/jep.15.4.143).
4. Huber, P. J. (1964). "Robust Estimation of a Location Parameter." *Annals of Mathematical Statistics* 35(1), 73-101. [DOI](https://doi.org/10.1214/aoms/1177703732).
5. Bellec, P. C. et al. (2025). "Error Estimation and Adaptive Tuning for Unregularized Robust M-Estimator." *JMLR* 26, 1-40. [Paper](https://jmlr.org/papers/volume26/24-0060/24-0060.pdf).
6. Lawson, C. L. and Hanson, R. J. (1995). *Solving Least Squares Problems*. SIAM. [DOI](https://doi.org/10.1137/1.9781611971217).
7. Yuan, M. and Lin, Y. (2006). "Model Selection and Estimation in Regression with Grouped Variables." *JRSS-B* 68(1), 49-67. [DOI](https://doi.org/10.1111/j.1467-9868.2005.00578.x).
8. Simon, N., Friedman, J., Hastie, T., and Tibshirani, R. (2013). "A Sparse-Group Lasso." *Journal of Computational and Graphical Statistics* 22(2), 337-359. [DOI](https://doi.org/10.1080/10618600.2012.681250).
9. Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*. Chapman and Hall.
10. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*. Chapman and Hall.
11. Pya, N. and Wood, S. N. (2015). "Shape Constrained Additive Models." *Statistics and Computing* 25, 543-556. [DOI](https://doi.org/10.1007/s11222-013-9448-7).
12. Cameron, A. C. and Trivedi, P. K. (2013). *Regression Analysis of Count Data*. Cambridge University Press. [DOI](https://doi.org/10.1017/CBO9781139013567).
13. Zeileis, A., König, B., and Wooldridge, J. M. (2026). "Negative-Binomial, Hurdle, and Zero-Inflation with 'fastglm'." *R Journal* vignette. [Vignette](https://cloud.r-project.org/web/packages/fastglm/vignettes/count-firth-fastglm.html).
14. Smith, R. L. (1979). "Matrix Partial Least Squares." *Journal of Chemometrics* 33(1), 3-8. [DOI](https://doi.org/10.1002/cem.3800010102).
15. Friedman, J. H. (1991). "Multivariate Adaptive Regression Splines." *Journal of the American Statistical Association* 86(415), 1-25. [DOI](https://doi.org/10.1080/01621459.1991.758768).
16. Lundberg, S. M. and Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
17. Chen, A. et al. (2026). "Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction." *ICML*, PMLR 306. [Paper](https://proceedings.mlr.press/v306/chen26bm.html).
18. Duchemin, Q. and Obozinski, G. (2025). "Efficient Distributional Regression Trees Learning Algorithms for Calibrated Non-Parametric Probabilistic Forecasts." [arXiv:2502.05157](https://arxiv.org/abs/2502.05157).
19. Bendersky, M. and Fridkin, S. (2026). "The Evolving Landscape of Modern Regression: A Systematic Review and Mapping Study of Penalized, Algorithmic, and Interpretable Models." *WIREs Data Mining and Knowledge Discovery* 16(1), e70078. [DOI](https://doi.org/10.1002/widm.70078).
20. Billa, M. et al. (2026). "A Comparative Analysis of Interpretable Machine Learning Methods." [arXiv:2601.00428](https://arxiv.org/abs/2601.00428).
21. scikit-learn. "Linear Models." [Official documentation](https://scikit-learn.org/stable/modules/linear_model.html).
22. R `mgcv` package. [CRAN](https://cran.r-project.org/web/packages/mgcv/).
23. R `scam` package. [CRAN](https://cran.r-project.org/web/packages/scam/).
24. `solow-gam` crate. [crates.io](https://crates.io/crates/solow-gam).

## 19. Source Notes

- alphaXiv discovery supplied current candidates in distributional regression, Bayesian additive regression trees, sparse-group penalties, and quantile-regression variants.
- Primary papers, official SciPy, scikit-learn, statsmodels, and CRAN documentation supplied algorithm mechanics and reference behavior.
- Rust crate registries supplied reuse options and adoption cautions.
- Current source was read directly for the repository state section. Where older notes disagree with current source, current source wins.
- treg catalog exposed Perplexity, Brave, and Scholar endpoints, but live treg calls failed because saved authentication expired. No claim in this note depends on a failed treg response.
