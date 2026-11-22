# Spec 09: Causal and Uplift Trees

## Problem Statement

When analyzing interventions—such as marketing campaigns, clinical trials, discount pricing, or policy changes—practitioners need to estimate Heterogeneous Treatment Effects (HTE) or the Conditional Average Treatment Effect (CATE): $\tau(X) = E[Y(1) - Y(0) \mid X]$.

Standard decision trees fail catastrophically because they partition data to predict the outcome $Y$, not the differential treatment effect $\tau$. This confounds predictive features with effect-modifying features. Existing Python and R tooling (`grf`, `causalTree`, `EconML`, `CausalML`) is heavily fragmented, suffers from slow execution, forces heavy dependencies (`scikit-learn`, `numba`, R toolchains), and either forces constant leaf effects (step-function bias) or lacks robustness against propensity score misspecification (missing double robustness).

## Solution

Provide a native, pure-Rust causal and uplift modeling framework in `mltreelib` with zero scikit-learn dependency:
1. **Local Linear CATE Leaves**: Fit multivariate treatment effect slopes $\tau(X) = \alpha + \beta^T X$ inside leaves, eliminating boundary attenuation and step-function discretization.
2. **Doubly Robust (AIPW) Leaves & Splits**: Utilize Augmented Inverse Probability Weighting (AIPW) scores $\Gamma_i$, providing double robustness (unbiased even if either the propensity model or the conditional outcome model is misspecified).
3. **R-Learner (Robinson Residual) Leaves & Splits**: Regress residualized outcomes $(Y - \hat{m}(X))$ on residualized treatments $(W - \hat{e}(X))$ with Neyman-orthogonal weighting, bypassing regularisation bias.
4. **Instrumental Variables (IV / 2SLS) Trees**: Implement Wald/2SLS leaf estimators and covariance splitting to estimate Local Average Treatment Effects (LATE) under unmeasured confounding.
5. **Regression Discontinuity (RD / RDD) Trees**: Estimate heterogeneous treatment effects at threshold boundaries.
   - **Sharp RD**: Fit centered local linear slopes on either side of cutoff $c$ on running variable $R$, estimating $\tau(X) = (\alpha^+ - \alpha^-) + (\beta^+ - \beta^-)^T (X - c)$.
   - **Fuzzy RD**: Modeled as 2SLS / IV with cutoff indicator $Z = \mathbf{1}(R \ge c)$ instrumenting endogenous treatment adoption $W$.
   - **RD Tree Splitting**: Partition remaining covariates $X_{-R}$ to find subgroups where the discontinuity jump is largest.
6. **Difference-in-Differences (DiD) Trees**: Estimate heterogeneous treatment effects $\tau(X)$ across pre/post policy windows, integrating with Spec 27's panel fixed effects.
7. **Divergence-Based Uplift Splitting**: Support Kullback-Leibler (KL), Euclidean Distance (ED), and Chi-Square ($\chi^2$) divergence criteria for binary and multi-treatment campaign targeting.

## User Stories

1. As an econometrician, I want `gain="causal_honest"` with sample-splitting, so that my CATE estimates and confidence intervals remain asymptotically normal and unbiased.
2. As a marketing data scientist, I want `gain="uplift_kl"` and `gain="uplift_ed"`, so that I can directly optimize the separation between responders, persuadables, and do-not-disturbs.
3. As a quantitative researcher, I want `leaf="linear_cate"`, so that the treatment effect inside each partition is allowed to vary linearly with continuous covariates instead of being forced to a flat average.
4. As a machine learning engineer, I want `leaf="doubly_robust"`, so that my treatment effect predictions are protected against misspecification of either the propensity score or the baseline outcome models.
5. As a causal inference practitioner, I want to pass precomputed propensity scores `propensity=probs` to `fit()`, so that first-stage cross-fitted propensity estimates can drive double-robust splitting without recalculation.
6. As a policy researcher, I want `gain="r_learner"`, so that I can train Neyman-orthogonal causal trees on residualized outcomes and residualized treatment indicators.
7. As an applied microeconomist, I want instrumental variable splitting (`gain="iv"`, `instrument=Z`) and 2SLS leaf estimation (`leaf="iv_2sls"`), so that I can estimate treatment effects when unobserved confounders violate unconfoundedness.
8. As a policy researcher, I want Regression Discontinuity trees (`design="rd_sharp"` and `design="rd_fuzzy"` with running variable $R$ and cutoff $c$), so that I can estimate heterogeneous treatment effects around eligibility cutoffs.
9. As a pricing analyst, I want continuous treatment leaves (`treatment_type="continuous"`), so that I can estimate price elasticity curves $\frac{\partial E[Y \mid W=w, X=x]}{\partial w}$ across market segments.
9. As an uplift modeler, I want multi-treatment uplift trees (`treatment=[0, 1, 2]`), so that I can compare multiple competing discount offers against a control holdout simultaneously.
10. As a production engineer, I want `tree.predict_uplift(X)` to run in sub-millisecond Rust time, so that live bidding and real-time intervention systems can evaluate causal uplift without latency bottlenecks.
11. As an auditor, I want `leaf.summary()` on causal leaves to report standard errors and $t$-statistics for treatment coefficients $\tau$, so that I can verify statistical significance per segment.
12. As a maintainer, I want NA values in covariates $X$ to route through surrogate splits during causal scoring, so that missingness never causes drops in causal evaluation rows.

## Implementation Decisions

- **Causal Auxiliary Inputs**:
  The tree `fit` API accepts treatments $W$, optional propensity scores $e(X)$, and optional instruments $Z$ through unified auxiliary arrays or a structured tuple `(Y, W, e, Z)`.
- **Criteria Additions**:
  - `GainFn::CausalHonest { alpha, lambda }`: Variance of CATE minus leaf variance penalty.
  - `GainFn::DoublyRobust`: Weighted SSE reduction on AIPW pseudo-outcomes $\Gamma_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{W_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-W_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}$.
  - `GainFn::RLearner`: SSE reduction on $\frac{\tilde{Y}_i}{\tilde{W}_i}$ with weights $\tilde{W}_i^2$.
  - `GainFn::UpliftKl`, `GainFn::UpliftEd`, `GainFn::UpliftChi`: Distributional divergence criteria comparing $P(Y \mid W=1)$ vs $P(Y \mid W=0)$.
  - `GainFn::Iv`: Wald covariance divergence $\frac{(\text{Cov}_L(Y, Z))^2}{\text{Var}_L(Z)} + \frac{(\text{Cov}_R(Y, Z))^2}{\text{Var}_R(Z)}$.
- **Leaf Additions**:
  - `LeafKind::LinearCATE`: Solves bivariate or multivariate linear system estimating base coefficients $\beta_0$ and interaction treatment coefficients $\tau$.
  - `LeafKind::DoublyRobust`: Fits linear regression directly on AIPW scores $\Gamma_i$.
  - `LeafKind::Iv2SLS`: Two-Stage Least Squares solver inside terminal leaves using instrument matrix $Z$.
  - `LeafKind::RdLocalLinear`: Local linear regression centered at threshold $c$ on running variable $R$, estimating boundary jump $\tau(c)$.
- **Quasi-Experimental Seam**:
  - Sharp RD maps to centered `LinearCATE` on observations within bandwidth $h$: $|R - c| \le h$.
  - Fuzzy RD maps to `Iv2SLS` with instrument $Z_i = \mathbf{1}(R_i \ge c)$ on endogenous treatment $W_i$.
  - RD Tree splitting optimizes heterogeneity across other features $X_{-R}$, isolating which subgroups experience the largest policy jump at cutoff $c$.
- **High Seam**:
  All causal models interface through `ModelTree` with `predict_uplift(X)` and `predict_cate(X)` returning treatment effect arrays.

## Testing Decisions

- **Behavioral Tests**:
  - Planted heterogeneous treatment effect benchmark (e.g. IHDP or synthetic $Y = X_0 + W(5 X_1) + \epsilon$): verify tree splits on $X_1$ and recovers $\tau(X) \approx 5 X_1$.
  - Double robustness test: corrupt the propensity score while keeping the outcome model correct, and vice versa; verify that CATE estimation remains unbiased.
  - IV simulation test: generate unobserved confounder $U$ affecting both $W$ and $Y$, with valid instrument $Z$ affecting $W$ only; verify standard OLS tree is biased while IV tree recovers true treatment effect.
  - Uplift divergence test: verify `uplift_kl` separates positive responders from negative responders.

## Out of Scope

- Automated global propensity neural network fitting (propensity scores can be provided precomputed or estimated via logistic stump).
- Dynamic multi-stage reinforcement learning trees (offline contextual bandits are supported via DR leaves, but full MDP policy iteration is out of scope).

## Further Notes

- Aligns directly with Athey, Tibshirani, and Wager (2019) *Generalized Random Forests* and Nie & Wager (2021) *Quasi-Oracle Estimation of Heterogeneous Treatment Effects*.
