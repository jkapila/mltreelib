# Spec 12: Exotic Statistical and Metric Splitting Criteria

## Problem Statement

Standard tree-building algorithms (CART, C4.5, LightGBM, XGBoost) suffer from three deep theoretical limitations:
1. **Selection Bias (Cardinality Trap)**: Variables with many distinct values or high category cardinality have substantially more candidate split points, giving them artificially high chances of being chosen purely due to multi-testing noise (Loh 2002; Hothorn et al. 2006).
2. **Computational Bottleneck of Refitting in Model Trees**: In model trees (like M5 or GLM trees), refitting a multivariate model for every candidate threshold across all variables requires solving thousands of linear systems per node, which becomes prohibitive as feature dimensionality grows.
3. **Inability to Handle Complex/Metric Targets**: When the outcome $Y$ is a multi-dimensional embedding vector, a probability histogram, an image representation, or a point cloud, scalar variance reduction is undefined.
4. **Unconstrained Fairness Violations**: Standard impurity reduction greedily creates segments that maximize predictive performance while potentially introducing severe disparate impact on legally protected sub-populations.

Existing packages (`partykit::mob`, `partykit::ctree`, optimal transport libraries) are isolated in R or require heavy specialized Python environments without unified integration.

## Solution

Implement four advanced statistical and metric split paradigms in `mltreelib-core`:
1. **Model-Based Recursive Partitioning (MOB / Zeileis et al. 2008)**:
   Fit the leaf model **once** at the parent node. Compute the individual score residuals $\psi_i = \nabla_\theta \ell_i(\hat{\theta})$. Test for parameter instability along candidate covariates using cumulative sum (supLM) fluctuation tests. Only the variable showing statistically significant instability ($p < \alpha$) is split, bypassing candidate-level model refitting.
2. **Conditional Inference Framework (CTree / Hothorn et al. 2006)**:
   Separate variable selection from splitpoint selection using permutation rank hypothesis testing with linear rank statistics $T_j$. Variable selection is **strictly unbiased** against variable scale and cardinality.
3. **Maximum Mean Discrepancy (MMD) & Energy Distance Metric Splitting**:
   Support multi-output, high-dimensional vector, and manifold targets by maximizing the kernel two-sample test statistic or Energy Distance between child node distributions without parametric distributional assumptions.
4. **Wasserstein-2 Optimal Transport Splitting**:
   Partition nodes where each observation's target $Y_i$ is a histogram or continuous probability distribution, maximizing Wasserstein-2 dispersion reduction around Fréchet barycenters.
5. **Fairness-Constrained Splitting (`gain="fair"` / `demographic_parity_penalty`)**:
   Penalize node split gain by the demographic disparity divergence across protected attributes $S$, directly producing Pareto-optimal fair decision trees.

## User Stories

1. As an econometrician, I want `split_test="mob"`, so that the tree tests for parameter structural breaks using score residuals without refitting multivariate regressions across every candidate cutpoint.
2. As a survey data scientist, I want `gain="ctree"`, so that high-cardinality demographic factors (e.g., zip code with 1,000 levels) are not unfairly favored over low-cardinality binary factors.
3. As a computer vision researcher, I want `gain="mmd"`, so that I can train decision trees directly on multi-dimensional feature embeddings ($Y \in \mathbb{R}^D$).
4. As a data scientist handling complex distributions, I want `gain="energy_distance"`, so that multi-target split optimization accurately captures multi-modal clustering without computing heavy covariance matrices.
5. As an environmental scientist, I want `gain="wasserstein"`, so that trees can split samples whose targets are regional daily temperature distributions or pollution histograms.
6. As a compliance officer, I want `fairness_penalty=0.2` and `protected_attribute="gender"`, so that credit approval decision trees optimize accuracy subject to strict demographic parity bounds.
7. As an auditor, I want the tree inspection to output the asymptotic $p$-values of the score fluctuation tests for all candidate variables at each split, so that I can verify statistical justification for branching.
8. As a system engineer, I want MMD kernel matrices to evaluate in parallel with $O(N)$ rank-1 approximations where sample sizes are large, so that metric splitting scales to hundreds of thousands of rows.
9. As a developer, I want all exotic criteria to be usable with existing `LeafKind` options (mean, linear, huber, quantile), so that split logic remains fully orthogonal to leaf logic.
10. As a package maintainer, I want CTree and MOB tests to produce exact deterministic $p$-values under fixed random seeds or asymptotic normal approximations.

## Implementation Decisions

- **Score-Based Split Engine (`node_stats.rs` & `split.rs`)**:
  - Adds `score_residuals(model, rows)` returning $[\psi_{i, 1}, \dots, \psi_{i, P}]$.
  - Evaluates the cumulative empirical fluctuation process $W(t) = \hat{V}^{-1/2} n^{-1/2} \sum_{i=1}^{\lfloor nt \rfloor} \psi_{(i)}$ along sorted values of candidate covariate $X_j$.
  - Computes the supLM statistic: $\max_{t \in [\underline{t}, \bar{t}]} \frac{\|W(t)\|^2}{t(1-t)}$.
- **Permutation Rank Engine (`gate.rs` & `stats.rs`)**:
  - Implements asymptotic multivariate normal transformation and Bonferroni / Holm multiple testing adjustments for CTree.
- **Metric Criteria Additions (`criteria.rs`)**:
  - `GainFn::Mmd { kernel: KernelKind, bandwidth: f64 }`
  - `GainFn::EnergyDistance`
  - `GainFn::Wasserstein2`
  - `GainFn::FairConstrained { base_gain: Box<GainFn>, lambda: f64, protected_col: usize }`
- **High Seam**:
  Configurable via `ModelTree(gain="mob" | "ctree" | "mmd" | "energy_distance" | "wasserstein", fairness_penalty=...)`.

## Testing Decisions

- **Behavioral Tests**:
  - Cardinality bias test (Loh's benchmark): verify that when independent random noise variables have 2, 10, 50, and 100 levels, CTree selects each variable with equal probability (~25%), unlike CART which selects the 100-level variable >90% of the time.
  - MOB structural break test: generate linear regression where slope changes at $X_0 = 0$; verify MOB fluctuation test detects instability on $X_0$ and splits at 0 without candidate refits.
  - MMD multi-output test: generate two 10-dimensional Gaussian clusters with identical means but orthogonal covariances; verify MMD tree separates the regimes.
  - Demographic parity test: verify that adding `fairness_penalty > 0` progressively narrows the selection rate gap between protected groups while preserving tree structure.

## Out of Scope

- Non-Euclidean Riemannian manifold geodesic solvers for arbitrary Riemannian metrics (RBF and Energy metrics are supported).
- Multi-party federated secure multiparty computation splitting.

## Further Notes

- Bridges classical recursive partitioning with Zeileis, Hothorn & Hornik (2008) *Model-Based Recursive Partitioning* (J. Comp. Graph. Stat.) and Hothorn, Hornik & Zeileis (2006) *Unbiased Recursive Partitioning* (JCGS).
