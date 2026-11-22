# Spec 14: Cubist Committees, LinearForest & Model-Tree Forests

> **Status: Superseded by Spec 28 (`ModelForest` — the bagging family).**
>
> This specification is retained as a historical design record. Do not build from it. Its scope lives in `specs/28-model-forest-bagging-family.md`, with three corrections:
>
> 1. **The `m0` smoothing constant does not exist.** A grep across the Cubist GPL source returns zero hits for `m0` or `M0`. There are two distinct smoothing algorithms: Quinlan's M5 backup smoothing $p' = (n p + k q)/(n+k)$ with $k = 15$, and Cubist's variance-minimising coefficient shrink against the parent model. Spec 28 names them separately.
> 2. **A Cubist committee is not boosting.** Member $m \ge 2$ trains on the relabelled target $y' = 2y - \hat y_{m-1}$ and the final prediction is an **unweighted average** over members. There is no learning rate, no stage weights, and no row resampling. The effective correction rate is $1/2$ with no decay across members.
> 3. **The extrapolation default differs between distributions.** The RuleQuest CLI default is 5 percent; the CRAN `Cubist` default is 100 percent. Both are documented in Spec 28.
>
> Verified reference: `references/research/2026-11-22-forest-methods-landscape.md` and `references/research/2026-11-22-ensemble-families-and-api-architecture.md`.

## Problem Statement

While single model trees (such as M5' and `mltreelib`'s `ModelTree`) fit piecewise-linear surfaces that drastically outperform constant-leaf CART on smooth functional data, individual model trees suffer from three major vulnerabilities:
1. **High Variance & Step Discontinuities**: Splits create hard boundaries where adjacent linear models can produce discontinuous jump predictions for points near the partition margin.
2. **Extrapolation Instability**: In sparsely populated leaves, unregularized OLS coefficients can predict wild, unphysical values when evaluating test points slightly outside the training convex hull.
3. **Sub-optimal Additive Accuracy**: A single tree cannot easily model additive structures without splitting repeatedly, whereas boosting and bagging ensembles reduce bias and variance respectively.

Quinlan's original Cubist engine achieved state-of-the-art tabular regression performance through three distinct innovations: **Committees** (iterative residual-weighted tree boosting), **Hierarchical Model Smoothing** (blending leaf linear models with ancestral models up the tree), and **Composite Instance Adjustments** ($k$-NN residual correction). In parallel, recent literature introduces **LinearForest** (Random Forests with regularized linear leaves). None of these currently exist in a modern, performant, pure-Rust engine.

## Solution

Implement a complete model-tree ensembling and post-processing suite in `mltreelib-core`:

1. **Cubist Committees (`CubistRegressor`)**:
   - Sequential ensemble of $M$ model trees (default $M=5$ to $20$).
   - Committee member $m$ is trained to predict the adjusted pseudo-residuals $r_i^{(m)} = y_i - \hat{y}_i^{(1 \dots m-1)}$.
   - Sample cases with high residuals receive higher sample weights in subsequent committee rounds.
   - Final committee prediction is the unweighted average or median of all $M$ committee predictions: $\hat{y}(x) = \frac{1}{M}\sum_{m=1}^M \hat{y}^{(m)}(x)$.
2. **Hierarchical Model Tree Smoothing**:
   - For any leaf $L$ with sample size $n_L$ and linear model weights $\mathbf{w}_L$, recursively blend with parent model weights $\mathbf{w}_P$:
     $$\mathbf{w}' = \frac{n_L \mathbf{w}_L + m_0 \mathbf{w}_P}{n_L + m_0}$$
     where $m_0$ is a configurable smoothing hyperparameter (default $m_0 = 15.0$).
   - Propagates smoothing from the root down to every terminal leaf, eliminating boundary step discontinuities.
3. **Extrapolation Clamping Policy**:
   - Optional bounding: predictions in leaf $L$ are restricted to $[y_{\min}^{(L)} - \alpha R_L, y_{\max}^{(L)} + \alpha R_L]$, where $R_L = y_{\max}^{(L)} - y_{\min}^{(L)}$ and default $\alpha = 0.05$.
   - Non-negativity constraint: if all training targets in a leaf are strictly non-negative, predictions are clamped at $0$.
4. **Composite Models ($k$-NN Local Correction)**:
   - After leaf model prediction $\hat{y}(x)$, retrieve $K$ nearest neighbors (default $K=3$) from training data in normalized feature space.
   - Adjust final prediction by distance-weighted average training residual: $\hat{y}_{\text{final}}(x) = \hat{y}(x) + \sum_{k=1}^K w_k (y_k - \hat{y}(x_k))$.
5. **LinearForest (`LinearForestRegressor`)**:
   - Bagged ensemble of $B$ model trees with bootstrap sampling and random feature sub-spacing (`max_features`).
   - Terminal leaves fit Ridge-regularized linear models (`faer::llt`) using features selected along the branch or global features.
   - Parallelized tree training via `rayon`.

## User Stories

1. As a tabular data scientist, I want `CubistRegressor(n_committees=10, smooth=True)`, so that I can train boosted model trees matching or exceeding proprietary RuleQuest Cubist accuracy.
2. As a machine learning engineer, I want `LinearForestRegressor(n_estimators=100, max_features="sqrt", leaf="linear")`, so that I can run high-performance bagged model-tree forests.
3. As a financial modeler, I want extrapolation clamping (`extrapolation_limit=0.05, non_negative=True`), so that model predictions never output negative asset prices or absurd tail values.
4. As a quantitative analyst, I want hierarchical model smoothing (`smooth_factor=15.0`), so that continuous predictions smoothly transition across split boundaries without step jumps.
5. As a practitioner working with complex non-linear manifolds, I want composite $k$-NN adjustment (`composite_neighbors=5`), so that local training residuals correct systematic model bias.
6. As a performance engineer, I want `CubistRegressor.fit()` to parallelize committee evaluation and leaf OLS inversions, executing within $2\times$ the time of a single tree.
7. As a data scientist, I want `cubist.feature_importances_`, computing split usage and leaf linear model coefficient usage across all committees.
8. As a production engineer, I want individual committee models to be exportable to human-readable rule sets (`if X1 > 3.2 and X2 <= 1.0 then Y = 0.4*X1 + 1.2`), preserving interpretability.
9. As a researcher, I want out-of-bag (OOB) error estimation during `LinearForest` training, so that cross-validation is achieved without redundant training runs.
10. As an econometrician, I want `LinearForest` to support sample weights, allowing survey weights or inverse propensity weights to weight both tree splits and leaf ridge inversions.
11. As an API user, I want `CubistRegressor` to follow the standard `fit(X, y)` and `predict(X)` interface with zero scikit-learn dependency.
12. As a systems developer, I want all leaf linear matrices and $k$-NN search indices to use flat cache-aligned buffers, avoiding fragmented pointer indirection.

## Implementation Decisions

- **Crate Module Structure**:
  - Add module `crates/mltreelib-core/src/ensemble/`:
    - `committee.rs`: `CubistEnsemble` managing sequential committee training, residual updates, and ensemble averaging.
    - `smooth.rs`: Hierarchical tree smoothing traversing ancestor paths and blending linear weight vectors.
    - `clamping.rs`: Leaf range boundary checks and non-negative clamping logic.
    - `composite.rs`: Flat $k$-d tree or ball tree index for local training neighbor lookup and residual correction.
    - `linear_forest.rs`: `LinearForest` managing bootstrap sample generation, random subspace selection, and `rayon` parallel tree fitting.
- **Hierarchical Smoothing Algorithm**:
  ```rust
  pub fn smooth_node_weights(node: &Node, parent_weights: &[f64], m0: f64) -> Vec<f64> {
      let n = node.n_samples as f64;
      let w_leaf = &node.leaf_weights;
      let blended: Vec<f64> = w_leaf.iter().zip(parent_weights.iter())
          .map(|(&w_l, &w_p)| (n * w_l + m0 * w_p) / (n + m0))
          .collect();
      blended
  }
  ```
- **Python Surface (`crates/mltreelib-py`)**:
  - `mltreelib.ensemble.CubistRegressor(n_committees=5, smooth=True, smooth_factor=15.0, extrapolation_limit=0.05, composite_neighbors=0, min_rows=10, max_depth=10, l2_reg=1e-3, random_state=None)`
  - `mltreelib.ensemble.LinearForestRegressor(n_estimators=100, max_features="sqrt", bootstrap=True, max_depth=10, min_rows=5, l2_reg=1e-3, n_jobs=-1, random_state=None)`

## Testing Decisions

- **Official RuleQuest Parity Test**:
  - Run `CubistRegressor` on benchmark datasets (`boston.csv`, `concrete.csv`, `abalone.csv`).
  - Verify committee test RMSE is $\ge 15\%$ lower than a single ModelTree and achieves correlation $r \ge 0.95$ matching original RuleQuest Cubist binary benchmarks.
- **Step Discontinuity Smoothing Test**:
  - Generate 1D step function with linear trends: $y = x + (x > 0.5) \cdot 2$.
  - Evaluate prediction curve over fine grid $x \in [0.45, 0.55]$.
  - Verify smoothed model eliminates infinite gradient jump at $x = 0.5$, producing a monotone transition.
- **Extrapolation Clamping Verification**:
  - Train model on $y \in [10, 50]$. Evaluate at extreme outliers $x = 10^6$.
  - Verify predictions are strictly bounded within $[10 - 0.05 \cdot 40, 50 + 0.05 \cdot 40] = [8, 52]$.
- **Composite $k$-NN Correction Test**:
  - Train model with deliberate non-linear residual ripple $y = x_1 + \sin(10 x_2)$.
  - Verify that enabling `composite_neighbors=5` reduces test residual variance by $\ge 30\%$ compared to uncorrected linear leaves.

## Out of Scope

- Rule winnowing pre-selection across millions of candidate interaction terms (handled separately in the `preselect` catalog).
- Multi-target simultaneous vector committees (each target fits its own committee chain).

## Further Notes

- Quinlan's Cubist is widely regarded as one of the highest-performing tabular regression algorithms ever built, frequently outperforming XGBoost on smooth physical and engineering problems. A native Rust implementation provides modern speed while honoring Quinlan's exact inductive biases.
