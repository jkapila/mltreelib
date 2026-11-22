# Spec 15: Structured Tree Ensemble Methods

> **Status: Superseded by Specs 28, 29, 30, and 31.**
>
> This specification is retained as a historical design record. Do not build from it. Its scope is split:
>
> - **Bagging, honest forests, and the Cubist committee** → `specs/28-model-forest-bagging-family.md`
> - **Boosting and conformal intervals** → `specs/29-model-boost-boosting-family.md`
> - **Stacking and the combination-rule layer** → `specs/30-model-meta-estimators.md`
> - **The family-selection framework** → `specs/31-ensemble-family-selection-guide.md`
>
> Four corrections to the content below:
>
> 1. **The family is not one class.** A single `kind` switch over forests, committees, boosters, and stacks would either accept invalid parameter combinations or hide the family dispatch. The families are separate public classes over one shared Rust module.
> 2. **The Generalized Random Forest still needs its own specification.** Its splitting rule needs a node-level relabeling strategy, which `GainFn::Custom` cannot express. Spec 28 defers it explicitly rather than half-building it.
> 3. **Conformal intervals are finite-sample valid, but the guarantee is narrow.** They are distribution-free and assume exchangeability. They are not a substitute for a posterior interval, which is exact under its model.
> 4. **Honesty and double robustness fix different biases.** Honesty removes adaptive bias from selecting splits on the same outcomes averaged into leaves. Orthogonalisation removes confounding bias. Neither substitutes for the other.
>
> Verified reference: `references/research/2026-11-22-forest-methods-landscape.md`.

## Problem Statement

Single decision trees have high sampling variance and non-smooth piecewise-constant predictions. While standard Random Forests and GBDTs (such as `scikit-learn`, `xgboost`, or `lightgbm`) address variance and bias, they are severely limited for advanced structured learning:
1. **Lack of Honesty & Valid Inference**: Standard random forests reuse the same data points to select split thresholds and estimate leaf parameters. This causes adaptive overfitting, invalidating statistical inference and confidence interval coverage in econometrics and causal analysis.
2. **Inflexible Leaf Models**: Mainstream tree ensembles only support scalar mean outputs or gradient-step constants; they cannot ensemble model trees, distributional leaves, causal leaves, or survival curves.
3. **Absence of Calibrated Uncertainty**: GBDTs provide point predictions without finite-sample conformal prediction bounds or calibrated uncertainty intervals.
4. **Heavy External Dependencies**: Existing generalized ensemble packages (such as R's `grf` or Python's `econml`) rely heavily on C++ bindings, Fortran, or heavyweight Python runtime stacks, preventing unified high-performance deployment.

## Solution

Build a unified, modular ensemble engine in `mltreelib-core` supporting modern structured forest paradigms with zero external machine learning dependencies:

1. **Honest Random Forests (Generalized Random Forests - GRF)**:
   - Implement sample splitting ("honesty"): for each tree, partition the training subsample into split-determining instances $\mathcal{I}_{\text{split}}$ and leaf-evaluating instances $\mathcal{I}_{\text{leaf}}$.
   - Guarantees that conditional leaf estimators $\hat{\theta}(x)$ are asymptotically unbiased, Gaussian, and support closed-form confidence intervals.
2. **Adaptive Nearest-Neighbor Kernel Weighting**:
   - Compute forest-induced similarity weights:
     $$\alpha_i(x) = \frac{1}{B} \sum_{b=1}^B \frac{\mathbf{1}(X_i \in L_b(x))}{|L_b(x)|}$$
   - Enables solving any localized optimization problem at test time (Local Linear Regression, Local Quantile, Local CATE).
3. **Structured Gradient Boosting (ModelTreeBoost)**:
   - Implement second-order Newton-Raphson gradient boosting:
     $$g_i = \frac{\partial \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$$
   - Support weak learners as shallow model trees with linear leaves, accelerating convergence on smooth functional manifolds.
   - Support all newly implemented distributional loss functions (Tweedie deviance, Tobit log-likelihood, heteroscedastic Gaussian loss, Martingale survival loss).
4. **Sub-bagging & Random Subspaces**:
   - Sub-bagging (subsampling without replacement at rate $\gamma \in (0.5, 0.8)$) to preserve independent sampling asymptotics.
   - Column sub-sampling per tree (`colsample_bytree`) and per split node (`colsample_bynode`).
5. **Out-of-Bag (OOB) Conformal Uncertainty Bands**:
   - Track per-sample OOB predictions during forest training.
   - Compute exact non-conformity scores $s_i = |y_i - \hat{y}_{\text{oob}}(x_i)|$.
   - Produce distribution-free, finite-sample valid prediction intervals at target coverage level $1 - \alpha$:
     $$\hat{C}_{1-\alpha}(x) = [\hat{y}(x) - \hat{q}_{1-\alpha}, \hat{y}(x) + \hat{q}_{1-\alpha}]$$
6. **Multi-Model Stacking (Super Learner)**:
   - $K$-fold cross-validated out-of-fold meta-dataset generation.
   - Constrained Non-Negative Least Squares (NNLS) meta-solver ($\sum w_k = 1, w_k \ge 0$), provably asymptotically matching or outperforming the best single base learner.

## User Stories

1. As an econometrician, I want `HonestForestRegressor(honest=True, honest_ratio=0.5)`, so that my leaf estimates produce valid asymptotic standard errors and $95\%$ confidence intervals.
2. As a risk analyst, I want `forest.predict_interval(X, alpha=0.05)`, so that I receive finite-sample conformal prediction bounds with guaranteed $95\%$ marginal coverage.
3. As a machine learning practitioner, I want `ModelTreeBoostRegressor(n_estimators=100, learning_rate=0.1, leaf="linear")`, so that boosted shallow linear model trees converge faster than standard GBDT on continuous physical features.
4. As an actuarial scientist, I want `ModelTreeBoostRegressor(loss="tweedie", tweedie_p=1.5)`, so that I can train gradient boosted models directly on zero-inflated insurance claims.
5. As an empirical researcher, I want `forest.get_leaf_weights(query_x)`, extracting the exact sparse weights $\alpha_i(x)$ over all training points, enabling custom localized statistical testing.
6. As a data scientist, I want `colsample_bytree=0.8` and `subsample=0.7`, reducing tree correlation and speeding up training on high-dimensional datasets.
7. As a quantitative developer, I want `StackingRegressor(estimators=[cart, cubist, linear_forest], cv=5)`, so that ensemble stacking automatically blends diverse tree inductive biases without manual tuning.
8. As a production engineer, I want tree fitting to utilize `rayon::par_iter`, distributing tree construction across all CPU cores with near-linear scaling.
9. As a causal inference researcher, I want `HonestCausalForest`, combining honest sample splitting with causal difference-in-means and doubly-robust leaves.
10. As an auditor, I want `forest.oob_score_`, reporting unbiased validation metrics directly upon completion of training without requiring a separate holdout set.
11. As a systems architect, I want memory consumption during ensemble training to be bounded via batched tree generation, preventing out-of-memory errors on large datasets.
12. As a Python developer, I want all ensemble estimators to conform to the standard `fit(X, y)` and `predict(X)` interface with zero scikit-learn dependency.

## Implementation Decisions

- **Crate Module Structure**:
  - Add module `crates/mltreelib-core/src/ensemble/`:
    - `forest.rs`: Generic `RandomForest` struct parameterized over tree type `T: TreeEstimator`.
    - `honest.rs`: `HonestSampler` implementing stratified or uniform sample splitting ($\mathcal{I}_{\text{split}}$ vs $\mathcal{I}_{\text{leaf}}$).
    - `boost.rs`: `GradientBooster` managing gradient/hessian updates, shrinkage step, and additive tree aggregation.
    - `conformal.rs`: `ConformalCalibrator` calculating OOB quantile cutoffs and prediction intervals.
    - `stacking.rs`: `StackingEnsemble` with $K$-fold cross-validation engine and NNLS meta-optimizer.
- **Honest Tree Construction Flow**:
  1. Draw subsample $\mathcal{S} \subset \{1, \dots, N\}$ of size $s = \lfloor \gamma N \rfloor$ without replacement.
  2. Partition $\mathcal{S}$ randomly into $\mathcal{I}_{\text{split}}$ ($50\%$) and $\mathcal{I}_{\text{leaf}}$ ($50\%$).
  3. Grow tree partition structure using $\mathcal{I}_{\text{split}}$ only.
  4. Pass $\mathcal{I}_{\text{leaf}}$ down the frozen tree structure; re-estimate leaf parameters (means, linear coefficients, treatment effects) strictly on $\mathcal{I}_{\text{leaf}}$.
- **Python Surface (`crates/mltreelib-py`)**:
  - `mltreelib.ensemble.HonestForestRegressor`
  - `mltreelib.ensemble.ModelTreeBoostRegressor`
  - `mltreelib.ensemble.StackingRegressor`

## Testing Decisions

- **Honest Confidence Interval Coverage**:
  - Simulate non-linear function with known Gaussian noise: $y = \sin(X_1) + \epsilon, \epsilon \sim \mathcal{N}(0, 1)$.
  - Verify that empirical 95% confidence intervals constructed from `HonestForestRegressor` cover the true function value in $95\% \pm 2\%$ of repeated simulations, whereas non-honest forests suffer under-coverage ($<85\%$).
- **Conformal Interval Calibration**:
  - Evaluate `predict_interval(X_test, alpha=0.10)` on benchmark dataset `housing.csv`.
  - Verify empirical test set coverage is $\ge 90.0\%$.
- **ModelTreeBoost Tweedie Convergence**:
  - Train `ModelTreeBoostRegressor(loss="tweedie", tweedie_p=1.5)` on zero-inflated synthetic gamma data.
  - Verify Tweedie deviance decreases monotonically across boosting iterations.
- **Multiprocessing Scaling**:
  - Benchmark 100-tree forest training with `n_jobs=1` vs `n_jobs=4`. Verify speedup is $\ge 3.0\times$.

## Out of Scope

- Multi-node distributed cluster execution (e.g. MPI / Ray; execution is optimized for single-machine multi-core parallelism).
- Deep neural network base learners in stacking (stacking focuses on tree and linear estimators).

## Further Notes

- The honest tree formulation (Athey & Imbens 2016, Wager & Athey 2018) is the single most important mathematical development in tree methodology over the last decade, transforming decision trees from heuristic data-mining algorithms into rigorous econometric estimators.
