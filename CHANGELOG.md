# Changelog

This file records user-visible changes. [GitHub Releases](https://github.com/jkapila/mltreelib/releases) provides source archives. Tags under `archive/` mark historical snapshots. [PyPI](https://pypi.org/project/mltreelib/#history) lists published Python packages.

## Unreleased

### Standalone solver surface

- Fit a weighted Gaussian or generalized linear model without growing a tree. Use `mltreelib.solvers.fit` with `sample_weight`. The result exposes coefficients, predictions, convergence, and the solver stop reason.
- Share one solver dispatch across standalone fits and leaf fitting. CCD, L-BFGS, and the evolutionary solvers now run through the same options, status, and convergence contract.

### Generic optimization seam

- Add the shared generic optimization seam in the Rust core. One objective, bounds, constraint, budget, request, and result contract now backs every optimizer method. [Spec 26](specs/26-generic-global-and-constrained-optimizers.md)
- Minimize a callable scalar objective with `mltreelib.solvers.optimize`. Pass one `(lower, upper)` pair per parameter. The optional `gradient` callable supplies exact derivatives, and smooth methods fall back to central differences.
- Report the best point, best objective, iterations, evaluations, termination status, and optional history from every optimization run. Budget exhaustion never reports success.

## v0.4.12

### Python model behavior

- Fix multiclass split scoring when feature values are missing. Numeric and categorical scans now account for missing class mass in candidate gains. [Issue #10](https://github.com/jkapila/mltreelib/issues/10)
- Use `split_eval="histogram"` to opt into at most 256 bins per numeric feature. This mode supports Gini, entropy, gain ratio, and twoing without ExtraTrees. The default remains `"fast"`.
- Fit a binary isotonic calibrator with `mltreelib.calibration.IsotonicCalibrator`. Fit on held-out or out-of-fold scores and evaluate on separate test data.
- Preserve missing numeric values in anomaly and clustering workflows. Save and reload fitted DET, half-space, random cut, active anomaly, context anomaly, CLTree, HDBSCAN, and unsupervised proximity forest models as versioned JSON.

### R source package

- Build and install the first R source package from the repository. It exposes ModelTree fitting and prediction. Ubuntu 24.04 with R 4.3.3 is the checked platform. Installation requires a Rust toolchain and a C compiler.
- Run the `airquality` example for an ozone holdout with missing predictors. The example compares the tree with a training-mean baseline.

[Classification results](https://github.com/jkapila/mltreelib/blob/v0.4.12/references/research/2026-11-21-real-classification-results.md) · [Unsupervised results](https://github.com/jkapila/mltreelib/blob/v0.4.12/references/research/2026-11-21-real-unsupervised-results.md) · [R results](https://github.com/jkapila/mltreelib/blob/v0.4.12/references/research/2026-11-21-r-airquality-results.md) · [Changes](https://github.com/jkapila/mltreelib/compare/v0.4.11...v0.4.12)

## v0.4.11

### Model controls and JSON persistence

- Configure tree growth with `GrowthControl` and leaf fitting with `LeafControl`. Both accept structured options through `ModelTree`, and existing flat keywords still work.
- Choose a compatible optimizer through `LeafControl(solver=...)`. Use `predict(kind=...)` for intervals, adjusted predictions, and class labels.
- Save and reload fitted models as versioned JSON. The release supports `ModelTree`, predictive clustering trees (PCT), and extended isolation trees (EIF). Python pickle uses the same saved state.
- Start from the guided example suites for regression, classification, survival, clustering, anomaly detection, and time series.
- Install the mltreelib agent skill for task selection, runnable recipes, model evaluation, and serving guidance.

```sh
npx skills add jkapila/mltreelib --skill mltreelib
```

[Example suites](https://jkapila.github.io/mltreelib/examples/) · [Model persistence](https://jkapila.github.io/mltreelib/guides/serving-contract/) · [Agent skill](https://github.com/jkapila/mltreelib/tree/v0.4.11/skills/mltreelib) · [Changes](https://github.com/jkapila/mltreelib/compare/v0.4.10...v0.4.11)

## v0.4.10

### Frontier survival trees

- Fit accelerated failure time leaves with Weibull, log-normal, or log-logistic distributions.
- Use mixture cure leaves for populations with long-term survivors and spline leaves for flexible survival curves.
- Grow survival trees with pseudo-value, restricted mean survival time, and AFT deviance split criteria.
- Assess survival calibration with D-calibration and the Integrated Calibration Index.

This is the package version designated for Spec 19. The v0.4.9 tag already contains the implementation.

[Survival guide](https://jkapila.github.io/mltreelib/examples/survival-trees/) · [PyPI](https://pypi.org/project/mltreelib/0.4.10/) · [Changes](https://github.com/jkapila/mltreelib/compare/v0.4.9...v0.4.10)

## v0.4.9

- Added survival and competing-risk trees with weighted log-rank and Gray split tests, a streaming martingale split scan, and Aalen–Johansen cumulative incidence predictions.
- Added statistical and metric split criteria, Equalized Odds splitting, ExtraTrees split search, and EBM FAST interaction screening.
- Reworked the documentation around nine example suites. Added solver benchmarks, a two-stage survival guide, and leaf-level survival plots.

[PyPI](https://pypi.org/project/mltreelib/0.4.9/) · [Changes](https://github.com/jkapila/mltreelib/compare/v0.4.8...v0.4.9)

## v0.4.8

- Added practical modeling guides and streamlined the Elo arena examples.
- Documented softmax classification, survival, and heteroscedastic stump leaves with examples.
- Expanded benchmark coverage for ModelTree variants and measured solver performance at up to 250,000 rows.

[PyPI](https://pypi.org/project/mltreelib/0.4.8/) · [Changes](https://github.com/jkapila/mltreelib/compare/v0.4.7...v0.4.8)

## v0.4.7

- Added the `DropFallback` and `ImputeLocal` policies for missing values in leaf estimation.
- Added a tuned `ModelTree(opt)` entry to the Elo arena and documented HPO defaults.

Source archive only. The tagged package metadata declares v0.4.6.

[Changes](https://github.com/jkapila/mltreelib/compare/v0.4.6...v0.4.7)

## v0.4.6

- Added a streaming scan for multiclass splits.
- Added hierarchical leaf shrinkage and model presets.

[Changes](https://github.com/jkapila/mltreelib/compare/v0.4.5...v0.4.6)

## v0.4.5

- Added live training progress with three verbosity levels.
- Unified GLM and evolutionary leaf solver selection. Added benchmarks for evolutionary solvers.

[Changes](https://github.com/jkapila/mltreelib/compare/v0.4.4...v0.4.5)

## v0.4.4

- Added Leaf-KNN and distance-weighted KNN prediction, with distance metrics and a benchmark showcase.
- Extended retry budgets for slow Elo runs. Added overall, regression, and classification arena views.

[PyPI](https://pypi.org/project/mltreelib/0.4.4/) · [Changes](https://github.com/jkapila/mltreelib/compare/archive/v0.4.3...v0.4.4)

## v0.4.3

- Optimized the GLM leaf solver and documented the roadmap for split telemetry.
- Added examples for distributional split criteria and research notes on frontier tree methods.

Source archive only. The `archive/v0.4.3` tag keeps this snapshot out of the package publishing workflow.

[Changes](https://github.com/jkapila/mltreelib/compare/v0.4.2...archive/v0.4.3)

## v0.4.2

- Added Tweedie, heteroscedastic, Tobit, cross-validation, and martingale split criteria.
- Fixed math rendering and refreshed the README.

[PyPI](https://pypi.org/project/mltreelib/0.4.2/) · [Changes](https://github.com/jkapila/mltreelib/compare/v0.4.1...v0.4.2)

## v0.4.1

- Extract independent C5.0 rules with `extract_rules()`. The rule extractor simplifies feature intervals and prunes conditions with Quinlan’s pessimistic error bound.
- Inspect linear leaves with `leaf_summary(leaf_id)` and `summary()`. The summaries report coefficient estimates, standard errors, test statistics, and fit diagnostics.
- Adjust predictions with Cubist-style neighbor residuals using `predict_adjusted(X, neighbors=5)`.
- Added GitHub Actions builds for Python packages.

[PyPI](https://pypi.org/project/mltreelib/0.4.1/) · [Changes](https://github.com/jkapila/mltreelib/compare/archive/v0.4.0...v0.4.1)

## v0.4.0

- Added honest causal trees and uplift prediction with benchmark examples.
- Published technical articles, tree diagrams, and metrics plots. Organized the examples into three tracks.

Source archive only.

[Changes](https://github.com/jkapila/mltreelib/compare/archive/v0.3.3...archive/v0.4.0)

## v0.3.3

- Added the unified `ClusterTree` and `AnomalyTree` interfaces.
- Exposed estimator parameters and MARS leaf details. Updated softmax leaf and pruning logic.

Source archive only.

[Changes](https://github.com/jkapila/mltreelib/compare/archive/v0.3.2...archive/v0.3.3)

## v0.3.2

- Added tree-structured MARS leaves and a standalone MARS regressor.
- Added sample weights, forced linear predictors, GLM fitting, Fast MARS, and cross-validation pruning to the MARS implementation.
- Added the cross-validation purity gate and tail quantile protection.

Source archive only.

[Changes](https://github.com/jkapila/mltreelib/compare/archive/v0.3.0...archive/v0.3.2)

## v0.3.0

- Exposed algorithm parameters across the clustering and anomaly detection APIs.

Source archive only.
