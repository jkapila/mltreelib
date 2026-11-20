# Changelog

This file records user-visible changes. [GitHub Releases](https://github.com/jkapila/mltreelib/releases) provides source archives. Tags under `archive/` mark historical snapshots. [PyPI](https://pypi.org/project/mltreelib/#history) lists published Python packages.

## Unreleased

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
