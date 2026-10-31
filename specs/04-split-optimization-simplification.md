# [SPLIT-OPTIMIZATION] Decoupled Split Search and Linear Leaf Optimization Roadmap

## Problem Statement

When fitting model trees with linear or distributional leaves, tree construction is unexpectedly slow—taking seconds to build a single tree on modest datasets, whereas gradient boosted trees (GBDTs) build hundreds of trees in fractions of a second.

The root cause is a conceptual conflation in split search: during candidate evaluation, candidate splits evaluate full multivariate linear regressions (or iterative GLMs) across thousands of candidate cuts on every feature at every internal node. In the classical literature (Quinlan's M5, Wang & Witten's M5', Loh's GUIDE, and Cubist), split selection does **not** solve multivariate regressions at every candidate threshold; it uses fast scalar variance reduction (SDR/SSE) or residual curvature to partition space, and fits multivariate linear models **only once per terminal leaf** after tree partitioning is complete.

Because candidate split evaluation was solving linear systems inside the innermost loop, users and maintainers face a 100× performance penalty that prevents interactive use and benchmark scaling.

## Solution

Decouple **split-finding** from **terminal leaf fitting**:

1. **Fast Split Search**: The split search across all continuous and categorical features operates exclusively via scalar variance/deviance reduction (standard deviation reduction / sum of squared errors). Every candidate split threshold is evaluated in $O(1)$ arithmetic using running scalar accumulators, without Gram matrix construction or matrix inversions.
2. **Terminal Leaf Fitting**: Multivariate linear models (ridge regression, OLS, or IRLS GLMs) are fitted **only once per terminal leaf node** on the subset of data partitioned to that leaf.
3. **Preserved Semantics**: Predictions, linear coefficients, link functions, Chow tests, and breakdown explanations retain their full mathematical fidelity and expressiveness while running in milliseconds.
4. **Phased Optimization**: A clear 4-phase roadmap progressing from algorithmic decoupling (immediate 100× win) to zero-allocation scratch buffers, pre-sorting, and histogram-based binning.

## User Stories

1. As a modeler, I want `ModelTree(leaf="linear")` to build in milliseconds rather than seconds, so that I can use model trees interactively in exploratory workflows.
2. As a benchmark runner, I want `ModelTree` to complete 18 TabArena datasets within the standard wall-clock budget without hitting 300s process timeouts.
3. As an engineer, I want the split evaluation loop to operate on $O(1)$ scalar updates per candidate threshold, so that computational complexity is linear in sample size rather than cubic in feature count.
4. As a researcher, I want the split search to match Quinlan's M5 and Cubist standard deviation reduction (SDR), so that our methodology aligns with established classical tree literature.
5. As a user, I want terminal leaves to fit full regularized multivariate linear models (`l1`, `l2`, link functions), so that the resulting tree remains a genuine piecewise-linear model tree.
6. As a user, I want predictions, prediction intervals, and leaf coefficients to remain completely compatible with the existing API, so that downstream pipelines do not break.
7. As a researcher, I want an optional `split_eval="exhaustive"` flag, so that I can experimentally compare fast variance-split model trees against exhaustive candidate-regression model trees.
8. As a maintainer, I want zero heap allocations during the column scanning loop, so that memory allocator contention across parallel threads is completely eliminated.
9. As a user, I want the Chow significance test (`split_test="chow"`) to evaluate only the single winning split chosen by the fast scan, rather than evaluating Chow tests on every rejected candidate cut.
10. As a performance engineer, I want continuous features to be sorted at most once at dataset ingest (or partitioned via pre-sorted indices), eliminating $O(N \log N)$ dynamic float sorting at deep tree levels.
11. As a large-dataset user, I want optional pre-binned histogram scanning (256 bins), so that split search over millions of rows operates at L1 cache speeds with sibling histogram subtraction.
12. As a maintainer, I want all existing test suites to pass without regression, ensuring that model behavior and tree structures remain sound.

## Implementation Decisions

### Decision 1: Split Search Defaults to Fast Scalar Reduction
Split candidate scoring across numeric and categorical columns will use scalar running statistics ($N$, $\sum w$, $\sum w y$, $\sum w y^2$). The candidate threshold that maximizes weighted SSE reduction (or classification impurity gain) is selected. No Gram matrices, matrix inversions, or iterative solvers will run inside the candidate threshold sweep.

### Decision 2: Leaf Fitting Runs Exclusively at Leaf Finalization
When a node becomes terminal (due to reaching `max_depth`, falling below `min_rows`, satisfying `purity_gate`, or being pruned), the terminal leaf fitting algorithm runs on the rows assigned to that leaf. If `leaf="linear"`, the multivariate ridge regression solve runs once for that leaf. If `leaf_family="binomial"` or `"poisson"`, the weighted IRLS solver runs once for that leaf.

### Decision 3: Chow Test Evaluated Post-Scan
When `split_test="chow"` is enabled, candidate selection proceeds via fast scalar SSE reduction to identify the candidate boundary. The Chow structural break F-statistic and p-value are computed only for this single winning partition. If the test fails the significance alpha threshold, the node is not split.

### Decision 4: Phased Optimization Roadmap
The implementation will proceed in four distinct phases:
- **Phase 1 (Decoupling)**: Route column split scans to fast scalar variance reduction for all leaf kinds. Keep linear/GLM fitting exclusively in terminal leaf finalization.
- **Phase 2 (Zero-Allocation Buffers)**: Replace transient vector allocations in the column scanning loop with a preallocated scratch structure passed down recursive growth calls.
- **Phase 3 (Pre-Sorted Index Partitioning)**: Sort continuous columns once before recursive tree building, tracking active rows via index partitioning or bitmasks.
- **Phase 4 (Histogram Engine)**: Provide optional 256-bin histogram aggregation with sibling subtraction for massive tabular datasets.

## Testing Decisions

### Seam
The primary test seam is the existing core integration test suite (`crates/mltreelib-core/tests/model_tree.rs` and `tests/test_model_tree.py`). Testing at this external API seam verifies that:
1. `ModelTree.fit(X, y)` produces valid model trees with non-trivial splits and fitted leaf models.
2. Predictions from fast-split model trees accurately recover planted piecewise-linear functions and regime changes.
3. Fit runtime on benchmark datasets drops by at least 10× to 100×.
4. Monotone constraints, interaction groups, NA routing, and honesty continue to function as expected under fast split selection.

### Prior Art
- Existing leaf kind tests (`leaf="mean"`, `leaf="linear"`, `leaf="huber"`).
- Existing knob tour tests (`examples/knob_tour.py`).
- Existing rpart-parity test fixtures (`crates/mltreelib-core/tests/rpart_repro.rs`).

## Out of Scope

- Oblique / hyperplane splits (combinations of multiple features inside an internal split test).
- Quantile regression leaf fitting during the split search (empirical quantiles remain powered by terminal t-digests).
- Distributed or GPU-accelerated histogram computation.

## Further Notes

- Quinlan's original M5 paper (*"Learning with Continuous Classes"*, 1992) explicitly establishes this architecture:
  > *"The tree is constructed by creating a regression tree with standard deviation reduction, and subsequently replacing each subtree with a linear model."*
- Alexander and Gloor (2010) explored exhaustive linear leaf split evaluation, but concluded that the computational cost scales cubically with active features without consistent accuracy advantages over fast scalar split finding.
