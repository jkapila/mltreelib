# [TREE-OPTIMIZATION] Tree Growth Simplification and Memory Allocation Elimination

## Problem Statement

Even after decoupling split evaluation from candidate linear regressions, tree growth still incurs noticeable latency and allocation pressure on large tabular datasets:

1. **Transient Heap Allocation Storm**: During recursive tree growth, every column scan at every internal node allocates and deallocates multiple temporary heap vectors (`pairs`, `yws_sorted`, `missing_yws`). In a tree with 30 nodes and 20 features, this results in thousands of small heap allocations and deallocations across parallel Rayon worker threads, creating memory allocator mutex contention.
2. **Redundant Categorical Partition Overhead**: In categorical feature split evaluation (`scan_column_categorical`), every candidate level subset constructs a dynamic `std::collections::HashSet<usize>` and allocates two new vectors (`left_yws` and `right_yws`) copying all `(y, w)` pairs for every candidate partition. For columns with many levels, this repeatedly re-allocates and copies row-level data.
3. **Repeated Floating-Point Sorting**: At every internal tree node, continuous features sort floating-point values dynamically ($O(N \log N)$ per node). In deep trees, the cumulative sorting overhead dominates tree growth time.
4. **Intermediate Centroid Clustering**: Every candidate or intermediate node evaluation currently instantiates and flushes a 100-centroid `TDigest`, even though t-digests are never queried unless quantile predictions are requested on final leaves.

Users require trees to train with near-zero allocation overhead, sub-millisecond per-node evaluation, and minimal memory footprint.

## Solution

Implement Phases 2 and 3 of the simplification and optimization roadmap:

1. **Zero-Allocation Worker Scratch Buffers (`SplitScratch`)**: Preallocate a single reusable scratch buffer structure per Rayon worker thread (or passed down the recursive call stack) that retains vector capacity across all column scans and tree depths, completely eliminating dynamic heap allocations in the split search loop.
2. **Scalar Precomputed Moments for Categorical Splits**: Calculate per-level summary tuples `(w_sum, wy_sum, wy2_sum)` once per categorical column per node. Evaluate candidate level subsets (both cumulative prefix splits and single-vs-rest) via $O(1)$ scalar arithmetic over these tuples, eliminating `HashSet` allocations and vector copying.
3. **Pre-Sorted Index Permutation Partitioning**: Sort numeric columns once at dataset ingestion ($O(P \cdot N \log N)$), and partition active row index lists or boolean bitmasks down tree levels in $O(N)$ linear time without re-sorting float values.
4. **Lazy Final-Leaf-Only T-Digests**: Restrict 100-centroid t-digest construction strictly to finalized terminal leaves when `leaf_digest=true` (or when `leaf="quantile"`), retaining only lean scalar moments ($N, \text{mean}, \text{variance}$) on intermediate nodes.

## User Stories

1. As a machine learning engineer, I want `ModelTree.fit()` to execute with minimal heap allocations, so that parallel tree growth does not bottleneck on memory allocator locks.
2. As a data scientist training on datasets with 50+ continuous features, I want continuous columns to be sorted at most once upfront, so that tree building completes in fractions of a second.
3. As a user training on high-cardinality categorical features, I want categorical split evaluation to use scalar running sums over precomputed level moments, so that candidate evaluation is $O(K)$ rather than $O(K \cdot N)$.
4. As a researcher running hundreds of tree fits across benchmark datasets, I want zero allocation churn during node splitting, so that thread execution profiles show zero time spent in `malloc` / `free`.
5. As an interactive Python user, I want `ModelTree` to fit within 10–50 ms on datasets with tens of thousands of rows, so that model exploration is completely seamless.
6. As a low-memory embedded or containerized system user, I want intermediate tree nodes to store only scalar moments without centroid arrays, so that memory usage remains tiny during tree growth.
7. As a quantile regression user, I want t-digests to be constructed exclusively on terminal leaves when requested, so that intermediate split search does not waste cycles on centroid clustering.
8. As a maintainer, I want scratch buffers to be passed explicitly or thread-locally without global mutable state, so that parallel execution remains strictly deterministic and thread-safe.
9. As a developer, I want all categorical split ordering heuristics (`frequency_desc`, `target_mean`) to use cumulative scalar prefix sweeps, so that candidate scoring is continuous and branchless.
10. As a benchmark runner, I want single-vs-rest categorical evaluation to compute complement statistics via simple subtraction from total node moments, so that no data points are re-scanned.
11. As a maintainer, I want existing public interfaces and behaviors (`fit`, `predict`, `break_down`, `quantile`, `splits`, `leaves`) to remain 100% backward-compatible.
12. As a test author, I want all mathematical guarantees (optimal split thresholds, gain values, leaf coefficients, and predictions) to be identical to or better than previous implementations.
13. As an engineer, I want Rayon parallel column scans to borrow worker-local scratch buffers cleanly without allocation leaks.
14. As an analyst evaluating missing data, I want NA missing-direction routing (`missing_left`) to evaluate in $O(1)$ scalar addition on top of scratch buffers.
15. As a maintainer, I want performance profiling scripts in `examples/` to easily verify fit time speedups and allocation counts across optimization phases.

## Implementation Decisions

### Decision 1: Reusable `SplitScratch` Per Thread

Define a `SplitScratch` structure containing preallocated vectors:
- `pairs: Vec<(f64, f64, f64)>`
- `missing_yws: Vec<(f64, f64)>`
- `level_moments: Vec<(f64, f64, f64)>`

Instead of creating new vectors inside `scan_column_mean` and `scan_column_categorical`, each worker thread receives a mutable reference `&mut SplitScratch`. Vectors are cleared (`.clear()`) rather than re-allocated, reusing existing heap capacities across all nodes.

### Decision 2: Moment Accumulation for Categorical Partitioning

Replace the `HashSet<usize>` and `Vec<(f64, f64)>` cloning in `scan_column_categorical` with a two-pass algorithm:
1. **Pass 1 (Aggregation)**: Compute for each discrete level $k \in 0..K$ the summary tuple $(w_k, wy_k, wy^2_k)$.
2. **Pass 2 (Candidate Evaluation)**: 
   - For cumulative prefix splits: accumulate running sums $w_L = \sum_{k=0}^i w_k$, $wy_L = \sum_{k=0}^i wy_k$, and compute right-side sums by subtracting from the total node moments: $w_R = w_{\text{total}} - w_L$, $wy_R = wy_{\text{total}} - wy_L$.
   - For single-vs-rest splits: the single level $k$ is left, and the right side is the node total minus level $k$.
No row-level data is copied or stored in sets during candidate scoring.

### Decision 3: Upfront Numeric Sorting & Index Partitioning

For numeric columns, generate an initial sorted index array at the root node:
- For each continuous column $j$, pre-sort row indices by value: `sorted_idx[j]: Vec<usize>`.
- When partitioning a node into left and right children, filter active indices into `left_rows` and `right_rows` using an $O(N)$ boolean membership check or bitmask, preserving the sorted invariant without re-sorting at child nodes.

### Decision 4: Lazy T-Digest Finalization

Intermediate tree nodes during recursive growth store only scalar moments:
- `LeafStats` on intermediate nodes carries an empty digest or `None`.
- Only when a node is finalized as a terminal leaf (upon reaching stopping criteria or pruning), if `leaf_digest == true` or `leaf == Quantile`, the 100-centroid t-digest is populated and flushed from the leaf's assigned rows.

## Implementation Roadmap & Sequential TODOs

The optimizations will be implemented and validated sequentially against TabArena benchmarks (`houses` for regression, `bank-marketing` for classification):

- [x] **Phase 1: Split Search Decoupling** *(Completed in `8c7b838`)*:
  - Decouple candidate split search from linear regression solves.
  - Candidate split evaluation defaults to $O(1)$ scalar variance/deviance reduction (`SplitEval::Fast`).
  - Terminal leaves fit regularized multivariate linear / GLM regression models.
  - Retain `SplitEval::Exhaustive` for exact research comparisons.
- [x] **Phase 2A: Zero-Allocation Worker Scratch Buffers (`SplitScratch`)**:
  - Introduce preallocated per-thread reusable vectors (`pairs`, `missing_yws`).
  - Pass mutable scratch down the recursive growth stack, clearing capacities instead of re-allocating heap vectors.
- [x] **Phase 2B: Scalar Precomputed Moments for Categorical Splits**:
  - Two-pass level moment aggregation $(w_k, wy_k, wy^2_k)$.
  - Evaluate prefix and single-vs-rest splits via $O(1)$ scalar additions and subtractions.
  - Eliminate `HashSet<usize>` and `Vec<(f64, f64)>` cloning in `scan_column_categorical`.
- [x] **Phase 2C: Lazy T-Digest Finalization**:
  - Store only scalar moments ($N, \text{mean}, \text{variance}$) on internal tree nodes.
  - Construct 100-centroid t-digests strictly on finalized terminal leaves when `leaf_digest=true`.
- [ ] **Phase 3A: Pre-Sorted Index Partitioning**:
  - Sort numeric features once at root node ($O(P \cdot N \log N)$).
  - Partition active indices down tree levels in $O(N)$ linear time without re-sorting float values.
- [ ] **Phase 3B: Iterative Flat-Array Queue Tree Building**:
  - Replace recursive `fit_tree` call stack and cloned path vectors (`path.to_vec()`) with a flat index-addressed `Vec<Node>`.
- [ ] **Phase 3C: Single-Pass Bottom-Up Cost-Complexity Pruning**:
  - Compute effective alphas and prune subtrees in a single post-order traversal without repeated whole-tree traversals.
- [ ] **Phase 3D: Conditional Surrogate Search**:
  - Guard surrogate scanning: only scan surrogate features when the primary split feature has active missing values (`missing_count > 0`).
- [ ] **Benchmark Validation Suite**:
  - Establish `examples/bench_optimizations.py` to record fit time, memory, and accuracy on `houses` (20,640 rows, regression) and `bank-marketing` (45,211 rows, classification) after each incremental update.

## Testing Decisions

### Seam

The primary test seam is the public Rust API in `crates/mltreelib-core/tests/model_tree.rs` and the Python API in `tests/test_model_tree_raw.py`:
- All existing tests in `crates/mltreelib-core/tests/model_tree.rs` (70 tests) must continue to pass without modification.
- Split thresholds, gains, predictions, and categorical partitions must match reference values to within floating-point tolerance ($10^{-9}$).
- Verified speedup benchmarks across synthetic and TabArena datasets.

### Prior Art

- Existing `scan_column_mean` and `scan_column_categorical` tests.
- Existing categorical parity tests (`oracle_agrees_on_categorical_split`).
- Scikit-learn's `_splitter.pyx` and LightGBM's `dataset.cpp` pre-sorting and histogram structures.

## Out of Scope

- Pre-binned 256-bin histogram engines with SIMD vectorization (Phase 4, reserved for multi-million-row scaling).
- Oblique or multi-column hyperplane cuts.
- GPU-accelerated split scans.

## Further Notes

- Breiman et al. (CART, 1984) and Quinlan (1993) both document the scalar moment technique for discrete features, proving that when levels are ordered by target mean, the optimal split is guaranteed to be a prefix cut among the ordered levels.
- Pre-sorted index partitioning eliminates the $O(D \cdot N \log N)$ sorting tax across tree depth $D$, reducing sorting work to a single $O(N \log N)$ ingest pass.
