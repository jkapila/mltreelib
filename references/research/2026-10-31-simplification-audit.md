# Simplification Audit: Unnecessary Complexity in Split Search and Tree Growth

**Date**: 2026-10-31  
**Status**: Active Tracking  
**Context**: Discovered while investigating why gradient boosted trees (LightGBM, CatBoost, forust) train hundreds of trees in 0.35s while `ModelTree` takes 5s–13s per tree.

---

## 1. Candidate-Level Multivariate Regression Solves (`scan_column_linear`)

- **What was built**: In `crates/mltreelib-core/src/split.rs`, whenever `leaf == LeafKind::Linear`, `scan_column_linear` evaluates candidate splits by maintaining $(2k+1) \times (2k+1)$ Gram matrices ($X^T X$) and cross-product vectors ($X^T y$), solving linear systems via rank-1 Cholesky updates at **every candidate threshold of every column**.
- **Why it is unnecessary / flawed**:
  - Quinlan's M5 (1992), M5' (1997), and Cubist use standard scalar **SDR (Standard Deviation Reduction)** based on scalar means for split search.
  - Linear models in classical literature are fitted **only once per terminal leaf** on the subset of data reaching that leaf.
  - On a 120-column dataset (e.g. `parallel_scan_is_deterministic`), `scan_column_linear` attempts to invert a $241 \times 241$ matrix thousands of times per node, freezing single unit tests for minutes.
- **Action**: Decouple split finding from leaf fitting. Split finding defaults to fast scalar variance reduction (`scan_column_mean`). Multivariate linear regression is solved only when finalizing terminal leaves.

---

## 2. Candidate-Level IRLS Solves in Distributional Leaves (`scan_column_family`)

- **What was built**: In `crates/mltreelib-core/src/split.rs`, whenever `leaf_family != Gaussian`, `scan_column_family` evaluates candidate splits by running iterative IRLS (Iteratively Reweighted Least Squares) to compute deviance for each candidate cut.
- **Why it is unnecessary / flawed**:
  - Binomial/logistic trees (CART, C4.5, C5.0) select splits using fast frequency counts (Gini or Entropy / Log-Loss of constant class shares), taking $O(1)$ operations per threshold.
  - Poisson trees select splits using constant-rate deviance reduction ($2 \sum y \log(y / \bar{y})$), which depends strictly on scalar sums ($\sum y$, $\sum w$).
  - Solving iterative Newton/IRLS inside candidate split loops creates massive computational overhead with no empirical gain over scalar deviance reduction.
- **Action**: Use scalar deviance / Gini / entropy for candidate split search; execute IRLS only when fitting terminal leaf coefficients.

---

## 3. Hoisted Node-Level Parent Pre-computation (`ParentPre`)

- **What was built**: In `best_split()`, `ParentPre` was introduced to compute full design matrices, Gram matrices, and parent SSE across all rows at the current node to feed `scan_column_linear`.
- **Why it is unnecessary / flawed**:
  - Once split search uses scalar accumulators ($N$, $\sum w$, $\sum w y$, $\sum w y^2$), parent variance is computed in $O(1)$ directly from the node's running sums: $\text{SSE}_P = \sum w y^2 - (\sum w y)^2 / \sum w$.
  - `ParentPre` and its associated matrix allocations are completely redundant.
- **Action**: Eliminate `ParentPre` from the split search path.

---

## 4. Per-Column, Per-Node Dynamic Heap Allocations

- **What was built**: Inside `scan_column_mean`, `scan_column_linear`, and `scan_column_categorical`:
  ```rust
  let mut pairs: Vec<(f64, f64, f64)> = rows.iter().filter_map(...).collect();
  let yws_sorted: Vec<(f64, f64)> = pairs.iter().collect();
  let mut missing_yws: Vec<(f64, f64)> = Vec::new();
  ```
- **Why it is unnecessary / flawed**:
  - Allocates and deallocates dozens of `Vec` instances on the heap for every column at every internal node.
  - On a tree with 30 nodes and 20 columns, this triggers over 1,200 heap allocations and deallocations, causing thread contention on the memory allocator.
- **Action**: Allocate a single reusable `SplitScratch` buffer per worker thread / recursive branch and pass `&mut SplitScratch` down the call stack.

---

## 5. Repeated Floating-Point Sorting at Every Depth Level

- **What was built**: At every node, for every numeric column, rows are collected into `pairs` and sorted dynamically:
  ```rust
  pairs.sort_by(|a, b| a.0.partial_cmp(&b.0)...);
  ```
- **Why it is unnecessary / flawed**:
  - Standard CART / Scikit-Learn sort continuous features **once** upfront at the root node ($O(P \cdot N \log N)$).
  - During tree growth, split partitions are tracked via boolean bitmasks or index lists in $O(N)$ without re-sorting floating-point values.
- **Action**: Pre-sort continuous columns once at tree ingestion; partition active indices down the tree.

---

## 6. `HashSet` Construction and Slice Cloning in Categorical Splits (`scan_column_categorical`)

- **What was built**: Inside `scan_column_categorical`, for every candidate partition of categorical levels:
  ```rust
  let in_left: std::collections::HashSet<usize> = left_idx.iter().copied().collect();
  let mut left_yws: Vec<(f64, f64)> = Vec::new();
  let mut right_yws: Vec<(f64, f64)> = Vec::new();
  ```
- **Why it is unnecessary / flawed**:
  - When categorical levels are ordered (by target mean or frequency), cumulative splits are simply prefix splits ($0..i$ vs $i..K$).
  - A prefix split can be tracked with $O(1)$ scalar accumulators ($\sum w, \sum w y$) by precomputing per-level sums once, requiring zero `HashSet` lookups and zero row cloning.
  - Even for single-vs-rest splits, subtracting the single level's scalar sum from the total node sum takes $O(1)$ math without vector allocations.
- **Action**: Precompute per-level summary tuples `(w_sum, wy_sum, wy2_sum)` once per column per node; evaluate candidate level subsets by summing these scalar tuples directly.

---

## 7. Eager T-Digest Centennial Centroid Allocations in Intermediate Nodes

- **What was built**: In `crates/mltreelib-core/src/node_stats.rs:43`, every call to `LeafStats::from_targets` allocates:
  ```rust
  let mut digest = TDigest::new_with_size(100);
  digest.extend_values(idx.iter().map(|&i| y[i]));
  digest.flush();
  ```
- **Why it is unnecessary / flawed**:
  - T-digests are only needed if the user queries empirical percentiles (e.g. `tree.quantile(x, tau)`).
  - During tree exploration or for intermediate nodes, allocating, inserting, and centroid-clustering hundreds of floating point values is pure overhead.
- **Action**: Intermediate nodes store only scalar moments ($N$, mean, variance). T-digests are built only for finalized terminal leaves, and only when `leaf_digest=true` (or when `leaf=Quantile`).

---

## 8. Summary of Complexity Debt & Performance Payoffs

| Complexity Source | Old Mechanism | Classical / Optimized Equivalent | Speedup Potential |
| :--- | :--- | :--- | :--- |
| **Candidate Evaluation** | Gram matrix update + rank-1 Cholesky inversion per candidate threshold | Scalar running variance reduction ($O(1)$) | **20× – 100×** (Implemented in Phase 1) |
| **Categorical Search** | `HashSet` + vector cloning per candidate partition | Precomputed level moments + scalar addition | **5× – 10×** on high-cardinality features |
| **Per-Node Sorting** | $O(N \log N)$ dynamic float sorting at every depth | Pre-sorted index permutation tracked via bitmask | **3× – 5×** on deep trees |
| **Memory Allocator** | Dozens of heap `Vec` allocations per column per node | Single preallocated `SplitScratch` per worker | **2× – 3×** (eliminates thread contention) |
| **Intermediate T-Digests** | Centroid-clustering t-digest at every evaluated node | Scalar moments for internal nodes; digest only on final leaves | **1.5× – 2×** memory and fit reduction |

