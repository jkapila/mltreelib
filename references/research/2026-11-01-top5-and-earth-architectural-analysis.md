# Top 5 Model Architecture Analysis, EARTH/MARS Deep Dive, and Core Empirical Findings

**Date**: 2026-11-01  
**Author**: Antigravity & Jitin Kapila  
**Status**: Research & Methodology Reference  
**Scope**: TabArena-v0.1 Regression (18 datasets) and Classification (24 datasets)

---

## 1. Executive Summary

This note analyzes the architectural mechanics that drive the top-performing tabular models on the TabArena benchmark suite, investigates Jerome Friedman’s Multivariate Adaptive Regression Splines (**MARS** / R **`earth`**) as a unique structural bridge between trees and linear models, and consolidates the core algorithmic findings from the Phase 3 optimization benchmarks in `mltreelib-core`.

Key insights:
1. **The Ensembling & Local Linearity Dominance**: In regression, pure trees suffer from high step-function variance. The top 5 models achieve superior Elo through **variance reduction** (Random Forest / ExtraTrees), **hybrid linear modeling** (Local Linear Forests `grf_llf`), and **gradient shrinkage** (`ChimeraBoost`).
2. **Probability Calibration in Classification**: In classification, oblivious trees with ordered target statistics (`CatBoost`) and cyclic univariate GAMs with interaction splines (`EBM`) dominate LogLoss by generating smooth, well-calibrated class margins rather than extreme binary probabilities.
3. **EARTH's Structural Superpower**: Unlike decision trees that slice space into discontinuous orthogonal hyper-rectangles (causing sharp boundary artifacts), `earth` constructs **continuous piecewise linear basis functions** via reflected hinge pairs $(x - t)_+$ and $(t - x)_+$. This gives it the interpretability of trees and the smoothness of splines.
4. **Cache Dynamics in Tree Growth**: Sorting rows in contiguous L1 cache via in-place PDQsort (`sort_unstable_by`) significantly beats multi-column pre-sorted index partitioning on large datasets (2× faster on `bank-marketing`), because partitioning $P$ index vectors generates hundreds of megabytes of heap reallocation churn.

---

## 2. Regression Top 5 Architectural Breakdown

### 1. ChimeraBoost (Elo: 1690.0 | Fit: 12.01 s)
- **Architecture**: Boosted hybrid ensemble combining shallow decision tree structures with linear residual corrections.
- **Why it wins**: Sits between pure tree ensembles (which cannot extrapolate linear trends) and pure linear models (which miss localized regimes). It fits gradient boosts with a low learning rate ($\eta = 0.05$) and regularized ridge leaf updates. By shrinking successive trees, it prevents individual tree overfits while capturing global trends through the linear residual backbone.

### 2. ranger (Elo: 1622.3 | Fit: 18.24 s)
- **Architecture**: High-performance C++ implementation of Breiman’s Random Forest (500 trees, unpruned).
- **Why it wins**: Ensembles fully grown, high-variance, zero-bias CART trees. At each split, only $\lfloor\sqrt{p}\rfloor$ or $\lfloor p/3 \rfloor$ random predictors are searched. The bootstrap aggregation decorrelates the trees, causing individual variance to cancel out ($Var \propto 1/B + \rho$) while retaining non-linear interaction signal. Handles R factors natively via two-pass frequency ordering.

### 3. sklearn Random Forest (Elo: 1618.1 | Fit: 19.20 s)
- **Architecture**: Scikit-learn’s Cython-based CART forest (100 trees, bootstrap).
- **Why it wins**: Highly consistent baseline across tabular geometries. While slower per tree than C++ implementations, its continuous float split search and robust greedy criterion (SSE reduction) make it resilient across noisy, non-monotone targets.

### 4. ExtraTrees (Elo: 1568.1 | Fit: 13.02 s)
- **Architecture**: Extremely Randomized Trees (Geurts, Ernst, & Wehenkel, 2006).
- **Why it wins**: Rather than computing the globally optimal split threshold across all unique feature values ($O(N \log N)$ per node), ExtraTrees draws a random uniform threshold for each selected feature and picks the best among them. This trade-off slightly increases bias but drastically cuts inter-tree correlation $\rho$, often yielding higher ensemble accuracy and 2–3× faster tree growth on continuous data.

### 5. grf_llf — Local Linear Forest (Elo: 1532.3 | Fit: 23.17 s)
- **Architecture**: Generalized Random Forests (Athey, Tibshirani, & Wager, JASA 2019).
- **Why it wins**: This model directly validates `mltreelib`’s thesis. Standard random forests predict a weighted average of training targets in the leaf: $\hat{y} = \sum \alpha_i(x) y_i$. Local Linear Forests use the forest's leaf co-occurrence frequency strictly as an **adaptive kernel / metric** to compute observation weights $w_i(x)$, and then solve a localized regularized linear ridge regression around the query point $x$. This eliminates the boundary bias and step-function flat-lining of standard trees on smooth manifolds.

---

## 3. Classification Top 5 Architectural Breakdown

### 1. EBM — Explainable Boosting Machine (Elo LogLoss: 1473.4 | Elo Acc: 1357.9 | Fit: 31.47 s)
- **Architecture**: Generalized Additive Model with Pairwise Interactions ($GA^2M$) trained via cyclical gradient boosting (InterpretML / Caruana et al.).
- **Formula**: $g(E[y]) = \beta_0 + \sum f_i(x_i) + \sum f_{ij}(x_i, x_j)$.
- **Why it wins**: Instead of building full multi-variable decision trees, EBM cycles through features one by one, fitting a tiny 1D spline/tree update per feature with a very small learning rate ($\eta = 0.01$) across hundreds of rounds. It then ranks and fits the strongest pairwise interaction terms $f_{ij}$. Because predictions are sums of 1D shape functions and 2D heatmaps, probabilities never saturate or overfit extreme margins, yielding the highest LogLoss calibration in the field.

### 2. perpetual (Elo LogLoss: 1468.1 | Elo Acc: 1380.6 | Fit: 8.02 s)
- **Architecture**: Self-terminating gradient booster written in pure Rust (Veritree).
- **Why it wins**: Employs an exact budget-aware generalization stopping rule based on statistical concentration inequalities. It eliminates manual `n_estimators` tuning and early-stopping patience heuristics. By dynamically modulating shrinkage and leaf depth as statistical signal fades, it halts before noise fitting degrades cross-entropy.

### 3. CatBoost (Elo LogLoss: 1440.8 | Elo Acc: 1294.4 | Fit: 2.56 s)
- **Architecture**: Oblivious (Symmetric) Decision Trees + Ordered Target Statistics (Prokhorenkova et al., NeurIPS 2018).
- **Why it wins**: Oblivious trees apply the identical split predicate $(x_j \le t)$ across all nodes at depth $d$, making the tree a balanced hyper-cube that evaluates via SIMD bitmasks without pointer chasing. Crucially, its **ordered boosting** calculates target statistics strictly on random permutation prefixes ($r' < r$), eliminating the target leakage that degrades cross-entropy on categorical columns in standard gradient boosters.

### 4. ChimeraBoost (Elo LogLoss: 1435.0 | Elo Acc: 1108.4 | Fit: 2.37 s)
- **Architecture**: Boosted binomial trees with internal categorical frequency grouping and adaptive margin regularization.
- **Why it wins**: Fast convergence on high-cardinality binary classification sets by directly penalizing leaf log-odds divergence.

### 5. bart — Bayesian Additive Regression Trees (Elo LogLoss: 1429.7 | Elo Acc: 1236.3 | Fit: 8.57 s)
- **Architecture**: Sum-of-trees model evaluated via MCMC sampling under an explicit regularization prior (Chipman, George, & McCulloch, 2010).
- **Why it wins**: Each tree is constrained by a prior $P(T) = \alpha (1 + d)^{-\beta}$ to remain shallow (depth 2–3), and leaf parameters follow a conjugate normal prior. Posterior draws explore the tree space through reversible-jump MCMC steps (birth, death, change). Averaging across the posterior chain produces smooth posterior probabilities with coherent uncertainty bounds.

---

## 4. Deep Dive: EARTH / MARS (Multivariate Adaptive Regression Splines)

### The Classical Problem with Trees
Classification and Regression Trees (CART) partition feature space into disjoint rectangular regions:
$$\hat{y}(x) = \sum_{m=1}^M c_m \cdot I(x \in R_m)$$
This formulation has two fundamental limitations:
1. **Discontinuity**: As $x$ crosses a split boundary $t$, the prediction jumps instantaneously from $c_{\text{left}}$ to $c_{\text{right}}$. In continuous physical systems (e.g., housing prices, kinematics, yields), this step-function nature requires very deep trees to approximate simple linear slopes, consuming enormous sample sizes and leaving jagged artifacts.
2. **Inability to Extrapolate**: Beyond the extreme training values, trees predict a flat constant line, failing completely on trended or out-of-bounds regimes.

### Jerome Friedman’s Insight (1991)
Jerome H. Friedman (Annals of Statistics, 1991) introduced **MARS**, implemented in modern R by Stephen Milborrow’s **`earth`** package.

Instead of step-functions $I(x \le t)$, MARS uses **reflected pairs of piecewise linear hinge functions**:
$$(x - t)_+ = \max(0, x - t)$$
$$(t - x)_+ = \max(0, t - x)$$

```
        (t - x)+                  (x - t)+
           \                         /
            \                       /
             \                     /
    __________\___________________/__________
              t (knot location)
```

At the knot location $t$:
- Both functions equal zero.
- Together, they form a **continuous, unbroken bend** at $t$.
- Summing linear combinations of hinge pairs produces an arbitrary continuous non-linear curve without jumps!

### The Two-Stage Algorithm

1. **Forward Pass (Over-Fitting & Basis Expansion)**:
   - Starts with a constant model $B_0(x) = 1$.
   - At each step, scans all existing basis functions $B_m(x)$, all variables $x_j$, and all observed candidate knot locations $t$.
   - Selects the split $(B_m, x_j, t)$ that maximally reduces residual sum of squares when adding the paired products:
     $$B_{2k-1}(x) = B_m(x) \cdot (x_j - t)_+$$
     $$B_{2k}(x) = B_m(x) \cdot (t - x_j)_+$$
   - Multiplying by an existing basis function $B_m(x)$ automatically creates **localized multi-way interactions**! If $B_m$ depends on $x_1$ and the new hinge is on $x_2$, the product $B_m(x_1) \cdot (x_2 - t)_+$ is non-zero *only in the subregion where $B_m$ is active*.
   - Repeats until a maximum number of basis functions (e.g., 30–50) is reached.

2. **Backward Pass (GCV Pruning)**:
   - The forward model overfits. The backward pass removes basis functions one at a time, selecting the model that minimizes **Generalized Cross Validation (GCV)**:
     $$\text{GCV} = \frac{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}(x_i))^2}{\left(1 - \frac{C(M)}{N}\right)^2}$$
     where $C(M) = M + d \cdot \frac{M - 1}{2}$ penalizes both the number of parameters $M$ and the cost of searching for knot locations (typically $d \approx 2$ or $3$).
   - This backward pass runs in fractions of a second via fast linear algebra updates (Cholesky / QR downdates).

### Why EARTH / MARS Matters to `mltreelib`

1. **Bridging Tree Partitions and Continuous Splines**:
   `mltreelib`'s core thesis is that tree partitioning combined with linear models captures tabular structure that flat foundation models lose. MARS is the continuous generalization of this exact idea.
2. **Continuous Leaf Boundaries**:
   Instead of hard disjoint leaf boundaries where predictions jump abruptly, hinge functions allow leaf predictions to blend continuously at the split threshold.
3. **Automated Interaction Discovery**:
   A decision tree discovers interactions implicitly by splitting on $x_2$ inside the subtree of $x_1$. MARS does this explicitly via product basis functions $B_m(x_1) \cdot (x_2 - t)_+$, but retains an additive structure that can be decomposed and inspected directly (like partial dependence plots).
4. **Where EARTH Struggles (and Where Trees Win)**:
   - High-cardinality categoricals: MARS relies on dummy/one-hot encoding, causing the basis expansion to explode on columns with 50+ levels. Trees handle arbitrary categorical subsets in $O(K \log K)$ via scalar moments.
   - Missing data: Classical MARS drops missing values or requires imputation. `mltreelib` treats missingness as a first-class branch with $O(1)$ learned default routing.

---

## 5. Empirical Optimization Discoveries from Today's Session

### A. L1 Cache-Local PDQsort vs Pre-sorted Multi-Column Partitioning
In Phase 3A, we benchmarked upfront numeric sorting ($O(P \cdot N \log N)$ pre-sort) with active index list partitioning down tree levels.
- **Findings**:
  - On `houses` (20.6k rows, 8 cols), pre-sorted partitioning took **588 ms**.
  - On `bank-marketing` (45.2k rows, 13 cols), partitioning 13 column vectors of 32-byte structs at every split generated over **470 MB of heap memory allocation churn**, slowing fit time from **1,334 ms to 2,692 ms (over 2× slower)**.
- **Mechanism**: Modern CPUs possess large L1/L2 data caches. In-place PDQsort (`sort_unstable_by`) operating on a single preallocated contiguous vector (`SplitScratch.pairs`) stays entirely in CPU cache and executes at ~15–20 million pairs/sec. Creating, cloning, and re-allocating heap vectors across recursive tree nodes incurs memory allocator mutex contention that dwarfs the sorting savings.
- **Result**: `presort` was exposed as an opt-in diagnostic knob (`presort: bool`), keeping cache-local scratch PDQsort as the default engine.

### B. Single-Pass Moments & O(1) Missing Complement Calculation
- **Algorithm**:
  Instead of multiple sequential loops over rows:
  1. Stream observed values into `scratch.pairs`, apply grid precision rounding, and accumulate observed sums $(tn, tsw, tswy, tswy2)$ in **one unified pass**.
  2. Compute missing row statistics via exact scalar complement from the precomputed node moments:
     $$\text{msw} = \text{node\_sw} - \text{tsw},\quad \text{mswy} = \text{node\_swy} - \text{tswy},\quad \text{mswy2} = \text{node\_swy2} - \text{tswy2}$$
- **Impact**: Zero rows are scanned to evaluate missingness. Both `missing_left = true` and `missing_left = false` branches are evaluated in $O(1)$ scalar additions. Prediction parity is maintained to $10^{-9}$.

### C. Single-Pass Bottom-Up Pruning Deviance
- **Algorithm**:
  In Phase 3C, we eliminated `mean_sse(rows)` from `crates/mltreelib-core/src/prune.rs`. `FrameStats::from_ty` already calculates the node's exact weighted sum of squared residuals (`stats.sse`).
  - Leaf cost calculation became $O(1)$ scalar algebra:
    $$\text{cost}_{\text{Cp}} = \text{stats.sse} + \text{cp\_alpha},\quad \text{cost}_{\text{Bic}} = N \cdot \ln(\text{stats.sse}/N) + \ln(N)$$
- **Impact**: Reduced `houses` fit time to **575.8 ms** (a new repo record, down from 629.4 ms baseline).

### D. Global HPO Defaults: The "Deep Ridge" Frontier
Our global HPO defaults search across diverse TabArena shapes revealed two distinct optimal regimes:
1. **The Classical Default** (Rank 1):
   `leaf=linear, leaf_cols=all, l2=0.5, depth=5, min_rows=50, cp_alpha=5.0, split_precision=0.001` (RMSE: 3.0197). Fine precision grid rounding with moderate regularization.
2. **The Deep Ridge Speed Frontier** (Rank 2):
   `leaf=linear, leaf_cols=all, l2=2.0, depth=6, min_rows=50, cp_alpha=5.0, split_precision=None` (RMSE: 3.0252, Fit Time: **62.7 ms** vs 95.4 ms).
   - Higher ridge penalty ($\lambda = 2.0$) stabilizes leaf regressions sufficiently that continuous split precision rounding can be skipped entirely, speeding up tree fits by **34%** with zero meaningful loss in accuracy.

---

## 6. Actionable Synthesis for Future Work

1. **Continuous Tree Leaves (MARS-Tree Hybrids)**:
   In a future phase, evaluate replacing discontinuous step leaves with localized hinge basis functions $(x_j - t)_+$ at terminal nodes, blending piecewise linear cuts continuously across leaf boundaries.
2. **Oblivious Tree Flavor (CatBoost in Rust)**:
   Investigate a fast symmetric tree engine (`ObliviousTree`) using bitmask evaluation across tree depths for high-throughput inference scenarios.
3. **Local Linear Forest Component**:
   Extend `mltreelib`'s ensemble roadmap to evaluate a native Rust implementation of Local Linear Forests (`grf_llf`), using fast random tree projections to compute local kernel weights for ridge regressions.
