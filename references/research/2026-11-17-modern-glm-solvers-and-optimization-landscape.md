# Modern GLM Solvers & Optimization Landscape: Beyond Naive IRLS

**Date**: 2026-11-17  
**Scope**: Optimization algorithms for Generalized Linear Models (logistic, Poisson, Gamma, Tweedie) in leaf models and fast tabular learners.

---

## 1. Executive Summary & Why Naive IRLS Fails

In classical statistics (e.g. R's base `glm()`), Generalized Linear Models are fit using **Iteratively Reweighted Least Squares (IRLS)**.
At iteration $t$:
1. Compute working weights $W_i = \mu_i(1 - \mu_i)$ (for binomial) and working link targets $z_i = \eta_i + \frac{y_i - \mu_i}{W_i}$.
2. Form the weighted Gram matrix $X^T W X$ in $O(N \cdot P^2)$ time.
3. Solve or factorize $(X^T W X + \lambda I)$ in $O(P^3)$ time.
4. Backtrack along step $\Delta \beta$ evaluating the global loss over all $N$ rows.

### The Scaling Wall
- **Dense High-Dimensional Features**: When $P \ge 100$ or missingness indicators expand features to $2P$, forming $X^T W X$ and factorizing takes millions of FLOPs per iteration.
- **Line-Search Overhead**: Up to 50 backtracking passes over $N$ rows per iteration produces $10{,}000$ full data passes per leaf.
- When placed at tree leaves (even with decoupled split evaluation), fitting dozens of leaves causes minute-long stalls.

Modern statistical computing packages (`scikit-learn`, `glmnet`, `fastglm`, `liblinear`, `H2O`) abandon naive IRLS in favor of first-order quasi-Newton, coordinate descent, or preconditioned iterations.

---

## 2. Taxonomy of Modern GLM Solvers

```
                    ┌─────────────────────────┐
                    │    Modern GLM Solvers   │
                    └────────────┬────────────┘
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
 │ Quasi-Newton  │       │  Coordinate   │       │ Accelerated   │
 │   (L-BFGS)    │       │ Descent (CD)  │       │     PIRLS     │
 └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
         │                       │                       │
         ├─ Scikit-learn         ├─ Glmnet               ├─ FastGLM (R)
         ├─ SciPy / PyTorch      ├─ Liblinear            ├─ Speedglm
         └─ No Hessian inverse   └─ Exact L1 + sparsity  └─ Cholesky + warm
```

---

## 3. Algorithm Deep Dive

### A. L-BFGS (Limited-Memory Broyden-Fletcher-Goldfarb-Shanno)
- **Standard Reference**: Nocedal & Wright, *Numerical Optimization*; Byrd et al. (1995).
- **Core Principle**: Never forms or inverts the $P \times P$ Hessian. Stores only the last $m \approx 5\text{--}10$ displacement vectors $s_k = \beta_{k+1} - \beta_k$ and gradient differences $y_k = g_{k+1} - g_k$.
- **Per-Iteration Complexity**: $O(N \cdot P)$ to compute the gradient $g = X^T (\mu - y) + \lambda \beta$, plus $O(m \cdot P)$ for the two-loop recursion.
- **Why It Wins**:
  - Scales gracefully up to $P = 100{,}000$.
  - Line search uses Wolfe conditions, converging in 15–25 iterations.
  - Used as the default solver in `scikit-learn`'s `LogisticRegression(solver='lbfgs')`.

### B. Cyclic Coordinate Descent (glmnet / NewGLM)
- **Standard Reference**: Friedman, Hastie, & Tibshirani (2010), *Regularization Paths for Generalized Linear Models via Coordinate Descent*.
- **Core Principle**: Updates one scalar coefficient $\beta_j$ at a time while holding all others fixed.
  $$\beta_j \leftarrow \frac{S\left(\sum_i w_i x_{ij} (z_i - \eta_i + x_{ij} \beta_j), \lambda_1\right)}{\sum_i w_i x_{ij}^2 + \lambda_2}$$
  where $S(v, \lambda) = \text{sign}(v)(|v| - \lambda)_+$ is the soft-thresholding operator.
- **Why It Wins**:
  - Handles L1 (Lasso) penalties exactly without non-smooth optimization artifacts.
  - Explores active sets: coefficients stuck at zero are skipped, reducing work from $P$ to the active subset size $k \ll P$.
  - $0$ matrix factorizations.

### C. Preconditioned / Truncated PIRLS (fastglm)
- **Standard Reference**: R `fastglm` package; Wood (2011).
- **Core Principle**: Uses pivoted Cholesky or QR on an unweighted or pre-scaled design. Caps iterations at 10–15 with early stopping tolerances ($\Delta \text{deviance} < 10^{-7}$).
- **Why It Wins**: 10–50x faster than base R `glm()` while retaining exact maximum likelihood compatibility.

### D. Derivative-Free Global Search (CMA-ES / Direct Search)
- **Use Case**: Niche loss functions (non-convex GLMs, zero-inflated hurdle models, arbitrary quantile kinks) where link gradients are unavailable or non-smooth.
- **Trade-off**: Requires hundreds to thousands of function evaluations; ideal as a meta-optimizer for hyperparameter tuning or exotic non-smooth leaf objectives, but too slow for inner leaf regression.

---

## 4. Comparison Matrix for Tree Leaf Models

| Solver | Per-Iter Complexity | Memory | Handles L1? | Convergence | Best Fit In mltreelib |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla IRLS** (Old) | $O(N P^2 + P^3)$ | $O(P^2)$ | Weak (CD inner) | 20–200 iters | Legacy, avoid for $P > 30$ |
| **Capped IRLS** (Current) | $O(N P^2 + P^3)$ (cap 20) | $O(P^2)$ | Yes | $\le 20$ iters | Safe baseline for $P < 50$ |
| **L-BFGS** | $O(N P + m P)$ | $O(m P)$ | Smooth approximations | 15–25 iters | **Primary target for dense leaves** |
| **Coordinate Descent** | $O(N)$ per feature | $O(N)$ | Native & exact | 10–30 sweeps | **Best for sparse / Lasso leaves** |
| **0D Fallback** | $O(1)$ | $O(1)$ | N/A | Immediate | High-$P$ safeguard & missing rows |

---

## 5. Architectural Recommendations for `mltreelib`

1. **Phase 1 (Completed Today)**:
   - Dropped NA indicator columns (eliminating $2P$ explosion).
   - Capped IRLS outer loop to 20 and backtracking to 5.
   - Enforced 0D baseline fallback on missing row values.
   - Guarded high-dimensional leaves ($P \ge 50 \to \text{path}$, $P \ge 100 \to \text{mean}$).
2. **Phase 2 (Next Milestone)**:
   - Implement pure Rust L-BFGS solver with two-loop recursion in `crates/mltreelib-core/src/optim/lbfgs.rs`.
   - Wire coordinate descent for L1 elastic-net leaf models.
   - Benchmark against `scikit-learn` logistic regression across TabArena classification suite.
