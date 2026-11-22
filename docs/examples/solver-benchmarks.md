---
title: "Suite 10: Solvers, Optimizers & Real-Data Tuning"
description: "Unified benchmark of numerical and evolutionary leaf solvers, standalone weighted GLM fitting, bounded black-box optimization, and real-data tree tuning in mltreelib."
keywords:
  - decision tree leaf solver
  - L-BFGS model tree
  - IRLS Cholesky regression tree
  - evolutionary decision tree
  - SHADE differential evolution
  - CMA-ES tree solver
  - coordinate descent elastic net
  - generic optimizer
  - black-box optimization
---

# Suite 10: Solvers, Optimizers & Real-Data Tuning

Benchmark numerical, quasi-Newton, and continuous evolutionary leaf solvers, standalone GLM fits, and bounded black-box optimizers implemented natively in `mltreelib-core`.

Run suite from repository root after building release extension:

```bash
uv run python examples/suites/solver_benchmarks.py
```

---

## 1. Architectural Decoupling: Tree Partitions vs. Leaf Solvers

In standard model trees, evaluating complex optimization problems at every candidate split threshold incurs prohibitive computational cost ($O(K \cdot N \cdot P^2)$). `ModelTree` solves this through **two-control decoupling**:

1. **Tree Search**: Fast scalar heuristics (Variance reduction, Chow test, Log-rank) discover spatial partition boundaries.
2. **Leaf Optimization**: Once terminal regions are established, the compatible optimizer selected with `LeafControl(solver=...)` fits each local model.

All seven choices apply to Gaussian linear leaves. Supported non-Gaussian linear GLMs use IRLS or L-BFGS.

```mermaid
flowchart TD
    Data["Candidate Data in Leaf Node\n[X, y]"] --> Fam{"Is the loss smooth\nand convex?"}

    Fam -- "Yes: Gaussian / Ridge" --> S1["irls (Cholesky / faer)\nExact closed-form solution\nLatency: ~1.28 ms"]
    Fam -- "Yes: Smooth GLM / Tobit" --> S2["lbfgs (Quasi-Newton)\nArmijo line-search\nLatency: ~5.12 ms"]
    Fam -- "L1 Elastic Net Sparsity" --> S3["ccd (Coordinate Descent)\nExact zero-coefficients\nLatency: ~4.10 s"]
    Fam -- "No: Non-Convex / Multi-Modal / Glitchy" --> S4{"Derivative-Free\nEvolutionary Family"}

    S4 --> E1["shade: Adaptive Differential Evolution\nLatency: ~26.6 ms | Near-optimal"]
    S4 --> E2["cmaes: Covariance Matrix Adaptation\nLatency: ~11.6 ms | Ill-conditioned"]
    S4 --> E3["ga: Real-Coded Genetic Algorithm\nLatency: ~41.1 ms | Global exploration"]
    S4 --> E4["nes: Natural Evolution Strategies\nLatency: ~10.7 ms | Gradient estimation"]

    classDef dataBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef gateBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef gradBox fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;
    classDef evoBox fill:#4338ca,stroke:#6366f1,stroke-width:2px,color:#ffffff;

    class Data dataBox;
    class Fam,S4 gateBox;
    class S1,S2,S3 gradBox;
    class E1,E2,E3,E4 evoBox;
```

---

## 2. Mathematical Formulations of the Seven Solvers

### 1. Analytical Cholesky (`leaf_solver="irls"`)
For continuous Gaussian targets with Ridge $L_2$ regularization:
$$\min_{\beta} \frac{1}{2N} \|y - X\beta\|_2^2 + \frac{\lambda_2}{2} \|\beta\|_2^2$$

The gradient vanishes at the exact normal equations:
$$(X^T X + N \lambda_2 I) \beta^* = X^T y$$

`mltreelib-core` computes the lower-triangular Cholesky factor $L L^T = (X^T X + N \lambda_2 I)$ via `faer`, followed by forward and backward substitution. Latency is $O(P^3)$ where $P$ is feature count.

### 2. Quasi-Newton L-BFGS (`leaf_solver="lbfgs"`)
For smooth non-linear likelihoods (GLM families, Tobit censored regression), L-BFGS updates search direction $d_k = -H_k \nabla f(\beta_k)$ using two-loop recursion on past $m=5$ vector pairs:
$$s_k = \beta_{k+1} - \beta_k, \qquad y_k = \nabla f(\beta_{k+1}) - \nabla f(\beta_k)$$

Step length $\alpha_k$ satisfies the Armijo condition:
$$f(\beta_k + \alpha_k d_k) \le f(\beta_k) + c_1 \alpha_k \nabla f(\beta_k)^T d_k$$

### 3. Cyclic Coordinate Descent (`leaf_solver="ccd"`)
Minimizes the non-smooth Elastic Net objective:
$$\min_{\beta} \frac{1}{2N} \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \frac{\lambda_2}{2} \|\beta\|_2^2$$

Updates each coordinate $\beta_j$ sequentially using soft-thresholding operator $\mathcal{S}(z, \gamma) = \text{sign}(z)\max(|z| - \gamma, 0)$:
$$\beta_j \leftarrow \frac{\mathcal{S}\left( \frac{1}{N} X_j^T (y - \sum_{k \ne j} X_k \beta_k), \, \lambda_1 \right)}{\frac{1}{N} X_j^T X_j + \lambda_2}$$
Produces exact zero coefficients for irrelevant features inside terminal leaves.

### 4. Success-History Adaptive Differential Evolution (`leaf_solver="shade"`)
Derivative-free continuous optimization. Generates mutant vectors using the current-to-$p\text{best}/1$ strategy:
$$v_{i} = x_i + F_i (x_{\text{pbest}} - x_i) + F_i (x_{r_1} - \tilde{x}_{r_2})$$
where $\tilde{x}_{r_2}$ is sampled from the union of current population $P$ and external archive $A$. Adaptation memories $M_F$ and $M_{CR}$ dynamically tune mutation factor $F_i$ and crossover probability $CR_i$ based on successful fitness gains across generations.

### 5. Covariance Matrix Adaptation (`leaf_solver="cmaes"`)
Samples candidate parameter vectors $x_i \sim \mathcal{N}(m^{(g)}, (\sigma^{(g)})^2 C^{(g)})$. Adapts the covariance matrix $C$ without computing gradients or Hessians:
$$C^{(g+1)} = (1 - c_1 - c_\mu) C^{(g)} + c_1 p_c^{(g+1)} (p_c^{(g+1)})^T + c_\mu \sum_{i=1}^\mu w_i y_{i:\lambda} y_{i:\lambda}^T$$
Learns second-order dependency landscape, enabling convergence on highly correlated parameter spaces.

### 6. Real-Coded Genetic Algorithm (`leaf_solver="ga"`)
Evolves a continuous chromosome population using Simulated Binary Crossover (SBX) with distribution index $\eta_c$ and polynomial mutation with index $\eta_m$:
$$\beta_{1, 2} = \frac{1}{2} \left[ (1 \mp \gamma) x_1 + (1 \pm \gamma) x_2 \right]$$
Provides broad global exploration of non-convex and multi-modal leaf loss functions.

### 7. Separable Natural Evolution Strategies (`leaf_solver="nes"`)
Updates parameter distribution parameters $\theta = (\mu, \sigma)$ along estimated natural gradient directions:
$$\nabla_\mu J = \sum_{k=1}^\lambda w_k z_k, \qquad \nabla_\sigma J = \sum_{k=1}^\lambda w_k (z_k^2 - 1)$$
Provides fast derivative-free updates along coordinate-aligned search directions.

---

## 3. Empirical Benchmark on Real Data

We benchmark all seven solvers on the **Yeh (1998) Concrete Compressive Strength** dataset (1,030 rows, 8 features; 80% train / 20% test).

=== "Exact Cholesky (`irls`)"

    ```python
    from mltreelib import ModelTree

    # Analytical closed-form normal equations
    tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="irls", l2=0.01)
    tree.fit(X_train, y_train)
    ```

=== "Quasi-Newton (`lbfgs`)"

    ```python
    from mltreelib import ModelTree

    # Quasi-Newton L-BFGS with two-loop recursion
    tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="lbfgs", l2=0.01)
    tree.fit(X_train, y_train)
    ```

=== "Adaptive Evolutionary (`shade`)"

    ```python
    from mltreelib import ModelTree

    # Success-history based adaptive differential evolution (derivative-free)
    tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="shade", l2=0.01, seed=42)
    tree.fit(X_train, y_train)
    ```

=== "Coordinate Descent (`ccd`)"

    ```python
    from mltreelib import ModelTree

    # Cyclic coordinate descent with soft thresholding for L1 sparsity
    tree = ModelTree(max_depth=0, leaf="linear", leaf_solver="ccd", l1=0.1, l2=0.01)
    tree.fit(X_train, y_train)
    ```

### Benchmark Results (Standalone Global Fit, `max_depth=0`)

| Solver | Family | Latency (ms) | Holdout Test RMSE (MPa) | Test $R^2$ | Convergence Character |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`irls`** | Analytical Cholesky | **1.28 ms** | **10.224** | **0.631** | Exact analytical optimum in 1 step. |
| **`lbfgs`** | Quasi-Newton | **5.12 ms** | 10.287 | 0.626 | Converges in 14 iterations to machine precision. |
| **`shade`** | Continuous DE | 26.62 ms | 10.269 | 0.627 | Reaches 99.6% of exact analytical optimum without gradients. |
| **`ga`** | Genetic Algorithm | 41.15 ms | 10.360 | 0.621 | Robust global convergence across 100 generations. |
| **`cmaes`** | Evolutionary Covariance | 11.65 ms | 11.346 | 0.545 | Fast adaptation; sensitive to raw unstandardized scale. |
| **`nes`** | Natural Gradient | 10.68 ms | 16.919 | -0.011 | Fast step updates; requires standardized inputs. |
| **`ccd`** | Cyclic Coordinate Descent | 4,101.09 ms | **10.224** | **0.631** | Exact analytical match; coordinate sweeps on raw scale. |

---

## 4. Latency vs. Accuracy Pareto Diagnostic

![Pareto Frontier of mltreelib Leaf Solvers](../assets/solver_benchmark_pareto.png)

### Key Observations:

1. **Analytical & Quasi-Newton Supremacy on Convex Surfaces**:
   `irls` (1.28 ms) and `lbfgs` (5.12 ms) define the Pareto frontier for smooth convex losses. Both reach optimal holdout test RMSE (10.224 MPa).
2. **Derivative-Free Precision with SHADE**:
   `shade` achieves **10.269 MPa** (within 0.4% of the exact analytical minimum) in only 26.6 ms without evaluating a single gradient or Hessian. This makes it ideal for non-differentiable penalties, multi-objective fitness, or discontinuous loss landscapes.
3. **Tree-Decoupled Leaf Performance (`max_depth=2`, 4 Leaves)**:
   When deployed inside a partitioned model tree:
   - `irls`: 3.53 ms total fit time $\to$ **7.183 MPa** RMSE ($R^2 = 0.818$).
   - `lbfgs`: 3.30 ms total fit time $\to$ **7.583 MPa** RMSE ($R^2 = 0.797$).
   - `shade`: 29.19 ms total fit time $\to$ **7.606 MPa** RMSE ($R^2 = 0.796$).
   - `ga`: 45.43 ms total fit time $\to$ **8.005 MPa** RMSE ($R^2 = 0.774$).
   - `cmaes`: 12.45 ms total fit time $\to$ **10.285 MPa** RMSE ($R^2 = 0.626$).

---

## 5. Generic Bounded Optimization & Real-Data Hyperparameter Tuning

`mltreelib.solvers.optimize` minimizes arbitrary bounded scalar objectives through one unified contract:

```python
from mltreelib.solvers import optimize

# Black-box bounded minimization
res = optimize(
    lambda p: (p[0] - 1.0)**2 + 100.0 * (p[1] - p[0]**2)**2,  # Rosenbrock
    bounds=[(-2.0, 2.0), (-2.0, 2.0)],
    method="lbfgs_b",
    max_evaluations=200,
)
print(f"Status: {res.status}, Best objective: {res.best_value:.6f}")
```

### Standard Mathematical Landscapes Benchmark

| Problem | Method | Evaluations | Best Value | Latency (ms) | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Sphere ($D=5$)** | `lbfgs_b` | 11 | **0.000000** | 1.29 ms | converged |
| | `nelder_mead` | 150 | **0.000000** | 0.57 ms | evaluation_limit |
| | `cmaes` | 150 | **0.000000** | 0.15 ms | evaluation_limit |
| | `shade` | 150 | **0.000000** | 0.65 ms | evaluation_limit |
| | `random_search` | 64 | 11.796410 | 0.70 ms | converged |
| **Rosenbrock ($D=2$)** | `nelder_mead` | 150 | **0.000000** | 0.12 ms | evaluation_limit |
| | `cmaes` | 150 | 0.089766 | 0.12 ms | evaluation_limit |
| | `shade` | 150 | 0.412864 | 0.13 ms | evaluation_limit |
| | `lbfgs_b` | 150 | 0.610721 | 0.20 ms | evaluation_limit |
| | `random_search` | 64 | 1.177699 | 0.04 ms | converged |

### Real-Data ModelTree Hyperparameter Tuning

Optimizing `ModelTree` hyperparameters (tree depth, minimum leaf rows, L2 penalty) directly on held-out tabular validation loss (Airfoil Self-Noise):

| Method | Evaluations | Held-out RMSE (Airfoil) | Wall Time |
| :--- | :---: | :---: | :---: |
| **`random_search`** | 100 | 3.26 dB | 0.26 s |
| **`latin_hypercube`** | 100 | 3.28 dB | 0.25 s |
| **`nelder_mead`** | 100 | **2.65 dB** | 0.44 s |
| **`powell`** | 100 | 2.77 dB | 0.39 s |
| **`simulated_annealing`** | 100 | 2.92 dB | 0.45 s |

---

## 6. Production Recommendations

!!! tip "When to Select Each Solver"
    - **Use `irls`** for standard linear ridge regression (`leaf="linear"`) on Gaussian continuous targets. It uses Rust's `faer` dense Cholesky solver with minimal overhead.
    - **Use `lbfgs`** for GLMs (`leaf_family="binomial"`, `"poisson"`, `"gamma"`, `"tweedie"`, `"tobit"`).
    - **Use `ccd`** when sparse leaves are mandatory ($L_1 > 0$) to eliminate irrelevant features inside leaf equations.
    - **Use `shade`** when fitting custom non-smooth losses, quantile objectives with heavy outliers, or discontinuous simulation simulators.

---

## 7. Operational Limitations & Numerical Failure Modes

1. **Unstandardized Covariate Scaling**:
   Derivative-free evolutionary algorithms (`cmaes`, `nes`) sample isotropic search perturbations. If covariates differ by orders of magnitude (e.g., Cement $500 \, \text{kg/m}^3$ vs. Water $0.18 \, \text{kg/m}^3$), initial spherical covariance updates degenerate, yielding sub-optimal convergence (as seen with `nes` RMSE 16.919 MPa). Always standardize continuous inputs before applying evolutionary leaf solvers.
2. **Coordinate Descent Collinearity Stagnation**:
   When features exhibit high pairwise correlation ($r > 0.95$), cyclic coordinate descent (`ccd`) undergoes zig-zagging oscillations, requiring thousands of coordinate sweeps to achieve convergence (latency 4,101 ms). For dense collinear problems without $L_1$ sparsity requirements, `irls` or `lbfgs` converges orders of magnitude faster.
3. **Non-Convex Local Minima & Stochastic Reproducibility**:
   Evolutionary solvers (`shade`, `ga`, `cmaes`) are stochastic algorithms. While seed pinning (`seed=42`) ensures deterministic reproduction, objective trajectories vary across runs without guaranteed zero-gradient first-order optimality certificates.
4. **Box Bounds vs. Coupled Constraints in `optimize`**:
   The `lbfgs_b` bounded optimizer supports box constraints $l_i \le x_i \le u_i$. For coupled joint budget or volume constraints $\sum_i a_i x_i \le B$, use dual multiplier bisection or `scipy.optimize.minimize(method="SLSQP")`.

---

## References

- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
- Tanabe, R., & Fukunaga, A. S. (2014). "Improving the search performance of SHADE using linear population size reduction." *IEEE CEC*, 1658-1665.
- Hansen, N. (2016). "The CMA evolution strategy: A tutorial." *arXiv preprint arXiv:1604.00772*.
