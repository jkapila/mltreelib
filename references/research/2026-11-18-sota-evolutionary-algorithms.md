# State-of-the-Art Evolutionary & Derivative-Free Optimization Solvers

**Date**: 2026-11-18  
**Author**: Jitin Kapila / mltreelib core research  
**Status**: Implemented in `crates/mltreelib-core/src/solver/evolutionary.rs`  

---

## 1. Executive Summary & Research Motivation

In machine learning and statistical modeling, gradient-based methods (IRLS, L-BFGS, Coordinate Descent) dominate when the loss objective $\mathcal{L}(\theta)$ is convex, smooth, and twice continuously differentiable. However, several critical practical regimes break gradient assumptions:
1. **Non-Differentiable / Robust Loss Functions**: Quantile loss (pinball), Least Absolute Deviations ($L_1$ median), Huber loss with non-smooth thresholds, and Tobit censored likelihoods.
2. **Non-Convex & Multi-Modal Regularization**: Hard $L_0$ feature sparsity, non-convex Bridge penalties ($L_q$ with $q < 1$), SCAD, and MCP penalties.
3. **Discrete & Combinatorial Constraints**: Integer bounds, strict monotone gating, step penalties, and combinatorial subset selection.
4. **Ill-Conditioned & Non-Separable Landscapes**: Correlated predictors where condition numbers $\kappa(X^T X) > 10^8$ cause first-order gradient descent to oscillate or stall.

Derivative-free optimization (DFO) and Evolutionary Algorithms (EAs) treat the objective as a black box $\beta \mapsto \mathcal{L}(\beta)$, relying purely on function queries without requiring gradient or Hessian evaluations.

---

## 2. Competitive Landscape & SOTA Benchmark Winners

Comprehensive benchmarks across the international Black-Box Optimization Benchmarking (BBOB/COCO) suite and IEEE Congress on Evolutionary Computation (CEC) real-parameter competitions identify three dominant paradigms:

| Family | SOTA Algorithm | Authors / Year | Core Mechanism | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **Evolution Strategy (ES)** | **CMA-ES** | Hansen & Ostermeier (2001) | Full covariance matrix $C \in \mathbb{R}^{P \times P}$ adaptation via isotropic ($p_\sigma$) and anisotropic ($p_c$) evolution paths. | Ill-conditioned, non-separable continuous spaces ($P < 100$). |
| **Differential Evolution (DE)** | **SHADE / L-SHADE** | Tanabe & Fukunaga (2013/2014) | `current-to-pbest/1` mutation with external archive $A$, historical memories $M_F, M_{CR}$, Lehmer mean updates. | Multi-modal, rugged, deceptive landscapes (CEC competition winner). |
| **Natural Evolution Strategies** | **sNES / NGL** | Wierstra et al. (2014) | Natural gradient ascent on Gaussian distribution manifold $\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ with rank fitness shaping. | High-dimensional separable problems ($P > 100$, $O(P)$ time/memory). |
| **Real-Coded GA (RCGA)** | **SBX + PolyMut** | Deb & Agrawal (1995) | Simulated Binary Crossover (SBX, $\eta_c=20$) mimicking single-point crossover + polynomial mutation ($\eta_m=20$). | Discrete, bounded, and genetic exploration baselines. |

---

## 3. Algorithmic Formulations

### 3.1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

CMA-ES models the search distribution as a multivariate normal distribution $\mathcal{N}(m, \sigma^2 C)$.

1. **Sampling Offspring**:
   $$\beta_k = m + \sigma y_k, \quad y_k = L z_k, \quad z_k \sim \mathcal{N}(0, I_P)$$
   where $L L^T = C$ is the Cholesky factor of the covariance matrix.

2. **Recombination**:
   $$m^{(g+1)} = \sum_{i=1}^\mu w_i \beta_{i:\lambda}, \quad y_w = \sum_{i=1}^\mu w_i y_{i:\lambda}$$
   with super-linear rank weights $w_i \propto \ln(\mu + 0.5) - \ln(i)$.

3. **Cumulative Step-Size Adaptation (CSA / Path Length Control)**:
   $$p_\sigma \leftarrow (1 - c_\sigma) p_\sigma + \sqrt{c_\sigma(2 - c_\sigma)\mu_{\text{eff}}} \, C^{-1/2} y_w$$
   $$\sigma \leftarrow \sigma \exp\left( \frac{c_\sigma}{d_\sigma} \left( \frac{\|p_\sigma\|}{\mathbb{E}\|\mathcal{N}(0, I)\|} - 1 \right) \right)$$

4. **Covariance Matrix Adaptation**:
   $$p_c \leftarrow (1 - c_c) p_c + h_\sigma \sqrt{c_c(2 - c_c)\mu_{\text{eff}}} \, y_w$$
   $$C \leftarrow (1 - c_1 - c_\mu + c_1 \delta(h_\sigma)) C + c_1 p_c p_c^T + c_\mu \sum_{i=1}^\mu w_i y_{i:\lambda} y_{i:\lambda}^T$$

---

### 3.2. SHADE (Success-History Based Adaptive Differential Evolution)

Classical Differential Evolution requires meticulous manual tuning of the mutation scale factor $F \in (0, 1]$ and crossover rate $CR \in [0, 1]$. SHADE resolves this via an adaptive memory mechanism:

1. **Parameter Sampling**:
   For each individual $i$, select memory slot $r_i \in \{1, \dots, H\}$ uniformly at random:
   $$CR_i \sim \mathcal{N}(M_{CR, r_i}, 0.1), \quad F_i \sim \text{Cauchy}(M_{F, r_i}, 0.1)$$

2. **`current-to-pbest/1` Mutation**:
   $$v_i = x_i + F_i (x_{\text{pbest}} - x_i) + F_i (x_{r1} - x_{r2})$$
   where $x_{\text{pbest}}$ is randomly drawn from the top $p \cdot 100\%$ individuals ($p \approx 0.15$), $x_{r1}$ from the active population, and $x_{r2}$ from the union of the active population and an external archive of inferior solutions $A$.

3. **Lehmer Mean Success Adaptation**:
   Successful parameter values generating fitness improvements $\Delta f_k = f(x_k) - f(u_k)$ update the memory:
   $$M_{F, k} \leftarrow \text{mean}_L(S_F) = \frac{\sum w_k S_{F, k}^2}{\sum w_k S_{F, k}}, \quad w_k = \frac{\Delta f_k}{\sum_m \Delta f_m}$$
   The Lehmer mean prevents small $F$ values from dominating the memory, averting premature stagnation.

---

### 3.3. sNES (Separable Natural Evolution Strategies)

sNES maximizes the expected fitness $J(\theta) = \mathbb{E}_{\beta \sim p_\theta} [f(\beta)]$ by following the natural gradient:
$$\tilde{\nabla}_\theta J = F^{-1} \nabla_\theta J$$
where $F$ is the Fisher Information Matrix of the search distribution family.

Under a diagonal Gaussian $\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$, the Fisher information matrix is diagonal and can be inverted in closed form in $O(P)$ time:
$$\nabla_\mu = \sum_{k=1}^\lambda u_k z_k, \quad \nabla_\sigma = \sum_{k=1}^\lambda u_k (z_k^2 - 1)$$
where $u_k$ are rank-based utility values invariant to arbitrary strictly monotonic fitness scaling.

---

## 4. Integration into Tree Leaves & Standalone Models

In `mltreelib-core`:
- All evolutionary solvers consume the zero-copy `DesignMatrix` trait ([crates/mltreelib-core/src/solver/design.rs](file:///mnt/d/Programs/mltreelib/crates/mltreelib-core/src/solver/design.rs)).
- No allocations occur per evaluation; evaluations operate directly over node row subsets `rows: &[usize]`.
- Monotonicity constraints (feature $j$ non-decreasing or non-increasing) are satisfied via projection within `eval_fitness`.
- Tree leaves can be fitted with `leaf_solver="cmaes"`, `leaf_solver="shade"`, or `leaf_solver="nes"`.

---

## 5. References

1. Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159-195.
2. Tanabe, R., & Fukunaga, A. (2013). Success-history based parameter adaptation for Differential Evolution. *2013 IEEE Congress on Evolutionary Computation*, 71-78.
3. Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J., & Schmidhuber, J. (2014). Natural evolution strategies. *Journal of Machine Learning Research*, 15(1), 949-980.
4. Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. *Complex Systems*, 9(2), 115-148.
5. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. *arXiv preprint arXiv:1703.03864*.
6. Rapin, J., & Teytaud, O. (2018). Nevergrad: An open source tool for derivative-free optimization. *GitHub repository*.
