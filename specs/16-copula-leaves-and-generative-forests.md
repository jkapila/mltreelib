# Spec 16: Copula Leaves, Generative Forests & Probabilistic Circuits

## Problem Statement

Standard tree models (CART, Random Forests, XGBoost) are strictly discriminative: they model conditional expectations $E[Y \mid X]$ or class probabilities $P(Y \mid X)$ under fixed input features. This imposes severe limitations:
1. **Inability to Model Joint Distributions**: They cannot answer joint probabilistic queries, evaluate joint likelihoods $P(X_1, \dots, X_d)$, or generate realistic synthetic multi-feature samples.
2. **Failure Under Arbitrary Missingness**: If test instances arrive with varying or unobserved subsets of features $X_S$, discriminative models fail or require ad-hoc imputation, whereas true probabilistic models compute exact marginals $P(X_S) = \int P(X_S, X_{\setminus S}) dX_{\setminus S}$.
3. **Tail Dependence Blindness**: Standard regression assumes Gaussian or homoscedastic errors, failing to model asymmetric tail risk (e.g. financial crash contagion or joint catastrophic insurance claims) where variables exhibit heavy upper or lower tail dependence.
4. **Intractable Generative Alternatives**: Deep generative models (VAEs, Diffusion, Normalizing Flows) require expensive GPU training, suffer from mode collapse on tabular data, and lack exact closed-form marginalization capabilities.

## Solution

Implement a native, pure-Rust probabilistic generative tree architecture in `mltreelib-core` unifying **Generative Forests (GeFs / Probabilistic Circuits)** with **Vine Copula Leaf Models**:

1. **Generative Forests as Tractable Probabilistic Circuits**:
   - Re-interpret tree ensembles as Sum-Product Networks (SPNs) / Probabilistic Circuits (Correia et al. 2020):
     - **Sum Nodes**: Represent the forest ensemble mixture with weights $w_b = 1/B$.
     - **Product Nodes**: Represent internal orthogonal decision splits, decomposing the joint distribution into mutually independent rectangular partitions.
     - **Tractable Leaf Distributions**: Model marginals within hyper-rectangles using piecewise-uniform, Gaussian, or kernel density distributions.
   - **Tractable Inference Engines**:
     - **Exact Marginalization**: Compute $P(X_S)$ for any feature subset $S$ in $O(B \cdot \text{depth})$ time without Monte Carlo sampling.
     - **Exact Conditioning**: Compute conditional densities $P(X_{\text{target}} \mid X_{\text{observed}} = x)$ for arbitrary variable configurations.
     - **Exact Synthetic Generation**: Ancestral sampling: choose tree $b \sim \text{Uniform}(1 \dots B)$, traverse to leaf $L$, sample $x \sim P_L(X)$.
2. **Pair-Copula Constructions (Vine Copula Leaves)**:
   - Implement bivariate copula building blocks in terminal leaves via Sklar's theorem:
     $$F(x_1, x_2) = C(F_1(x_1), F_2(x_2))$$
   - Support parametric copula families:
     - **Gaussian Copula**: Symmetric correlation without tail dependence.
     - **Student-$t$ Copula**: Symmetric heavy tails (joint extreme co-movements).
     - **Clayton Copula**: Asymmetric strong lower tail dependence (joint market crashes).
     - **Gumbel Copula**: Asymmetric strong upper tail dependence (joint peak surges).
     - **Frank Copula**: Symmetric association with weak tail dependence.
   - Fit pair-copula parameters via maximum pseudo-likelihood (Kendall's $\tau$ inversion) inside leaves.
3. **Copula Quantile Regression Trees**:
   - Fit copula quantile curves $C^{-1}(\tau \mid U_1, \dots, U_p)$ at terminal leaves, provably eliminating quantile crossing issues across multiple quantiles $\tau \in (0, 1)$.

## User Stories

1. As a synthetic data engineer, I want `GenerativeForest(n_estimators=50).sample(n_samples=100_000)`, so that I can generate realistic synthetic tabular data with preserved non-linear correlations and boundaries.
2. As a probabilistic ML researcher, I want `model.marginal_likelihood(X_subset)`, computing exact marginal probabilities for arbitrary feature subsets in milliseconds without MCMC.
3. As a quantitative risk manager, I want `leaf="clayton_copula"`, so that the tree leaf models asymmetric lower-tail crash dependence between financial asset returns.
4. As an actuarial scientist, I want `leaf="gumbel_copula"`, so that the joint probability of simultaneous catastrophic insurance payouts is accurately modeled in the upper tail.
5. As a fraud detection engineer, I want `model.log_prob(X_test)`, so that joint density estimation tree scores isolate multivariate anomalies in low-density regions.
6. As a clinician, I want `model.predict_conditional(target="blood_pressure", given={"age": 55, "bmi": 28.5})`, imputing missing clinical values conditional on exact observed patient features.
7. As an analyst, I want `model.kendall_tau_matrix()`, reporting non-parametric rank correlation matrices per leaf partition.
8. As a production engineer, I want synthetic generation to run in parallel across CPU threads using Rust's `rayon` with zero Python GIL contention.
9. As a data privacy officer, I want differential privacy noise injection options in generative leaf histograms, preventing memorization of training instances.
10. As a domain modeler, I want `CopulaQuantileRegressor(quantiles=[0.05, 0.5, 0.95])`, generating monotonicity-guaranteed quantile curves without quantile crossing.
11. As an API user, I want `GenerativeForest` to offer `fit(X)`, `sample(n)`, `log_prob(X)`, and `impute(X_with_nans)` with zero scikit-learn dependency.
12. As a systems developer, I want all copula density evaluations and cumulative probability transforms to be vector-accelerated with numerically stable log-domain evaluations.

## Implementation Decisions

- **Crate Module Structure**:
  - Add module `crates/mltreelib-core/src/copula/`:
    - `bivariate.rs`: `CopulaFamily` enum (`Gaussian`, `StudentT`, `Clayton`, `Gumbel`, `Frank`) with CDF, PDF, and conditional inverse sampling formulas.
    - `fitting.rs`: Kendall's $\tau$ non-parametric rank estimator and pseudo-maximum likelihood parameter inversion.
    - `leaf.rs`: `CopulaLeaf` fitting marginal empirical CDFs and bivariate dependence structures per partition.
  - Add module `crates/mltreelib-core/src/generative/`:
    - `circuit.rs`: Probabilistic circuit graph representation (Sum, Product, Leaf nodes).
    - `forest.rs`: `GenerativeForest` constructing density estimation trees via unsupervised variance or log-likelihood splits.
    - `inference.rs`: Exact marginalization and conditional density algorithms.
    - `sampling.rs`: Ancestral sampling engine generating synthetic records.
- **Tractable Inference Contract**:
  - For any leaf $L$ defining bounds $[a_j, b_j]$ on feature $X_j$:
    $$\int P(X_j \mid L) dX_j = 1 \quad \text{if unobserved (marginalized out)}$$
  - Product nodes multiply active feature probabilities: $P(X_S \mid L) = \prod_{j \in S} P(X_j \mid L)$.
  - Sum node evaluates uniform mixture: $P(X_S) = \frac{1}{B} \sum_{b=1}^B P(X_S \mid L_b)$.
- **Python Surface (`crates/mltreelib-py`)**:
  - `mltreelib.generative.GenerativeForest(n_estimators=50, max_depth=8, min_rows=10, leaf="uniform", random_state=None)`
  - `mltreelib.copula.CopulaTreeRegressor(family="clayton", quantiles=[0.1, 0.5, 0.9])`

## Testing Decisions

- **Exact Marginal Likelihood Integration Test**:
  - Fit `GenerativeForest` on a 2D uniform box $[0, 1]^2$.
  - Marginalize out $X_2$ and compute $P(X_1 \in [0.2, 0.6])$.
  - Verify result is $0.4000 \pm 10^{-4}$ (exact analytical match).
- **Synthetic Data Distribution Match**:
  - Fit on multimodal Gaussian mixture dataset.
  - Sample 50,000 synthetic rows.
  - Verify Kolmogorov-Smirnov test does not reject distribution identity ($p > 0.05$) on marginals, and synthetic covariance matrix matches empirical covariance within $2\%$.
- **Clayton Lower-Tail Dependence Verification**:
  - Fit `Clayton` copula leaf on simulated lower-tail correlated data ($\lambda_L = 2^{-1/\theta} = 0.5$).
  - Verify estimated $\hat{\theta}$ recovers true parameter within $5\%$ error.
- **Quantile Non-Crossing Verification**:
  - Fit `CopulaTreeRegressor` for $\tau \in [0.01, 0.05, 0.5, 0.95, 0.99]$.
  - Evaluate on 10,000 test points.
  - Assert strictly that $\hat{y}_{\tau_1}(x) \le \hat{y}_{\tau_2}(x)$ for all $\tau_1 < \tau_2$ with zero violations.

## Out of Scope

- Arbitrary high-order $D > 20$ fully saturated regular vine decomposition trees (focus is on bivariate pair copulas and generative hyper-rectangles).
- Continuous image or video synthesis (tabular and multi-feature numeric synthesis only).

## Further Notes

- Generative Forests represent a breakthrough bridge between classical tree structures and modern probabilistic circuits (SPNs), offering exact tractable likelihoods that deep neural networks cannot compute while preserving tree interpretability.
