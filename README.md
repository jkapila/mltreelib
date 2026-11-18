# mltreelib — Machine Learning Tree-Based Library

[![PyPI](https://img.shields.io/pypi/v/mltreelib.svg)](https://pypi.org/project/mltreelib/)
[![CI](https://github.com/jkapila/mltreelib/actions/workflows/test.yaml/badge.svg)](https://github.com/jkapila/mltreelib/actions/workflows/test.yaml)
[![Docs](https://github.com/jkapila/mltreelib/actions/workflows/deploy.yaml/badge.svg)](https://jkapila.github.io/mltreelib/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)](pyproject.toml)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg?logo=rust)](crates/mltreelib-core/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Zero Sklearn](https://img.shields.io/badge/dependencies-zero--sklearn-brightgreen)](#)
[![Missing Data](https://img.shields.io/badge/missing--data-first--class-blueviolet)](#)

Tree-based machine learning with a **pure Rust core** (`mltreelib-core`) and a high-performance **Python frontend** (with **R** bindings underway). 

Modern tree and ensemble architectures often outperform flat tabular foundation models on complex structured data. `mltreelib` brings together the best of classical algorithmic lineages (Quinlan's M5, C4.5, C5.0, Cubist, GritBot; Breiman's CART and URF; Athey & Imbens' Causal Trees) modernized in memory-safe Rust with sub-millisecond execution, zero scikit-learn dependency, first-class missing data routing, and rich distributional leaves.

---

## Installation

Install pre-compiled native binary wheels directly from PyPI:

```sh
pip install mltreelib
```

For development or compiling from source (requires Rust toolchain and [uv](https://docs.astral.sh/uv/)):

```sh
git clone https://github.com/jkapila/mltreelib.git
cd mltreelib
uv sync
uv run maturin develop --release
```

---

## Core Capabilities

- **Zero Scikit-Learn Dependency**: Lightweight standalone runtime; clean `fit`/`predict` ergonomics without packaging bloat.
- **First-Class Missing Data**: `None` and `NaN` are normalized at ingestion; never panics; automatic surrogate split search and branch routing.
- **Modernized Quinlan Lineage**:
  - **M5 Model Tree**: Multivariate linear models at every leaf with Ridge/ElasticNet regularization.
  - **R `summary(lm)` Inspection**: Rich statistical tables for every leaf node (coefficients, standard errors, $t$-statistics, $p$-values, $R^2$, and $F$-tests).
  - **C5.0 Rule Extraction**: Automatic rule extraction via `export_rules()` with condition simplification and binomial upper-confidence pessimistic pruning.
  - **Cubist Neighbor Adjustments**: Test-time manifold corrections via `predict_with_neighbors(X, k=5, neighbors_weight=0.5)` using training feature neighbors.
  - **GritBot**: Contextual anomaly mining detecting surprising condition-dependent outliers with subpopulation context.
- **Rich Splitting Criteria**:
  - *Standard Regression*: `sse`, `friedman_mse`, `absolute_error` (L1 median), `huber`, `poisson`, `sdr` (standard deviation reduction).
  - *Distributional & Censored*: `tweedie` (compound Poisson-Gamma deviance for $1 < p < 2$), `heteroscedastic` (Gaussian NLL deviance isolating variance breaks with identical means), `tobit` (censored continuous hurdle deviance), `cv` (coefficient-of-variation reduction).
  - *Survival & Causal*: `logrank` (Leblanc-Crowley test statistic), `martingale` (survival residual variance reduction), `causal` / `uplift` (Athey & Imbens treatment effect heterogeneity).
  - *Classification*: `entropy`, `gini`, `gain_ratio` (C4.5 normalized information gain), `twoing` (CART balanced contrast).
  - *Custom*: User-defined Python callables receiving lazy $(y, w)$ candidate slices.
- **Distributional Leaves**: Gaussian, Binomial (Logit/Probit), Poisson (Rate), Softmax / Multinomial ($C \ge 3$), Survival (Nelson-Aalen / Kaplan-Meier), Tobit (Censored), Tweedie, and Heteroscedastic $(\mu, \sigma)$.
- **Unsupervised Clustering Trees**: HDBSCAN (Condensed Tree, GLOSH outlier scores), TrieKNN (categorical prefix tree + leaf numeric KNN), Predictive Clustering Trees (PCT), CLTree, and Unsupervised Random Forest (URF).
- **Anomaly & Density Trees**: DensityEstimationTree (DET), ExtendedIsolationTree (EIF), HalfSpaceTree (HST), RandomCutTree (RCT), ActiveAnomalyTree (AAD), ContextAnomalyTree (CAT), and GritBot.
- **Native Categorical Encoders (12)**: CatBoost, Target, Embedding, Weight of Evidence (WoE), James-Stein, Leave-One-Out, M-Estimator, Quantile, One-Hot, Base-N, Sum, Tfidf, and Hashing.

---

## Quickstart

### 1. Supervised ModelTree & Leaf Summaries

```python
from mltreelib import ModelTree

# Build a multivariate model tree with C4.5 / M5 lineage
model = ModelTree(
    max_depth=4,
    min_rows=10,
    leaf="linear",              # 'linear', 'mean', 'median', 'quantile', 'huber'
    l2=0.1,                     # Ridge penalty on leaf coefficients
    gain="sse",                 # 'sse', 'tweedie', 'heteroscedastic', 'tobit', 'cv', 'causal', etc.
    split_precision=0.001,      # Threshold scan grid
    leaf_digest=True,           # T-digest for post-hoc conditional quantiles
    prune="cp",                 # Cost-complexity pruning with cp_alpha
)

model.fit(X_train, y_train)

# Predictions and conditional quantiles (NAs route automatically)
y_pred = model.predict(X_test)
q90 = model.quantile(X_test, tau=0.90)

# Statistical summary of leaf linear models (like R's summary(lm))
for leaf in model.leaf_models():
    print(leaf.summary())
    # Output:
    # Residuals: Min 1Q Median 3Q Max
    # Coefficients: Estimate Std. Error t value Pr(>|t|)
    # Multiple R-squared, F-statistic, p-value
```

### 2. C5.0 Rules & Cubist Neighbor Adjustments

```python
# Extract and prune decision rules with pessimistic confidence bounds
ruleset = model.export_rules(prune=True, confidence=0.25)
print(f"Extracted {ruleset.n_rules} rules:")
for rule in ruleset.rules:
    print(f"IF {rule.antecedent} THEN y = {rule.consequent} (err={rule.error_rate:.2%})")

# Predict with RuleSet
rule_preds = ruleset.predict(X_test)

# Cubist-style k-NN sample neighbor inference adjustment
adjusted_preds = model.predict_with_neighbors(
    X_test,
    k=5,
    neighbors_weight=0.5,
)
```

### 3. Distributional & Causal Splitting

```python
# Tweedie model tree for zero-heavy insurance claims (1 < p < 2)
tweedie_tree = ModelTree(
    max_depth=4,
    gain="tweedie",
    tweedie_p=1.5,
)
tweedie_tree.fit(X_claims, y_amounts)

# Heteroscedastic tree: detects variance regimes even with identical means
hetero_tree = ModelTree(
    max_depth=3,
    gain="heteroscedastic",     # Minimizes Gaussian negative log-likelihood
)
hetero_tree.fit(X_volatility, y_returns)

# Causal tree: maximizes treatment effect divergence between partitions
causal_tree = ModelTree(
    max_depth=3,
    gain="causal",
)
causal_tree.fit(X_covariates, (y_outcomes, treatment_indicators))
uplifts = causal_tree.predict_uplift(X_new)
```

### 4. Unsupervised Clustering Trees

```python
from mltreelib import HDBSCAN, UnsupervisedRandomForest, TrieKNN

# HDBSCAN density clustering & GLOSH anomaly detection
hdb = HDBSCAN(min_cluster_size=15, min_samples=5)
hdb.fit(X)
labels = hdb.labels                 # -1 = noise / outlier
outliers = hdb.outlier_scores       # GLOSH score in [0, 1]
tree = hdb.condensed_tree()         # cluster hierarchy & stability

# Unsupervised Random Forest proximity clustering
urf = UnsupervisedRandomForest(n_trees=50, max_depth=6)
urf.fit(X, n_clusters=4)
prox = urf.proximity_matrix         # [N x N] leaf co-occurrence similarity
```

### 5. Density & Outlier Detection Suite

```python
from mltreelib import (
    DensityEstimationTree,
    ExtendedIsolationTree,
    HalfSpaceTree,
    RandomCutTree,
    ActiveAnomalyTree,
    ContextAnomalyTree,
    GritBot,
)

# Oblique hyperplane cuts without axis-aligned ghost artifacts
eif = ExtendedIsolationTree(max_depth=8, extension_level=1)
eif.fit(X)
anomaly_scores = eif.score_samples(X)

# RuleQuest GritBot: surprising condition-dependent subgroup outlier mining
grit = GritBot(min_cases=10, filter_percent=1.0)
grit.fit(X)
anomalies = grit.detect_anomalies(X)
for a in anomalies:
    print(f"Row {a.row_id}: Value {a.value:.2f} violates local context: {a.condition}")
```

---

## Synthetic Data Generator & Structure Recovery

Evaluate learning algorithms on synthetic ground truth with calibrated latent structure:

```python
from mltreelib import generate

data = generate(
    n_rows=10_000,
    n_numeric=5,
    cardinalities=[3, 8, 50, 200],
    hierarchy_levels=3,
    branching=[4, 10, 25],
    group_effect_strength=0.6,
    hierarchy_effect_strength=0.3,
    interaction_strength=0.2,
    noise=0.1,
    tail_behavior="bathtub",        # U-shaped tail noise matching DAR phenomena
    seed=42,
)

X = data["columns"]                 # Typed columns (None = NA)
y = data["target"]                  # Continuous target or discrete class
truth = data["truth"]["variance"]   # Realized variance shares per latent component
```

---

## Technical Articles & Case Studies

In-depth technical perspectives and domain use cases:

- **[Why India's Digital Lending Boom Needs Glass-Box Trees Over Black-Box Boosters](https://jkapila.github.io/mltreelib/blog/why-credit-scoring-needs-glass-box-trees/)**: Monotonic shape constraints, WOE binning, and auditable credit scorecards.
- **[Zero-Imputation Predictive Maintenance for Industrial Sensor Feeds](https://jkapila.github.io/mltreelib/blog/zero-imputation-predictive-maintenance/)**: Why synthetic imputation corrupts sensor failure dynamics, and how native surrogate routing and linear leaves solve sensor dropouts.
- **[Uplift Modeling from Scratch: Maximizing ROI with Honest Causal Trees](https://jkapila.github.io/mltreelib/blog/uplift-modeling-with-causal-trees/)**: Why standard response modeling fails, and how sample-split honest trees isolate incremental treatment uplift.
- **[Why We Rebuilt Quinlan's Cubist and C5.0 in Rust](https://jkapila.github.io/mltreelib/blog/modernizing-quinlan-classics-in-rust/)**: Modernizing 1990s C rule engines with memory-safe SIMD Cholesky factorization in Rust.
- **[Beyond Mean Squared Error: Distributional Decision Trees for Heavy Tails, Insurance, and Censored Data](https://jkapila.github.io/mltreelib/blog/beyond-mse-distributional-trees/)**: Tweedie compound Poisson-Gamma claims, Tobit censored log-likelihood, and heteroscedastic volatility regimes.
- **[Breaking the Step Function: Why Model Trees Beat Gradient Boosters on Continuous Physics](https://jkapila.github.io/mltreelib/blog/breaking-step-functions-model-trees-vs-gbdt/)**: Overcoming orthogonal step staircases on continuous physical manifolds with piecewise linear leaves and hierarchical smoothing.

---

## TabArena Benchmarks & Elo Arena

`mltreelib` is measured against competitive tabular baselines across **Rust, Python, and R**:
- **Regression Arena**: 18 benchmark datasets, 36 models (`mltreelib`, `rpart`, `ctree`, `lmtree`, `cubist`, `grf`, `mboost`, `ranger`, `bart`, `earth`, `LightGBM`, `CatBoost`, `EBM`, `forust`, `perpetual`), round-robin Elo on RMSE.
- **Classification Arena**: 24 TabArena datasets (OpenML suite 457), 28 models across Python, Rust, and R.

Every entry tracks native categorical handling and execution resources under strict process-group timeouts. Runbooks and manifests live in [`benchmarks/`](benchmarks/).

---

## Roadmap

- [x] **Data & Feature Layer**: typed ingest, 12 categorical encoders, time-series transforms, synthetic ground-truth generator.
- [x] **Core ModelTree**: multivariate linear leaves, Ridge/L1 regularization, t-digest conditional quantiles, cost-complexity pruning.
- [x] **Quinlan Modernizations**:
  - [x] M5 model-tree regression with R `summary(lm)` statistics.
  - [x] C5.0 rule extraction with pessimistic confidence-limit pruning.
  - [x] Cubist k-NN training-neighbor inference adjustments.
  - [x] GritBot contextual anomaly mining.
- [x] **Distributional & Specialized Splitting Criteria**:
  - [x] Tweedie unit quasi-deviance ($1 < p < 2$).
  - [x] Heteroscedastic Gaussian NLL deviance (variance break isolation).
  - [x] Tobit censored regression hurdle deviance.
  - [x] Coefficient of variation reduction (`gain="cv"`).
  - [x] Martingale survival residual variance reduction.
  - [x] Causal / uplift treatment effect heterogeneity.
  - [x] Survival Log-Rank statistic.
- [x] **Unsupervised & Anomaly Trees**: HDBSCAN, URF, CLTree, TrieKNN, PCT, DET, EIF, HST, RCT, AAD, CAT.
- [x] **Modern Leaf Optimization**: Quasi-Newton L-BFGS, Cyclic Coordinate Descent, and SOTA continuous evolutionary solvers (CMA-ES, SHADE, sNES, RCGA) augmenting exact Cholesky and IRLS.
- [ ] **Streaming Multiclass Split Optimization**: $O(N \cdot C)$ streaming class-frequency histogram and 256-bin histogram subtraction trick for multiclass splits.
- [x] **Live Training Logging & Telemetry**: Configurable real-time verbosity levels (`verbose=1/2/3`) for split search progress, candidate gains, and leaf GLM convergence.
- [ ] **Anomaly facade**: preserve missing values across engines, expose context scoring and active feedback, and define threshold tie behavior.
- [ ] **Clustering missingness**: remove implicit `None` to zero conversion in unified clustering engines and check native missing-value behavior.
- [ ] **Survival and uplift metrics**: expose checked censoring-aware and randomized policy evaluation workflows beyond the examples.
- [ ] **Probability recalibration**: fit post-hoc calibrators on separate validation or out-of-fold predictions, and evaluate them on independent data. Calibration plots already assess reliability.
- [ ] **Ensembles & Committees**: Cubist-style iterative boosting committees with leaf model smoothing.
- [ ] **R Frontend Package**: CRAN package wrapping `mltreelib-core`.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
