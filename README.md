# mltreelib — Machine Learning Tree-Based Library

[![PyPI](https://img.shields.io/pypi/v/mltreelib.svg)](https://pypi.org/project/mltreelib/)
[![CI](https://github.com/jkapila/mltreelib/actions/workflows/test.yaml/badge.svg)](https://github.com/jkapila/mltreelib/actions/workflows/test.yaml)
[![Docs](https://github.com/jkapila/mltreelib/actions/workflows/deploy.yaml/badge.svg)](https://jkapila.github.io/mltreelib/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)](pyproject.toml)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg?logo=rust)](crates/mltreelib-core/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Zero Sklearn](https://img.shields.io/badge/dependencies-zero--sklearn-brightgreen)](#)
[![Missing Data](https://img.shields.io/badge/missing--data-first--class-blueviolet)](#)

Tree-based machine learning with a **pure Rust core** (`mltreelib-core`) and a high-performance **Python frontend** and an **R** source package for ModelTree.

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

See the [changelog](CHANGELOG.md) for release history and the [release guide](RELEASING.md) for publishing steps.

### R ModelTree

From a v0.4.15 checkout, use R, a C compiler, and Cargo to build and install the source package. The source package includes the Rust core. Cargo needs access to its dependencies.

```sh
r/build-r-package.sh dist
R CMD INSTALL dist/mltreelib_0.4.15.tar.gz
Rscript --vanilla r/airquality_model_tree.R
```

The package exposes ModelTree fitting and prediction. The source tarball was checked on Ubuntu 24.04 with R 4.3.3. [Holdout results and source checks](references/research/2026-11-21-r-airquality-results.md)

### Example suites and agent skill

Follow the [example suites](https://jkapila.github.io/mltreelib/examples/) for runnable walkthroughs. Start with the [quickstart](https://jkapila.github.io/mltreelib/quickstart/) to fit and inspect your first model tree.

Install the mltreelib skill for an agent that supports the Skills CLI. The skill includes recipes for selecting, fitting, evaluating, and saving models. Install the Python package separately to run the recipes. [Skill source](https://github.com/jkapila/mltreelib/tree/main/skills/mltreelib)

```sh
npx skills add jkapila/mltreelib --skill mltreelib
```

---

## Core Capabilities

- **Zero Scikit-Learn Dependency**: Lightweight standalone runtime; clean `fit`/`predict` ergonomics without packaging bloat.
- **First-Class Missing Data**: `None` and `NaN` are normalized at ingestion; never panics; automatic surrogate split search and branch routing.
- **Modernized Quinlan Lineage**:
  - **M5 Model Tree**: Multivariate linear models at every leaf with Ridge/ElasticNet regularization.
  - **R `summary(lm)` Inspection**: Rich statistical tables for every leaf node (coefficients, standard errors, $t$-statistics, $p$-values, $R^2$, and $F$-tests).
  - **C5.0 Rule Extraction**: Automatic rule extraction via `extract_rules()` with condition simplification and binomial upper-confidence pessimistic pruning.
  - **Cubist Neighbor Adjustments**: Test-time manifold corrections via `predict_adjusted(X, neighbors=5)` using training feature neighbors.
  - **GritBot**: Contextual anomaly mining detecting surprising condition-dependent outliers with subpopulation context.
- **Rich Splitting Criteria**:
  - *Standard Regression*: `sse`, `friedman_mse`, `absolute_error` (L1 median), `huber`, `poisson`, `sdr` (standard deviation reduction).
  - *Distributional & Censored*: `tweedie` (compound Poisson-Gamma deviance for $1 < p < 2$), `heteroscedastic` (Gaussian NLL deviance isolating variance breaks with identical means), `tobit` (censored continuous hurdle deviance), `cv` (coefficient-of-variation reduction).
  - *Survival & Causal*: `logrank` (LeBlanc-Crowley test statistic), `tarone_ware` & `peto` (weighted log-rank), `martingale` (O(N) streaming residual scan), `gray` (Gray's subdistribution test for competing risks), `survival_brier` (multi-cause proper scoring), `causal` / `uplift` (Athey & Imbens treatment effect heterogeneity).
  - *Classification*: `entropy`, `gini`, `gain_ratio` (C4.5 normalized information gain), `twoing` (CART balanced contrast).
  - *Custom*: User-defined Python callables receiving lazy $(y, w)$ candidate slices.
- **Distributional Leaves**: Gaussian, Binomial (Logit/Probit), Poisson (Rate), Softmax / Multinomial ($C \ge 3$), Survival & Competing Risks (Nelson-Aalen / Kaplan-Meier / Aalen-Johansen CIF), Tobit (Censored), Tweedie, and Heteroscedastic $(\mu, \sigma)$.
- **Unified Clustering (`ClusterTree`)**: Unified facade supporting Predictive Clustering Trees (PCT), Unsupervised Random Forest (URF), HDBSCAN (condensed hierarchy, GLOSH outlier scores), and CLTree.
- **Unified Anomaly Detection (`AnomalyTree` & `GritBot`)**: Unified facade supporting Extended Isolation Forest (EIF), Density Estimation Trees (DET), Half-Space Trees (HST), Random Cut Trees (RCT), Active feedback trees (AAD), Context sequence trees (CAT), and RuleQuest GritBot.
- **Native Categorical Encoders (14)**: CatBoost, Target, Embedding, Weight of Evidence (WoE), James-Stein, Leave-One-Out, M-Estimator, Quantile, One-Hot, Base-N, Sum, Tfidf, Count, and Hashing.

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
for leaf in model.leaves():
    summary = model.leaf_summary(leaf["id"])
    if summary:
        print(summary)
```

```text
======================================================================
Leaf Node 0 (Leaf ID 0) [n=20]
----------------------------------------------------------------------
Residual standard error: 0.0931 on 17 degrees of freedom
Multiple R-squared:  0.9996,  Adjusted R-squared:  0.9996
F-statistic: 21487.89 on 2 and 17 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)              0.0544       0.0456       1.19       0.2331    
x0                       1.9699       0.0187     105.45      < 2e-16 ***
x1                       0.4994       0.0029     169.59      < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================
```

### 2. C5.0 Rules & Cubist Neighbor Adjustments

```python
# Extract and prune decision rules with pessimistic confidence bounds
ruleset = model.extract_rules(prune=True, cf=0.25)
print(f"Extracted {len(ruleset)} rules:")
for rule in ruleset.rules:
    print(f"IF {' AND '.join(rule.conditions)} THEN y = {rule.prediction:.4f} (err={rule.errors}/{rule.cover})")

# Predict with RuleSet
rule_preds = ruleset.predict(X_test)

# Cubist-style k-NN sample neighbor inference adjustment
adjusted_preds = model.predict_adjusted(
    X_test,
    neighbors=5,
)
print(f"Adjusted neighbor predictions: {[round(p, 4) for p in adjusted_preds[:3]]}")
```

```text
Extracted 4 rules:
IF x0 <= 5.0000 AND x1 > 22.0000 THEN y = 23.0642 (err=0/31)
IF x0 > 3.0000 AND x1 <= 22.0000 THEN y = 18.3591 (err=0/25)
Adjusted neighbor predictions: [0.0497, 2.2362, 4.5648]
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
uplifts = causal_tree.predict(X_new, kind="uplift")
print(f"Predicted CATE uplifts (first 4): {[round(u, 4) for u in uplifts[:4]]}")
```

```text
Predicted CATE uplifts (first 4): [0.4215, 0.3891, -0.1502, 0.7314]
```

### 4. Unsupervised Clustering Trees (`ClusterTree`)

The unified `ClusterTree` interface provides a consistent scikit-learn-shaped API across divisive and density clustering engines (`kind="pct"`, `"urf"`, `"hdbscan"`, `"cltree"`):

```python
from mltreelib.cluster import ClusterTree

# 1. Predictive Clustering Tree (divisive clustering with rule extraction)
pct = ClusterTree(kind="pct", max_depth=2)
pct.fit(X)
print(f"Discovered clusters: {set(pct.labels_)}")
print(f"Out-of-sample predictions: {pct.predict([[0.0, 0.0], [4.0, 9.0]])}")

# 2. Unsupervised Random Forest (proximity-based manifold clustering)
urf = ClusterTree(kind="urf", n_clusters=3, seed=42)
urf.fit(X)
print(f"URF cluster assignments: {set(urf.labels_)}")
```

```text
Discovered clusters: {0, 1, 2, 3}
Out-of-sample predictions: [0, 3]
URF cluster assignments: {0, 1, 2}
```
*(Also supports `kind="hdbscan"` for density hierarchy/GLOSH outliers and `kind="cltree"` for bounding boxes)*

### 5. Density & Outlier Detection Suite (`AnomalyTree`)

The unified `AnomalyTree` facade wraps multiple tree-based anomaly engines (`kind="eif"`, `"det"`, `"hst"`, `"rct"`, `"active"`, `"context"`) under a consistent API:

```python
from mltreelib import GritBot
from mltreelib.anomaly import AnomalyTree

# 1. Extended Isolation Forest (oblique hyperplanes without ghost artifacts)
detector = AnomalyTree(kind="eif", contamination=0.05, seed=42)
detector.fit(X)
scores = detector.predict(X, kind="score")
preds = detector.predict(X)         # 1 = anomaly, 0 = inlier
print(f"Anomaly score range: [{min(scores):.3f}, {max(scores):.3f}]")
print(f"Total flagged anomalies: {sum(preds)}")

# 2. RuleQuest GritBot (context-dependent conditional anomalies)
grit = GritBot(min_items=10, sig_threshold=0.01)
anomalies = grit.scan(X, col_names=["x0", "x1"])
for a in anomalies[:2]:
    print(a["summary"])
```

```text
Anomaly score range: [0.391, 0.847]
Total flagged anomalies: 5
case 51: [0.0000] x1 = 999.00 (51 cases, mean 100.00, 98.0% <= 100.00)
```
*(Also supports `kind="det"` for piecewise density, `kind="hst"` for streaming, `kind="rct"` for co-displacement, and `kind="active"` for expert feedback)*

### 6. Statistical & Evolutionary Solvers

`mltreelib-core` embeds a high-performance solver engine for linear and generalized linear leaf models:
- **Exact OLS / Ridge (`leaf_solver="irls"`)**: Fast Cholesky factorization with QR fallback and complete R-style hypothesis testing tables ($t$-stats, $p$-values, $R^2$, $F$-test).
- **Coordinate Descent (`leaf_solver="ccd"`)**: Cyclic active-set coordinate descent for $L_1$ Lasso, ElasticNet, and monotone coefficient constraints.
- **Quasi-Newton L-BFGS (`leaf_solver="lbfgs"`)**: Two-loop recursion L-BFGS for GLM deviance minimization (Binomial logit, Poisson rate, Tweedie, Tobit).
- **Evolutionary Solvers (`leaf_solver="cmaes" | "shade" | "nes" | "ga"`)**: Derivative-free and natural-gradient global search for non-convex or discontinuous loss surfaces.

For a standalone weighted fit, use `mltreelib.solvers.fit`. It accepts
`sample_weight` and returns a fitted `FitResult` with coefficients, predictions,
convergence status, and the solver stop reason.

For a bounded scalar objective of any shape, use `mltreelib.solvers.optimize`.
It takes a callable, one `(lower, upper)` pair per parameter, and a method
name, and returns the best point, the best objective, the evaluation count,
and the termination status. Methods include `random_search`, `latin_hypercube`,
`lbfgs_b`, `bfgs`, `nonlinear_cg`, `nelder_mead`, `powell`, `simulated_annealing`,
`cobyla`, `basin_hopping`, `cmaes`, `shade`, `nes`, and `ga`.
See the [generic optimizer guide](docs/examples/generic-optimizer.md).

```python
# Fit global OLS regression with full R-style statistical tables
ols = ModelTree(max_depth=0, leaf="linear", leaf_solver="irls", l2=0.0)
ols.fit(X_train, y_train)
print(ols.leaf_summary(ols.leaves()[0]["id"]))

# Fit sparse ElasticNet with active-set Coordinate Descent
lasso = ModelTree(max_depth=0, leaf="linear", leaf_solver="ccd", l1=0.1, l2=0.01)
lasso.fit(X_train, y_train)
print("Sparse coefficients:", lasso.leaves()[0]["coef"])
```

```text
======================================================================
Leaf Node 0 (Leaf ID 0) [n=120]
----------------------------------------------------------------------
Residual standard error: 0.1941 on 116 degrees of freedom
Multiple R-squared:  0.9981,  Adjusted R-squared:  0.9980
F-statistic: 20070.14 on 3 and 116 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)             -0.0149       0.0360      -0.41       0.6793    
x0                       3.5147       0.0149     236.09      < 2e-16 ***
x1                      -1.9684       0.0299     -65.94      < 2e-16 ***
x2                       0.4984       0.0120      41.46      < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================
Sparse coefficients: [3.5134, -1.9667, 0.4969]
```
*(Runnable script: [`examples/guides/solvers_overview.py`](examples/guides/solvers_overview.py))*

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
print("Realized variance shares:", {k: round(v, 4) for k, v in truth.items()})
```

```text
Realized variance shares: {'numeric': 0.8319, 'categorical': 0.3178, 'hierarchy': 0.2984, 'interaction': 0.0871, 'noise': 0.0632, 'total': 2.4317}
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
- [x] **Modular model controls and persistence**. Configure growth and leaves separately. Save ModelTree, PCT, and EIF models as versioned JSON. [Spec 20](specs/20-architecture-two-fold-control-decoupling-serialization-and-agent-skill.md)
- [x] **Snapshots for additional clustering and anomaly engines**. Save and restore fitted DET, half-space, random cut, active anomaly, context anomaly, CLTree, HDBSCAN, and unsupervised proximity forest models.
- [x] **Streaming multiclass splits**. Use class-frequency histograms for multiclass gains and opt into at most 256 bins per numeric feature.
- [x] **Live Training Logging & Telemetry**: Configurable real-time verbosity levels (`verbose=1/2/3`) for split search progress, candidate gains, and leaf GLM convergence.
- [x] **Anomaly missing values**. Preserve missing numeric observations across engines and support context scoring and active feedback.
- [x] **Clustering missing values**. Pass missing numeric observations to the native engines.
- [x] **Survival and competing risks**: Gray's subdistribution test, weighted log-rank (Tarone-Ware, Peto-Prentice), Aalen-Johansen CIF with simplex invariant, and checked IPCW metrics (Harrell/Uno C-index, time-dependent Brier score matrix, IBS).
- [x] **Binary isotonic calibration**. Fit the mapping on held-out or out-of-fold probabilities. Evaluate the mapping on independent data.
- [x] **Generic optimizer**: Minimize a scalar objective through `mltreelib.solvers.optimize` with 14 methods, one shared budget, and one status contract. [Spec 26](specs/26-generic-global-and-constrained-optimizers.md)
- [ ] **Ensembles & Committees**: Cubist-style iterative boosting committees with leaf model smoothing.
- [x] **R source package**. Build the Linux-checked ModelTree frontend from the repository. [R example and checks](references/research/2026-11-21-r-airquality-results.md)
- [ ] **R package distribution**. Prepare a CRAN submission.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
