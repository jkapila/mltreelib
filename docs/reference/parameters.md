# Parameters Reference

Comprehensive documentation for all hyperparameters, knobs, and training controls in `mltreelib`.

Inspired by the parameter architecture of **CatBoost** and **LightGBM**, every knob in `mltreelib` is typed, deterministic, and mapped directly to the underlying `mltreelib-core` Rust engine.

---

## Quick Navigation
- [1. Core Tree Architecture & Sizing](#1-core-tree-architecture-sizing)
- [2. Splitting & Search Space Controls](#2-splitting-search-space-controls)
- [3. Regularization, Purity Gates & Split Penalties](#3-regularization-purity-gates-split-penalties)
- [4. Statistical Hypothesis Stopping & Pre-Selection](#4-statistical-hypothesis-stopping-pre-selection)
- [5. Categorical Handling & Monotonicity Constraints](#5-categorical-handling-monotonicity-constraints)
- [6. Pruning, Honesty & Missing Value Routing](#6-pruning-honesty-missing-value-routing)
- [7. Leaf Models, GLM Families & Penalized Solvers](#7-leaf-models-glm-families-penalized-solvers)
- [8. Growth and Leaf Controls](#8-growth-and-leaf-controls)
- [9. Telemetry & Execution Controls](#9-telemetry-execution-controls)

---

## 1. Core Tree Architecture & Sizing

Controls how deep and wide the tree can grow before termination.

Set the classification class count with `GrowthControl(n_classes=...)`. Existing calls can supply `n_classes` as a flat constructor keyword, a fit keyword, or a legacy `LeafControl` setting.

| Parameter | Type | Default | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`max_depth`** | `int` | `5` | `[1, 64]` | Maximum tree depth from root to any leaf. Set to `1` for decision stumps; increase to `8-12` to capture deep multi-way interactions. |
| **`max_leaf_nodes`** | `int \| None` | `None` | `[2, \infty)` | Maximum number of terminal leaf nodes. If set, growth follows best-first priority queue until leaf limit is reached. |
| **`min_rows`** | `int` | `50` | `[1, \infty)` | Minimum number of samples required to form a terminal leaf (`minbucket` in R / `min_child_samples` in LightGBM). Prevents overfitting to tiny sub-populations. |
| **`minsplit`** | `int \| None` | `None` | `[2, \infty)` | Minimum number of samples required to attempt a split. If `None`, defaults to `2 * min_rows`. |
| **`minprob`** | `float` | `0.02` | `(0.0, 0.5)` | Minimum fraction of training rows required in a terminal node ($N_{\text{leaf}} / N \ge \text{minprob}$). Safeguards against isolated slivers. |
| **`min_gain`** | `float` | `1e-9` | `[0.0, \infty)` | Minimum impurity reduction required to justify creating child nodes. Analogous to `min_split_gain` in LightGBM or `gamma` in XGBoost. |

---

## 2. Splitting & Search Space Controls

Governs candidate generation, feature subsampling, and candidate evaluations.

| Parameter | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`gain`** | `str \| None` | `None` | `"entropy"`, `"gini"`, `"variance"` | Split evaluation objective. If `None`, automatically resolves to `"entropy"` for classification and `"variance"` for regression. |
| **`split_eval`** | `str` | `"fast"` | `"fast"`, `"exact"` | Split evaluation engine. `"fast"` utilizes SIMD vectorization and streaming histogram accumulators; `"exact"` re-evaluates all continuous thresholds without binning. |
| **`split_precision`** | `int \| None` | `4` | `[1, 16]` | Decimal digit rounding applied to candidate thresholds. Rounds continuous splits (e.g. `3.1416` instead of `3.1415926535`) to prevent floating-point overfitting and improve explainability. |
| **`split_cols`** | `list[int] \| list[str] \| None` | `None` | Feature subset | Whitelist of features permitted to form splits. Useful when reserving specific features solely for leaf linear models. |
| **`col_sample`** | `float \| None` | `None` | `(0.0, 1.0]` | Fraction of features randomly subsampled at each split decision (`colsample_bytree` in XGBoost / `mtry` ratio). Injects stochastic decorrelation across ensemble trees. |
| **`col_weights`** | `list[float] \| None` | `None` | Non-negative weights | Prior weights determining the sampling probability of each feature during candidate selection. Higher weights bias search toward key business drivers. |
| **`interaction_groups`** | `list[list[int \| str]] \| None` | `None` | Feature sets | Enforces interaction constraints: features may only appear on the same root-to-leaf path if they belong to a shared interaction group. Prevents disallowed cross-variable leakage. |
| **`presort`** | `bool` | `False` | `True`, `False` | Whether to pre-sort continuous features at the root. Reduces per-node sort overhead at the cost of initial fit memory. |

---

## 3. Regularization, Purity Gates & Split Penalties

Structural regularization mechanisms designed to suppress noisy split boundaries.

| Parameter | Type | Default | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`purity_gate`** | `float \| None` | `None` | `(0.0, 1.0)` | Purity stopping criterion. If a node's target standard deviation or Gini impurity falls below this fraction of global variance, node splitting halts early. |
| **`purity_gate_metric`**| `str` | `"sd"` | `"sd"`, `"iqr"`, `"mad"` | Metric for evaluating node purity: standard deviation (`"sd"`), interquartile range (`"iqr"`), or median absolute deviation (`"mad"`). |
| **`tail_quantile`** | `float \| None` | `None` | `(0.0, 0.25)` | Extrema protection threshold. Penalizes splits that isolate extreme target quantiles into tiny leaves. |
| **`imbalance_penalty`**| `float` | `0.0` | `[0.0, 1.0]` | Penalizes asymmetric splits where $|N_{\text{left}} - N_{\text{right}}| \gg 0$. Smooths partition tree balance. |
| **`alpha`** | `float` | `0.0` | `[0.0, \infty)` | Structural complexity penalty subtracted from the candidate split gain ($G' = G - \alpha$). Forces simpler tree topologies. |

---

## 4. Statistical Hypothesis Stopping & Pre-Selection

Replaces heuristic gain thresholds with formal statistical hypothesis testing (analogous to CHAID and `partykit::ctree`).

| Parameter | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`split_test`** | `str` | `"none"` | `"none"`, `"f_test"`, `"permutation"` | Formal statistical test evaluated on prospective splits. Halts recursion when null hypothesis of independence cannot be rejected. |
| **`split_alpha`** | `float` | `0.05` | `(0.0, 1.0)` | Significance level threshold ($p < \alpha$) required to accept a split candidate. |
| **`bonferroni`** | `bool` | `False` | `True`, `False` | Applies Bonferroni multiplicity correction ($p < \alpha / P$) adjusting for the number of tested feature candidates. Eliminates selection bias toward high-cardinality features. |
| **`preselect`** | `str` | `"none"` | `"none"`, `"variance"`, `"correlation"` | Heuristic screening step executed before split scans to discard uninformative columns. |
| **`min_relevance`** | `float \| None` | `None` | `[0.0, 1.0]` | Minimum correlation/mutual information required to include a feature in split search. |

---

## 5. Categorical Handling & Monotonicity Constraints

Native handling of categorical variables and directional domain knowledge.

| Parameter | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`cat_order`** | `str \| None` | `None` | `"target"`, `"frequency"`, `"natural"` | Categorical level ordering strategy. `"target"` orders levels by target mean/rate, converting combinatorial $2^{K-1}$ subset search into linear $O(K)$ boundary scans (Fisher's exact theorem). |
| **`cat_search`** | `str` | `"cumulative"` | `"cumulative"`, `"exhaustive"` | Partition search algorithm across ordered levels. `"cumulative"` checks consecutive level subsets; `"exhaustive"` evaluates all partition subsets (small $K$ only). |
| **`monotone`** | `list[int] \| None` | `None` | `[-1, 0, +1]` | Monotonicity constraints per feature: `+1` forces non-decreasing target response, `-1` forces non-increasing, `0` unconstrained. |
| **`tree_monotone`** | `list[int] \| None` | `None` | `[-1, 0, +1]` | Enforces monotonic split step direction across parent-child tree partitions. |
| **`leaf_monotone`** | `list[int] \| None` | `None` | `[-1, 0, +1]` | Enforces non-negative/non-positive sign constraints on leaf GLM linear coefficients. |

---

## 6. Pruning, Honesty & Missing Value Routing

Controls post-fit pruning, missing value handling, and sample splitting.

| Parameter | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`prune`** | `str` | `"cp"` | `"cp"`, `"cost_complexity"`, `"bic"`, `"aic"`, `"none"` | Post-pruning algorithm. `"cp"` matches CART/rpart complexity pruning; `"bic"` penalizes number of active parameters using Bayesian Information Criterion. |
| **`cp_alpha`** | `float` | `5.0` | `[0.0, \infty)` | Complexity pruning penalty multiplier. Higher values produce shallower, more general trees. |
| **`honesty`** | `bool` | `False` | `True`, `False` | Enables honest estimation (Wager & Athey 2018). Tree splits are learned on a training sub-sample, while leaf models are fitted on a held-out estimation sub-sample. Guarantees asymptotic normality. |
| **`honesty_fraction`** | `float` | `0.5` | `(0.0, 1.0)` | Proportion of rows allocated to the split-learning sub-sample when `honesty=True`. |
| **`maxsurrogate`** | `int` | `0` | `[0, 10]` | Number of surrogate split variables stored per node (CART style). When a primary split feature is missing during prediction, the highest-agreement surrogate variable routes the sample. |

---

## 7. Leaf Models, GLM Families & Penalized Solvers

Defines the parametric or non-parametric local model fitted within each terminal leaf.

| Parameter | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`leaf`** | `str` | `"linear"` | `"linear"`, `"mean"`, `"median"`, `"quantile"`, `"huber"`, `"mars"`, `"knn"` | Mathematical formulation of the leaf model: piecewise linear OLS (`"linear"`), constant mean (`"mean"`), median (`"median"`), Huber robust regression (`"huber"`), adaptive splines (`"mars"`), or non-parametric nearest neighbors (`"knn"`). |
| **`knn_k`** | `int` | `5` | `[1, \infty)` | Number of nearest neighbors queried at leaf inference time. |
| **`knn_variant`** | `str` | `"uniform"` | `"uniform"`, `"distance"`, `"exponential"`, `"dual"`, `"double"` | Weighting scheme for leaf KNN: `"uniform"` (unweighted), `"distance"` (inverse distance $1/(d+\epsilon)^p$), `"exponential"` ($\exp(-\gamma d)$), `"dual"` (distance $\times$ compactness), or `"double"` (DW-KNN: distance consistency $\times$ neighbor reliability $v_i$). |
| **`knn_metric`** | `str` | `"euclidean"` | `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`, `"cosine"`, `"correlation"`, `"canberra"`, `"braycurtis"`, `"gower"`, `"mahalanobis"` | Distance proximity metric across leaf feature coordinates. `"gower"` normalizes mixed continuous/categorical features and handles missing entries; `"mahalanobis"` applies diagonal variance scaling. |
| **`knn_k_validity`** | `int` | `5` | `[1, \infty)` | Secondary neighborhood size for precomputing neighbor reliability $v_i$ (DW-KNN) or local compactness during `fit()`. |
| **`knn_gamma`** | `float \| None` | `None` | `(0.0, \infty)` | Exponential spatial decay rate parameter. If `None`, automatically calibrated by the inverse median neighbor distance within the queried leaf. |
| **`knn_p`** | `float` | `2.0` | `(0.0, \infty)` | Power parameter $p$ for Minkowski metric or inverse distance weighting ($1 / (d + \epsilon)^p$). |
| **`leaf_family`** | `str` | `"gaussian"` | `"gaussian"`, `"binomial"`, `"poisson"`, `"tweedie"`, `"tobit"`, `"softmax"` | Target distribution family for Generalized Linear Model (GLM) leaves. `"gaussian"` uses identity link, `"binomial"` uses logit link, `"poisson"` uses log link. |
| **`leaf_solver`** | `str` | `"irls"` | `"irls"`, `"lbfgs"`, `"ccd"`, `"cmaes"`, `"shade"`, `"nes"`, `"ga"` | Local optimization solver for leaf regression models: Iteratively Reweighted Least Squares / exact normal Cholesky (`"irls"`), Quasi-Newton L-BFGS with Armijo line search (`"lbfgs"`), Cyclic Coordinate Descent with active set (`"ccd"`), or SOTA continuous evolutionary algorithms: Covariance Matrix Adaptation Evolution Strategy (`"cmaes"`), Success-History Adaptive Differential Evolution (`"shade"`), Separable Natural Evolution Strategies (`"nes"`), or Real-Coded Genetic Algorithm with SBX crossover (`"ga"`). |
| **`leaf_cols`** | `str` | `"all"` | `"all"`, `"path_used"` | Feature policy for leaf linear models. `"all"` fits over all features (with L1/L2 shrinkage); `"path_used"` restricts the leaf model strictly to features split along that leaf's ancestry path (winnowing). |
| **`l1`** | `float` | `0.0` | `[0.0, \infty)` | Lasso regularization penalty on leaf linear coefficients. Drives irrelevant leaf weights strictly to zero. |
| **`l2`** | `float` | `0.5` | `[0.0, \infty)` | Ridge regularization penalty on leaf linear coefficients. Stabilizes collinear inversion matrices in high-dimensional leaves. |
| **`tau`** | `float` | `0.5` | `(0.0, 1.0)` | Quantile level when `leaf="quantile"`. Optimizes asymmetric pinball loss for conditional quantile forecasting. |
| **`tweedie_p`** | `float` | `1.5` | `[1.0, 2.0]` | Power parameter $p$ for Tweedie exponential dispersion modeling. Ideal for zero-inflated insurance claims ($1 < p < 2$). |
| **`tobit_threshold`** | `float` | `0.0` | $\mathbb{R}$ | Censoring threshold $y_c$ when `leaf_family="tobit"`. Models left-censored Gaussian variables via expectation-maximization. |
| **`leaf_digest`** | `bool` | `True` | `True`, `False` | Maintains streaming $t$-digests in leaves. Enables fast empirical prediction intervals (`predict_interval(coverage=0.9)`) without distributional assumptions. |
| **`mars_degree`** | `int` | `1` | `[1, 3]` | Maximum interaction degree for MARS spline hinge functions within leaves. |
| **`mars_terms`** | `int` | `10` | `[2, 50]` | Maximum number of basis terms retained in leaf spline models. |
| **`mars_penalty`** | `float` | `2.0` | `[0.0, 10.0]` | Generalized cross-validation (GCV) penalty per knot for MARS leaf models. |

---

## 8. Growth and Leaf Controls

Use the growth settings in sections 1–6 to configure the partition. Use the leaf settings in section 7 to configure the model fitted in each region. Under constraints, `tree_monotone` governs splits and `leaf_monotone` governs leaf coefficients. Supply data, feature names, types, and row weights to `fit`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`growth_control`** | `GrowthControl \| dict \| None` | `None` | Groups split search, stopping, feature sampling, honesty, and pruning settings. |
| **`leaf_control`** / **`control`** | `LeafControl \| dict \| None` | `None` | Groups the leaf form, family, solver, penalties, and other leaf settings. |
| **`train_groups`** | `bool` | `True` | Whether to preserve grouped hierarchical entities during leaf retraining. |
| **`train_ids`** | `bool` | `False` | Whether to index training entity row keys in leaf sample caches (`leaf_samples()`). |

For solver selection inside a leaf control, use `solver="lbfgs"` or another compatible solver. Flat callers can continue to use `leaf_solver`. Some leaf models use a dedicated fitting method. See the [model reference](model.md#configure-growth-and-leaves) for typed, dictionary, and flat examples.

| Leaf model | Compatible explicit solver choices |
| --- | --- |
| Gaussian linear | `irls`, `lbfgs`, `ccd`, `shade`, `cmaes`, `ga`, `nes` |
| Linear binomial, probit, Poisson, Gamma, Tweedie, or Tobit | `irls`, `lbfgs` |
| Leaves with a dedicated estimator | Use the estimator selected by the leaf form and family. |

Use L-BFGS with zero L1 penalty and no monotone leaf constraints. It supports ridge penalties. Specialized estimators reject explicit `solver` selections.

---

## 9. Telemetry & Execution Controls

| Parameter | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`verbose`** | `int` | `0` | `0`, `1`, `2`, `3` | Training verbosity. `0` = silent; `1` = live depth progress, per-depth summaries, and fit complete summary; `2` = stream per-node split candidates, winning features, threshold, gain, and leaf reasons; `3` = stream leaf solver optimization (GLM convergence iterations, deviance step, linear fit stats). |
| **`seed`** | `int` | `42` | `int` | Deterministic random seed controlling feature subsampling (`col_sample`) and honest partition splits. |
