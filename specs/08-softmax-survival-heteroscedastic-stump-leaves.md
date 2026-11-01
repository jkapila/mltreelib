# [LEAF-EXPANSION] Softmax, Survival, and Heteroscedastic Stump Leaves

## Problem Statement

Users require model trees that handle complex real-world data regimes beyond standard homoscedastic regression and binary classification:
1. **Multiclass Targets ($C \ge 3$)**: Classification targets with more than two discrete classes currently have no native path in `ModelTree`, forcing users to binarize or leave the library.
2. **Right-Censored Survival Data**: Time-to-event outcomes $(T, E)$ with censoring cannot be modeled with ordinary regression; treating duration as continuous introduces catastrophic survival bias and distorts hazard estimates.
3. **Heteroscedastic Regimes**: Datasets with input-dependent variance (noise that scales with features) currently output only conditional mean $\mu(x)$, giving no measure of prediction uncertainty $\sigma(x)$ or regime-dependent risk.

To deliver these capabilities reliably, users need robust leaf estimation for all three families, validated on single-node stumps (`max_depth=0`) with end-to-end working examples, while deferring complex multi-level tree splitting criteria to a subsequent optimization phase.

## Solution

Extend the `leaf_family` abstraction in the Rust core and Python frontend to support three specialized distributional families on stumps and single-leaf models:

1. **`Softmax` Leaf Family**:
   - Multinomial logit / softmax parameterization for target classes $y \in \{0, \dots, C-1\}$.
   - Solves multi-class coefficients via iterative reweighted least squares (IRLS) / Newton-Raphson.
   - Provides `predict_proba(X)` returning calibrated class probability matrices of shape $(N, C)$.

2. **`Survival` Leaf Family**:
   - Models right-censored time-to-event pairs $(T, E)$ where $T \ge 0$ is duration and $E \in \{0, 1\}$ is the event indicator.
   - Computes non-parametric Nelson-Aalen cumulative hazard $\hat{\Lambda}_0(t)$ and Kaplan-Meier survival curves $\hat{S}(t)$.
   - Establishes mathematical contract for LeBlanc & Crowley martingale residual reduction $M_i = E_i - \hat{\Lambda}_0(T_i)$ for future recursive split search.
   - Provides `predict_survival(X, time_grid)` and `predict_hazard(X)` endpoints.

3. **`Heteroscedastic` (Gaussian NLL) Leaf Family**:
   - Jointly estimates location $\mu(x)$ and dispersion/variance $\sigma^2(x)$ under Gaussian negative log-likelihood (NLL).
   - Provides `predict_dist(X)` exposing mean $\hat{\mu}$, standard deviation $\hat{\sigma}$, and parametric confidence intervals.

Each family will be fully trained, calibrated, and demonstrated on single-leaf stumps (`max_depth=0`), verified against canonical benchmark datasets (`veteran.csv` for survival, multiclass datasets, and heteroscedastic synthetic/real benchmarks).

## User Stories

1. As a data scientist with multiclass classification data, I want to set `leaf_family="softmax"`, so that my model tree natively fits multi-category targets without manual binarization.
2. As a machine learning engineer, I want `predict_proba(X)` on a softmax tree to return an $(N, C)$ probability array where each row sums to $1.0$, so that I can use standard probability-based decision thresholds.
3. As a researcher, I want softmax leaves to support both constant probabilities (`leaf="mean"`) and linear logit equations (`leaf="linear"`), so that I can capture intra-leaf feature effects.
4. As a clinician analyzing clinical trial data, I want `leaf_family="survival"` to accept duration and censoring indicator pairs $(T, E)$, so that censored subjects do not distort duration predictions.
5. As an epidemiologist, I want survival leaves to compute non-parametric Nelson-Aalen cumulative hazards, so that baseline risk is estimated honestly without premature parametric assumptions.
6. As a survival analysis practitioner, I want `predict_survival(X, times)` to return survival probabilities $\hat{S}(t \mid x)$ across specified time grids, so that I can plot individual survival curves.
7. As a financial risk analyst, I want `leaf_family="heteroscedastic"` to model continuous targets with input-dependent volatility, so that high-variance regimes are explicitly quantified.
8. As a quantitative researcher, I want heteroscedastic leaves to output both location $\hat{\mu}(x)$ and scale $\hat{\sigma}(x)$, so that I can build asymmetric prediction intervals.
9. As a developer, I want all three new leaf families to run cleanly on a single-node stump (`max_depth=0`), so that leaf estimation can be tested and verified independently of tree splitting heuristics.
10. As a performance engineer, I want softmax and heteroscedastic IRLS solvers to leverage the existing `faer` linear algebra stack, so that stump fits execute in sub-millisecond times.
11. As a user training survival models, I want the system to validate that duration is strictly non-negative ($T \ge 0$) and event flags are binary ($E \in \{0, 1\}$), so that invalid data is caught with descriptive error messages.
12. As an interactive Python user, I want runnable showcase examples in `examples/` demonstrating each of the three leaf families on real benchmark datasets, so that I can quickly adopt the API.
13. As an analyst evaluating lung cancer outcomes, I want to run survival estimation on `datasets/external/veteran.csv`, so that model survival curves can be compared against standard biostatistical references.
14. As a benchmark runner, I want heteroscedastic leaves to optimize Gaussian negative log-likelihood deviance rather than plain sum of squared errors, so that scale parameters converge to maximum likelihood estimates.
15. As a library maintainer, I want the leaf model interface to expose parameter serialization, so that models can be inspected, logged, and deployed without loss of internal distributional parameters.
16. As a developer extending tree growth later, I want survival leaves to compute martingale residuals $M_i = E_i - \hat{\Lambda}_0(T_i)$ at the node level, so that future recursive splitting can stream residuals through existing least-squares split routines.
17. As an API user, I want `tree.retrain(...)` to support refitting softmax, survival, and heteroscedastic leaf parameters on new or filtered data partitions, so that leaf parameters adapt without structural re-splitting.
18. As a developer, I want invalid combinations of leaf shapes and families (e.g. Huber loss requested on survival pairs) to raise clear configuration errors, so that misuse is prevented at parameter ingestion.

## Implementation Decisions

### Decision 1: Leaf Model Representation & Polymorphism

In the Rust core, extend the `LeafModel` enum to encompass the three new target families:
- `Softmax`: Stores class count $C$, class labels, and per-class intercept/coefficient vectors $\beta_c \in \mathbb{R}^P$ for $c \in \{0, \dots, C-1\}$ (with class 0 anchored to zero for identification in multinomial logit, or full symmetric formulation with ridge regularization).
- `Survival`: Stores event time grid $t_{(1)} < \dots < t_{(K)}$, Nelson-Aalen cumulative hazard jumps $d_j / n_j$, and Kaplan-Meier product-limit estimates $\hat{S}(t_{(j)}) = \prod_{k \le j} (1 - d_k / n_k)$.
- `Heteroscedastic`: Stores location parameters $\beta_\mu \in \mathbb{R}^P$ and scale parameters (scalar $\hat{\sigma}$ or log-linear scale coefficients $\gamma \in \mathbb{R}^P$ where $\ln \sigma = \mathbf{x}^T \gamma$).

### Decision 2: Stump-First Training Scope

Tree construction when `max_depth == 0` evaluates only the root node leaf fitting:
- The recursive splitting search and child node partitioning are completely bypassed.
- Target vectors for Softmax accept integer class codes $0 \dots C-1$.
- Target inputs for Survival accept either a structured 2D array / tuple of `(time, event)` or distinct target columns in ingestion.
- Target inputs for Heteroscedastic accept standard continuous 1D floats.

### Decision 3: Mathematical Formulation of Solvers

1. **Softmax Solver**:
   - Multi-class cross-entropy loss with softmax link $p_{ic} = \frac{\exp(\eta_{ic})}{\sum_k \exp(\eta_{ik})}$.
   - Newton-Raphson / IRLS updates with block diagonal or damped gradient steps using `faer` Cholesky/LU decomposition.
   - For constant leaf (`leaf="mean"`), exact closed-form empirical class frequencies $p_c = \frac{\sum_i w_i I(y_i = c)}{\sum_i w_i}$.
2. **Survival Estimator**:
   - Sort observations by duration $T_i$.
   - Compute risk set sizes $n_j = \sum_{i: T_i \ge t_{(j)}} w_i$ and event counts $d_j = \sum_{i: T_i = t_{(j)}, E_i = 1} w_i$.
   - Cumulative hazard $\hat{\Lambda}_0(t) = \sum_{t_{(j)} \le t} \frac{d_j}{n_j}$.
   - Survival curve $\hat{S}(t) = \exp(-\hat{\Lambda}_0(t))$ or Fleming-Harrington / Kaplan-Meier product.
   - Martingale residuals $M_i = E_i - \hat{\Lambda}_0(T_i)$ computed and cached on the leaf.
3. **Heteroscedastic Solver**:
   - Location step: weighted linear regression / mean solve for $\hat{\mu}$.
   - Scale step: compute sample residuals $e_i = y_i - \hat{\mu}_i$.
   - For constant scale: $\hat{\sigma}^2 = \frac{1}{\sum w_i} \sum w_i e_i^2$.
   - For linear log-scale: IRLS on auxiliary target $z_i = \ln(e_i^2 + \epsilon)$ regressed on path features to estimate $\ln \sigma(x)$.

### Decision 4: Unified Prediction Surface via `kind` Knob

Instead of sprawling into disconnected prediction functions (`predict_survival`, `predict_hazard`, `predict_dist`, etc.), unify predictions through a centralized `kind` parameter on `predict(x, kind="response", **kwargs)`:

- `kind="response"` (default):
  - Gaussian / Gamma / Tweedie / Tobit / Poisson: conditional mean $\hat{\mu}$.
  - Binomial / Probit: response probability $P(Y=1)$.
  - Softmax: hard predicted class label $\arg\max_c P(Y=c)$.
  - Survival: estimated median / expected survival duration $\hat{T}$.
  - Heteroscedastic: conditional mean $\hat{\mu}$.
- `kind="proba"`:
  - Softmax: $(N, C)$ class probability matrix.
  - Binomial / Probit: 1D probability array $P(Y=1)$ (or 2-column $(N, 2)$ class probabilities).
- `kind="dist"`:
  - Heteroscedastic: tuple or 2D array of location and scale `(mean, std)`.
- `kind="survival"`:
  - Survival: survival probabilities $\hat{S}(t \mid x)$ evaluated over `times` grid (defaults to observed event time deciles if `times=None`).
- `kind="hazard"`:
  - Survival: cumulative hazard $\hat{\Lambda}(t \mid x)$ over `times` grid.
- `kind="leaf_id"`:
  - Integer leaf membership (equivalent to `.apply(x)`).

Provide `predict_proba(x)` as a standard, backward-compatible convenience alias delegating directly to `predict(x, kind="proba")`.

### Decision 5: Alignment with `id_col` Entity Routing & Hierarchical `retrain`

The new leaf families must integrate seamlessly with `leaf_samples(x, id_col=...)` and `retrain(...)`:

1. **Entity Routing via `leaf_samples`**:
   - `tree.leaf_samples(x, id_col="patient_id")` groups entity primary keys into `{leaf_id: [id_1, id_2, ...]}`.
   - For stumps (`max_depth=0`), all rows map to `leaf_id=0`.
2. **Hierarchical Retrain API Integration**:
   - Extend `LeafControl` in Rust core (`crates/mltreelib-core/src/leaf.rs`) and PyO3 with `LeafFamily::Softmax`, `LeafFamily::Survival`, and `LeafFamily::Heteroscedastic`.
   - In `retrain_weighted_named`, generalize target ingestion:
     - Softmax accepts discrete class integers $y_i \in \{0, \dots, C-1\}$.
     - Survival accepts duration and event pairs $(T_i, E_i)$.
     - Heteroscedastic accepts continuous floats.
   - Retain full compatibility with `mode="internal"` (mutates leaf parameters in place) and `mode="external"` (returns a new `ModelTree` with updated leaf models, leaving original topology untouched).
   - Retain compatibility with `leaf_features` / `retrain_cols` micro-feature selection: allows the leaf model (e.g. survival hazard or heteroscedastic dispersion) to fit on a tailored feature subset.

## Testing Decisions

### What Makes a Good Test
- Tests must strictly evaluate observable external behavior: parameter convergence, prediction dimensions, probability calibration ($\sum p_c = 1.0$), and survival monotonicity ($\hat{S}(t_2) \le \hat{S}(t_1)$ for $t_2 \ge t_1$).
- Tests must never assert private internal solver iteration counts or temporary scratch buffer states.

### Modules Tested
- Rust core unit tests covering leaf model fitting, parameter recovery on synthetic data, and numerical stability.
- Python integration tests exercising `ModelTree(leaf_family=..., max_depth=0)` on benchmark datasets:
  - Multiclass calibration on standard benchmarks (`iris` / synthetic 3-class).
  - Survival curves on `veteran.csv`.
  - Heteroscedastic uncertainty bounds on synthetic heteroscedastic data ($y = x + \epsilon \cdot |x|$).

### Prior Art
- Existing distributional tests in `tests/test_model_tree_raw.py` and `crates/mltreelib-core/tests/model_tree.rs`.
- Practical showcase script structure in `examples/distributional_leaf_benchmarks.py`.

## Out of Scope

- Recursive multi-depth tree splitting criteria (multiclass Gini split search, log-rank/martingale tree splitting, Gaussian NLL deviance splitting across branches). Splitting criteria will be built in the subsequent phase.
- Dynamic boosting ensembles across rounds (NGBoost-style multi-tree additive boosting).
- Left-truncation or interval-censored survival data (only standard right-censoring $(T, E)$ is in scope).
- Automated hyperparameter optimization (HPO) for multiclass or survival regularization parameters.

## Further Notes

- The martingale residual $M_i = E_i - \hat{\Lambda}_0(T_i)$ computed at the stump leaf provides the foundational mathematical building block for the subsequent Phase C splitting engine: once stored on rows, tree growth reduces to streaming least-squares split search on $M_i$ using existing `SplitScratch` infrastructure.
