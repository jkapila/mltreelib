# Spec 20: Modular Control Architecture (GrowthControl & LeafControl), Leaf/Solver Decoupling, Native Serialization & Agent Skill

Status: ready-for-agent

## Problem Statement

As `mltreelib` has grown across 19 specifications—incorporating frontier survival models, exotic split metrics, causal trees, distributional families, and unsupervised algorithms—the user-facing interface faces five major structural friction points:

1. **Parameter Explosion and Control Conflation**: The root `ModelTree` constructor has ballooned into an unwieldy flat bag of over 70 parameters. Tree topology controls (`max_depth`, `min_rows`, `gain`, `col_sample`, `extratrees`, `honesty`) are tangled with leaf estimation details (`leaf_solver`, `l1`, `l2`, `tweedie_p`, `tobit_threshold`, `mars_degree`, `aft_dist`, `cure_fraction`, `spline_knots`, `knn_k`). Users cannot tell which parameters govern tree growth versus leaf estimation, or where custom knobs should be passed.
2. **Coupling of Leaf Types to Solvers**: Leaf modeling forms (constant, linear, polynomial, spline, hurdle) are conceptually orthogonal to the optimization solver used to fit them (IRLS, L-BFGS, Coordinate Descent, evolutionary algorithms like SHADE/CMA-ES). Currently, certain leaf types are implicitly coupled or assumed to use specific solvers, preventing users from pairing arbitrary leaf representations with specialized numerical optimizers.
3. **Second-Class Unsupervised Tree Surface**: While regression and classification are well-structured, `ClusterTree` and `AnomalyTree` lack dedicated first-class Python modules and documentation. They are buried within an internal models file without central class prominence, clear parameter catalogs, or dedicated API reference pages.
4. **Redundant Prediction Dispatchers**: Despite introducing the unified `predict(X, kind="...")` architecture in Spec 05, legacy auxiliary dispatchers (`predict_adjusted`, `predict_interval`, `predict_label`, `predict_proba`) remain exposed in the public interface, causing confusion over canonical usage.
5. **Absence of Model Persistence**: Fitted trees cannot be serialized to disk or byte streams. `ModelTree` objects cannot be pickled (`TypeError: cannot pickle 'builtins.ModelTree' object`) and provide no `save()` or `load()` methods, making production deployment, artifact caching, and serving impossible.
6. **Lack of an Agent Skill**: AI agents interacting with the repository have no unified, structured skill definition instructing them on how to select, parameterize, fit, evaluate, and diagnose `mltreelib` across its diverse task regimes (regression, classification, clustering, anomaly detection, survival, causal uplift, time series).

## Solution

Refactor and formalize the architecture into a clean, modular, and production-ready system:

1. **Modular Control Architecture (`ModelTree` with `GrowthControl` & `LeafControl`)**:
   - **`ModelTree` (Coordinator & Inference Engine)**: Coordinates fitting, routes inputs, exposes unified prediction dispatches, and handles model persistence.
   - **`GrowthControl` (Tree Growth & Topology)**: Encapsulates all spatial partitioning knobs:
     - Growth bounds: `max_depth`, `min_rows`, `minsplit`, `minbucket`, `max_leaf_nodes`.
     - Splitting criteria: `gain` (LogRank, Gray, Deviance, Pseudo-Value, RMST, AFT-Deviance, Chow, Gini, SSE), `split_precision`, `split_cols`.
     - Randomization & subspaces: `extratrees`, `col_sample`, `sporf_d`, `sporf_projections`, `seed`.
     - Causal & honesty: `honesty`, `honesty_fraction`.
     - Pruning & stopping: `prune`, `cp_alpha`, `purity_gate`, `purity_gate_metric`, `min_gain`.
     - Routing & constraints: `cat_search`, `cat_order`, `maxsurrogate`, `tree_monotone`.
   - **`LeafControl` (Leaf Modeling & Optimization)**: Encapsulates all terminal leaf behavior:
     - Leaf functional kind: `leaf` (Constant, Linear, Huber, Quantile, Mars, AFT).
     - Optimization solver: `solver` (Faer/IRLS, L-BFGS, CCD, SHADE, CMA-ES, GA, NES).
     - Distribution / Loss family: `family` (Gaussian, Binomial, Poisson, Tweedie, AFT, Cure, Spline).
     - Regularization: `l1`, `l2`, `tau`.
     - Family-specific knobs: `aft_dist`, `cure_fraction`, `spline_knots`, `tobit_threshold`, `tweedie_p`, `mars_degree`.
     - Operational leaf policies: `leaf_na_policy`, `leaf_digest`, `leaf_monotone`, `shrinkage`, `knn_k`.
2. **Polymorphic Parameter Ingestion (Objects, Dicts, or Flat Kwargs)**:
   - `ModelTree` accepts:
     - Typed control instances: `ModelTree(growth_control=GrowthControl(...), leaf_control=LeafControl(...))`.
     - Configuration dictionaries: `ModelTree(growth_control={...}, leaf_control={...})`.
     - Flat `**kwargs`: Automatically partitioned into `GrowthControl` and `LeafControl` with full backwards compatibility and helpful error messages for unknown keys.
3. **True Decoupling of Leaf Representation and Solver Engine**:
   - Explicit separation between `leaf_type` (what mathematical function lives in the leaf) and `leaf_solver` (how the parameters are optimized). Any compatible leaf representation can be optimized using any compatible solver.
4. **First-Class Unsupervised Packages (`mltreelib.cluster` and `mltreelib.anomaly`)**:
   - Elevate `ClusterTree` and `AnomalyTree` to primary, fully documented central classes located in dedicated submodules with standard `fit`/`predict` contracts, inspectable properties, and dedicated API reference documentation.
5. **Canonical Unified `predict(..., kind=...)` Interface**:
   - Standardize `predict(X, kind=...)` as the universal inference engine supporting `"response"`, `"proba"`, `"interval"`, `"adjusted"`, `"survival"`, `"hazard"`, `"cif"`, `"risk"`, and `"uplift"`. Deprecate legacy auxiliary methods into thin convenience aliases or retire them cleanly.
6. **High-Performance Model Serialization**:
   - Implement native zero-copy binary serialization (`to_bytes()` and `from_bytes()`) and disk persistence (`save(path)` and `ModelTree.load(path)`) using portable, version-tagged binary encoding (Bincode / JSON state).
   - Implement standard Python `__getstate__` and `__setstate__` protocols so all models are natively picklable via standard Python `pickle`.
7. **Comprehensive `mltreelib` Agent Skill**:
   - Create a specialized agent skill (`skills/mltreelib/SKILL.md`) that guides agents through task classification, parameter selection, modular control configuration, training, evaluation, rule extraction, and diagnostic plotting across all seven modeling pathways.

## User Stories

### Modular Control & Parameter Ergonomics
1. As a data scientist, I want `GrowthControl(max_depth=4, gain="rmst")` and `LeafControl(leaf="linear", l2=1.0)`, so that tree topology knobs and leaf optimization knobs are completely isolated.
2. As a pipeline developer, I want to pass dictionaries `ModelTree(growth_control={"max_depth": 5}, leaf_control={"l2": 0.5})`, so that I can construct trees directly from JSON/YAML configuration files.
3. As an existing user, I want `ModelTree(max_depth=4, leaf="linear", l2=1.0)` with flat kwargs to work automatically, so that my existing code does not break.
4. As an actuary, I want to configure `LeafControl(family="tweedie", tweedie_p=1.65, l2=1.0)` and pass it into `ModelTree`, so that my leaf distribution settings are cleanly isolated from tree growth constraints.
5. As an oncologist, I want to configure survival leaf knobs like `aft_dist="weibull"`, `cure_fraction=0.2`, and `spline_knots=4` exclusively within `LeafControl`, so that survival leaf parameters do not pollute the root tree constructor.
6. As an econometrics practitioner, I want `tobit_threshold` and `mars_degree` to live inside `LeafControl`, so that censored and adaptive regression settings are grouped with the estimators that use them.
7. As a novice user, I want mistyped kwargs to suggest the correct name (e.g. `max_dpeth` $\to$ `max_depth`), so that parameter mistakes are caught immediately with actionable guidance.
8. As an API user, I want the parameter catalog documentation to clearly tag every parameter with its scope (`[Growth Control]`, `[Leaf Control]`, or `[Fit Data]`), so that I instantly know where and how to supply each setting.

### Leaf Type & Solver Decoupling
9. As a quantitative researcher, I want to pair a linear leaf model with an evolutionary solver (`LeafControl(leaf="linear", solver="shade")`), so that I can optimize leaf coefficients on non-convex or discontinuous objective functions.
10. As a biostatistician, I want to pair an AFT leaf with L-BFGS or IRLS (`LeafControl(leaf="aft", solver="lbfgs")`), so that parametric survival curves converge rapidly via analytic gradients.
11. As a machine learning engineer, I want to pair a Huber robust leaf with Coordinate Descent (`LeafControl(leaf="huber", solver="ccd")`), so that I achieve exact sparse coefficients resistant to heavy-tailed outliers.
12. As a developer, I want clear validation errors when an incompatible leaf and solver combination is requested (e.g., closed-form Cholesky on a non-linear neural leaf), so that invalid configurations fail fast with actionable guidance.

### Centralized Unsupervised Class Identity
13. As an unsupervised learning practitioner, I want `from mltreelib.cluster import ClusterTree`, so that clustering trees have an obvious, first-class import location.
14. As a security analyst, I want `from mltreelib.anomaly import AnomalyTree`, so that anomaly detection models are organized logically alongside regression and classification models.
15. As an engineer using `ClusterTree`, I want full inspectability via `tree.labels_`, `tree.n_clusters_`, and `tree.rules()`, so that discovered segments can be audited and visualized.
16. As an engineer using `AnomalyTree`, I want calibrated binary predictions via `tree.predict(X)` and raw anomaly scores via `tree.score_samples(X)` or `predict(X, kind="score")`, so that triage thresholds can be tuned against operational review budgets.
17. As a documentation reader, I want dedicated API reference pages for `ClusterTree` and `AnomalyTree`, so that all hyperparameter options and usage examples are thoroughly documented.

### Prediction API Consolidation
18. As an application developer, I want `tree.predict(X, kind="proba")` to be the single canonical way to obtain class probabilities, so that I don't have to guess between `predict_proba` and `predict`.
19. As a financial analyst, I want `tree.predict(X, kind="interval", alpha=0.05)` to return prediction intervals, eliminating the need for a separate `predict_interval` method.
20. As a legacy user, I want calling `tree.predict_proba(X)` to redirect to `tree.predict(X, kind="proba")` with a gentle deprecation notice, so that older client code remains functional.
21. As an API caller, I want `tree.predict(X, kind="...")` to throw an informative `ValueError` listing all valid `kind` options for the fitted model when an invalid kind is supplied.

### Model Persistence & Serialization
22. As an MLOps engineer, I want `tree.save("model.mltree")` and `ModelTree.load("model.mltree")`, so that I can persist trained models to disk and reload them in a production scoring service.
23. As a pipeline author, I want `pickle.dump(tree, f)` and `pickle.load(f)` to work seamlessly, so that `mltreelib` models integrate natively with Python multiprocessing, joblib, and standard caching frameworks.
24. As an edge device developer, I want `tree.to_bytes()` and `ModelTree.from_bytes(data)`, so that model states can be transferred across network sockets or stored in binary databases without touching the filesystem.
25. As a systems engineer, I want serialized model binaries to include a schema version header, so that future engine upgrades detect and gracefully migrate or reject incompatible serialized artifacts.
26. As a quality engineer, I want deserialized models to produce bitwise identical predictions and rule sets compared to the pre-serialization model instance across all prediction kinds.

### Agent Skill
27. As an AI assistant, I want a dedicated `mltreelib` skill file, so that I can automatically select the optimal tree architecture, splitting criteria, and leaf family when users ask tabular modeling questions.
28. As an agent, I want structured recipe templates for regression, classification, survival, causal uplift, time series, clustering, and anomaly detection, so that I consistently generate correct, idiomatic `mltreelib` code.
29. As an agent, I want clear rules on Two-Control configuration, so that I never write bloated `ModelTree` constructor calls that mix topology and leaf parameters.
30. As an agent, I want guidelines for model persistence and evaluation plotting, so that I can deliver end-to-end production scripts to the user.

## Implementation Decisions

### Decision 1: Control Architecture & Ingestion Routing
- The system formalizes two primary control dataclasses:
  - `GrowthControl`: Holds all tree growth, splitting criteria, node stopping rules, surrogate splits, feature subsampling, honesty, and pruning.
  - `LeafControl`: Holds leaf estimator family, optimization solver, penalties, and family-specific hyperparameters.
- `ModelTree` acts as the root coordinator. Its `__init__` signature accepts:
  - `growth_control: GrowthControl | dict | None = None`
  - `leaf_control: LeafControl | dict | None = None`
  - `**kwargs`: Any loose kwargs are partitioned based on internal registries of `GROWTH_PARAM_NAMES` and `LEAF_PARAM_NAMES`.
- If an unrecognized kwarg is supplied, the constructor raises `ValueError` with `get_close_matches` suggestions.

### Decision 2: Orthogonal Leaf Representation and Solver Interfaces
- The leaf optimization engine will separate the leaf parameter representation from the optimization solver.
- Leaf representations define the model equation and parameter vector (e.g. Constant, Linear, Polynomial, AFT, Hurdle).
- Leaf solvers define the numerical optimization algorithm (e.g. Analytic Cholesky/Faer, Iteratively Reweighted Least Squares, L-BFGS Quasi-Newton, Coordinate Descent, or Evolutionary Strategy).
- The engine will validate compatibility upon initialization. Solvers requiring smooth gradients will verify that the chosen leaf family provides analytic or numerical score functions; derivative-free solvers will operate on any evaluable loss function.

### Decision 3: First-Class Python Namespace Reorganization
- Unsupervised models will be organized into dedicated public modules:
  - `mltreelib.cluster`: Exports `ClusterTree` and clustering utility functions.
  - `mltreelib.anomaly`: Exports `AnomalyTree` and score calibration utilities.
- Both classes will remain re-exported at the root `mltreelib` namespace (`mltreelib.ClusterTree`, `mltreelib.AnomalyTree`) to prevent breaking existing imports.
- `ClusterTree` and `AnomalyTree` will be refactored into first-class classes with explicit, typed constructor arguments, eliminating opaque `**params` dictionaries for core parameters.

### Decision 4: Unified Prediction Dispatcher and Deprecation Policy
- The canonical inference entrypoint across all models is `predict(X, kind=...)`.
- Supported values for `kind` will be dynamically validated based on the fitted leaf family and target type:
  - Regression/Classification: `"response"`, `"proba"`, `"log_odds"`, `"interval"`, `"adjusted"`.
  - Survival / Competing Risks: `"response"`, `"survival"`, `"hazard"`, `"cif"`, `"risk"`.
  - Causal / Uplift: `"response"`, `"uplift"`, `"cate"`.
  - Anomaly: `"binary"`, `"score"`, `"path_length"`.
  - Clustering: `"cluster"`, `"distances"`.
- Legacy methods (`predict_proba`, `predict_interval`, `predict_adjusted`, `predict_label`, `predict_uplift`) will be marked as legacy convenience wrappers that delegate directly to `predict(..., kind=...)`.

### Decision 5: Serialization Architecture & Binary Format
- Serialization will operate via Rust `serde` and `bincode` (with an optional JSON export mode).
- The binary format will include:
  - A magic byte header (`MLTREE\x01`).
  - An engine version tag.
  - Serialized tree topology, split conditions, categorical index mappings, and leaf parameters.
- In Python, `ModelTree`, `ClusterTree`, and `AnomalyTree` will implement:
  - `save(filepath: str | Path)`
  - `@classmethod load(filepath: str | Path) -> Model`
  - `to_bytes() -> bytes`
  - `@classmethod from_bytes(data: bytes) -> Model`
  - `__getstate__()` and `__setstate__()` using `to_bytes()` and `from_bytes()` to enable standard Python `pickle.dump()` and `pickle.load()`.

### Decision 6: Dedicated Agent Skill Architecture
- A comprehensive skill definition will be authored at `skills/mltreelib/SKILL.md`.
- The skill will document:
  - Seven canonical task pathways: Regression, Classification, Clustering, Anomaly Detection, Survival / Competing Risks, Uplift / Causal, and Time Series.
  - The Modular Control configuration pattern (`ModelTree` + `GrowthControl` + `LeafControl`).
  - When to choose each split criterion and leaf solver.
  - Complete code recipes with data ingestion, model fitting, unified prediction, evaluation metrics, and diagnostic plotting.

## Testing Decisions

- **Polymorphic Ingestion Testing**:
  Verify that:
  1. `ModelTree(growth_control=GrowthControl(...), leaf_control=LeafControl(...))`
  2. `ModelTree(growth_control={...}, leaf_control={...})`
  3. `ModelTree(**kwargs)`
  all produce bitwise identical trees, partitions, and predictions.
- **Leaf / Solver Orthogonality Testing**:
  Verify that linear leaves can be successfully fitted with both gradient solvers (L-BFGS, IRLS) and evolutionary solvers (SHADE, CMA-ES), comparing convergence accuracy and execution stability.
- **Serialization Roundtrip Testing**:
  For every model type (continuous regression, logistic classification, multiclass, survival, competing risks, clustering, anomaly), execute:
  1. Train model $M_1$ on training data.
  2. Serialize $M_1$ to bytes and file.
  3. Deserialize into model $M_2$.
  4. Assert that $M_1.\text{predict}(X_{\text{test}}, \text{kind}=k) == M_2.\text{predict}(X_{\text{test}}, \text{kind}=k)$ down to machine precision.
  5. Assert that $M_1.\text{extract\_rules}() == M_2.\text{extract\_rules}()$.
- **Pickle Compatibility Testing**:
  Verify that standard `pickle.dumps(model)` and `pickle.loads(data)` execute without error and preserve model state across all target families.
- **Unsupervised API Testing**:
  Test `ClusterTree` and `AnomalyTree` with explicit constructor parameters, verifying `fit()`, `predict()`, `labels_`, `n_clusters_`, and `rules()` output.
- **Deprecation and Warning Verification**:
  Assert that passing legacy flat leaf parameters emits appropriate `FutureWarning` notices without halting execution.

## Out of Scope

- Deep neural network tabular foundation models or GPU-accelerated backends.
- Automated hyperparameter optimization (AutoML search loops); `mltreelib` provides deterministic modeling primitives.
- Modifying underlying core mathematical splitting formulas (e.g. LogRank, Gray's test, Chow test math remains unchanged).
- Distributed cluster-scale training (e.g. Apache Spark or Ray clusters); `mltreelib` focuses on single-node multi-core parallelism via `rayon`.

## Further Notes

- This specification serves as the foundational architectural cleanup before landing remaining frontier specs (Spec 11 GAMLSS, Spec 14 Cubist, Spec 15 GRF, Spec 16 Copulas, Spec 17 TVAR).
- By decoupling `GrowthControl` and `LeafControl` and implementing native serialization now, all future leaf families (Negative Binomial, Beta, Student-t) and forest ensembles (LinearForest, GRF) will inherit clean parameter boundaries and serialization out of the box.
