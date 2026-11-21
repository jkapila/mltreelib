# Specifications & Roadmap Tracking — mltreelib

This directory tracks the architectural specifications, research blueprints, and implementation contracts for `mltreelib`.

---

## Specification Status Index

| Spec | Slug / Title | Status | Shipped In | Summary |
| :--- | :--- | :---: | :---: | :--- |
| **01** | [`01-elo-field-completion.md`](01-elo-field-completion.md) | **Completed** | `v0.4.0` | Elo field benchmark, secondary runners, and 3-tier wall-clock budgets. |
| **02** | [`02-model-tree-knobs.md`](02-model-tree-knobs.md) | **Completed** | `v0.4.1` | Honesty estimation, surrogate splits, cost-complexity pruning, and Chow gate. |
| **03** | [`03-leaf-family.md`](03-leaf-family.md) | **Completed** | `v0.4.2` | Distributional leaves: binomial-logit, probit, poisson, and tweedie. |
| **04** | [`04-split-optimization-simplification.md`](04-split-optimization-simplification.md) | **Completed** | `v0.4.3` | Decoupled split search (scalar variance reduction) and terminal OLS leaves. |
| **05** | [`05-predict-api-and-docs.md`](05-predict-api-and-docs.md) | **Completed** | `v0.4.3` | Unified `predict(..., kind=...)`, t-digest quantiles, and API docs. |
| **06** | [`06-cat-split-repro-and-fix.md`](06-cat-split-repro-and-fix.md) | **Completed** | `v0.4.3` | Categorical split partition fix and `ta_diamonds` repro harness. |
| **07** | [`07-tree-growth-simplification-and-allocation-elimination.md`](07-tree-growth-simplification-and-allocation-elimination.md) | **Completed** | `v0.4.4` | `SplitScratch` buffers, categorical scalar moments, and flat queue tree growth. |
| **08** | [`08-softmax-survival-heteroscedastic-stump-leaves.md`](08-softmax-survival-heteroscedastic-stump-leaves.md) | **Completed** | `v0.4.6` / `v0.4.8` | Softmax multiclass leaves, Nelson-Aalen survival leaves, and Gaussian NLL heteroscedastic leaves. |
| **09** | [`09-causal-and-uplift-trees.md`](09-causal-and-uplift-trees.md) | **Active** | `v0.4.7` (partial) | Causal trees, uplift modeling, and honest treatment effect estimation. |
| **10** | [`10-survival-and-competing-risks-trees.md`](10-survival-and-competing-risks-trees.md) | **Completed** | `v0.4.9` | Log-rank, Tarone-Ware, Peto, Martingale, Gray's test, Aalen-Johansen CIF, and Simplex Invariant. |
| **11** | [`11-gamlss-multiparameter-distributional-leaves.md`](11-gamlss-multiparameter-distributional-leaves.md) | **Planned** | — | GAMLSS multi-parameter leaves (location, scale, shape, and skewness). |
| **12** | [`12-exotic-statistical-and-metric-splitting-criteria.md`](12-exotic-statistical-and-metric-splitting-criteria.md) | **Completed** | `v0.4.9` | MOB score tests, CTree permutation tests, Equalized Odds, ExtraTrees, EBM FAST, MMD, Energy, and Wasserstein. |
| **13** | [`13-context-tree-weighting-and-suffix-trees.md`](13-context-tree-weighting-and-suffix-trees.md) | **Partially Landed** | `v0.4.4` | `ContextAnomalyTree` and `TrieKNN` in core; full CTW ensembling pending. |
| **14** | [`14-cubist-committees-and-model-tree-forests.md`](14-cubist-committees-and-model-tree-forests.md) | **Planned** | — | Cubist boosting committees, hierarchical model smoothing, and LinearForest. |
| **15** | [`15-structured-tree-ensembles.md`](15-structured-tree-ensembles.md) | **Planned** | — | Generalized Random Forests (GRF) and honest inference forests. |
| **16** | [`16-copula-leaves-and-generative-forests.md`](16-copula-leaves-and-generative-forests.md) | **Planned** | — | Vine copula leaves, generative forests, and tractable probabilistic circuits. |
| **17** | [`17-panel-var-vecm-trees.md`](17-panel-var-vecm-trees.md) | **Planned** | — | Panel fixed effects, TVAR, and TVECM longitudinal econometric trees. |
| **18** | [`18-documentation-examples-and-guides-modernization.md`](18-documentation-examples-and-guides-modernization.md) | **Completed** | `v0.4.9` | Comprehensive documentation, 9-suite example consolidation, real data grounding, scikit-learn style plotting, and Mermaid architecture diagrams. |
| **19** | [`19-frontier-survival-trees-aft-cure-splines-calibration.md`](19-frontier-survival-trees-aft-cure-splines-calibration.md) | **Completed** | `v0.4.10` | Parametric AFT (Weibull/LogNormal/LogLogistic), Mixture Cure, Spline leaves, Pseudo-Value/RMST/AFT-Deviance splits, D-Calibration, and on-demand dataset downloader. |
| **20** | [`20-architecture-two-fold-control-decoupling-serialization-and-agent-skill.md`](20-architecture-two-fold-control-decoupling-serialization-and-agent-skill.md) | **Main scope implemented** | `v0.4.11` | Growth and leaf controls, solver selection, first-class cluster/anomaly modules, JSON snapshots for ModelTree, PCT and EIF, and the `mltreelib` agent skill. Other unsupervised engines remain open. |
| **21** | [`21-standalone-solvers-and-benchmarks.md`](21-standalone-solvers-and-benchmarks.md) | **Ready for agent** | — | General bounded optimizer, production CMA-ES with ask/tell and IPOP restart, standalone GLM fitting with shared dispatch, and realistic benchmark program. |
| **22** | [`22-expanded-clustering-distance-evaluation-and-research-algorithms.md`](22-expanded-clustering-distance-evaluation-and-research-algorithms.md) | **Proposed** | — | Shared validated distance registry, scalable density clustering and DBSCAN/OPTICS, BIRCH streaming, Gower and convex clustering, clustering evaluation, and e-means/o-means/K-NN Tree research prototypes. |
| **23** | [`23-expanded-classical-leaf-family-catalog.md`](23-expanded-classical-leaf-family-catalog.md) | **Proposed** | — | Expanded classical leaf catalog: direct Rust/Python estimators, ordinal, discriminant, Bayesian, GAM, mixture, distributional, kernel, GP, and structured families with capability-gated promotion. |
| **24** | [`24-association-rules-and-context-tree-integration.md`](24-association-rules-and-context-tree-integration.md) | **Proposed** | — | Separate Rust association-rules capability with FP-growth, Eclat, closed itemsets, associative classifiers, tree-path export, and contextual rules from Context Tree counts. |
| **25** | [`25-unified-model-surface-components-and-diagnostics.md`](25-unified-model-surface-components-and-diagnostics.md) | **Proposed** | — | Unified task facades, package-native lifecycle, common Objective/Learner/Controller/Diagnostics reports, consolidated PCA/SVD/ICA/NMF capability, extended solver catalog, and removal of sklearn estimator compatibility. |
| **26** | [`26-generic-global-and-constrained-optimizers.md`](26-generic-global-and-constrained-optimizers.md) | **Proposed** | — | Leaf solver decoupling with a generic bounded optimization seam and constraint semantics, then Random Search/LHS, L-BFGS-B, BFGS, nonlinear CG, Nelder-Mead, Powell, bounded simulated annealing, COBYLA, and basin hopping in dependency order, plus canonical benchmark program. |
| **27** | [`27-regression-model-expansion.md`](27-regression-model-expansion.md) | **Proposed** | — | Group-aware regression layer: coefficient groups, panel entity keys, and hierarchical levels on the design view, then multivariate quantile, Huber with robust scale, expectile, NNLS, group and sparse-group lasso, negative binomial and beta GLMs, panel within-transformation and fixed-effects leaves, partial pooling, and conditional GAM work. |
