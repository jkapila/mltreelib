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
| **10** | [`10-survival-and-competing-risks-trees.md`](10-survival-and-competing-risks-trees.md) | **Active** | `v0.4.7` (partial) | Log-rank splitting, martingale residual variance reduction, and competing risks. |
| **11** | [`11-gamlss-multiparameter-distributional-leaves.md`](11-gamlss-multiparameter-distributional-leaves.md) | **Planned** | — | GAMLSS multi-parameter leaves (location, scale, shape, and skewness). |
| **12** | [`12-exotic-statistical-and-metric-splitting-criteria.md`](12-exotic-statistical-and-metric-splitting-criteria.md) | **Completed** | `v0.4.9` (target) | MOB score tests, CTree permutation tests, MMD, Energy Distance, and Wasserstein splitting. |
| **13** | [`13-context-tree-weighting-and-suffix-trees.md`](13-context-tree-weighting-and-suffix-trees.md) | **Partially Landed** | `v0.4.4` | `ContextAnomalyTree` and `TrieKNN` in core; full CTW ensembling pending. |
| **14** | [`14-cubist-committees-and-model-tree-forests.md`](14-cubist-committees-and-model-tree-forests.md) | **Planned** | — | Cubist boosting committees, hierarchical model smoothing, and LinearForest. |
| **15** | [`15-structured-tree-ensembles.md`](15-structured-tree-ensembles.md) | **Planned** | — | Generalized Random Forests (GRF) and honest inference forests. |
| **16** | [`16-copula-leaves-and-generative-forests.md`](16-copula-leaves-and-generative-forests.md) | **Planned** | — | Vine copula leaves, generative forests, and tractable probabilistic circuits. |
| **17** | [`17-panel-var-vecm-trees.md`](17-panel-var-vecm-trees.md) | **Planned** | — | Panel fixed effects, TVAR, and TVECM longitudinal econometric trees. |
| **18** | [`18-documentation-examples-and-guides-modernization.md`](18-documentation-examples-and-guides-modernization.md) | **Active** | `v0.4.9` | Comprehensive documentation, 8-suite example consolidation, real data grounding, scikit-learn style plotting, and Mermaid architecture diagrams. |
