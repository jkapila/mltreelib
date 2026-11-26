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
| **14** | [`14-cubist-committees-and-model-tree-forests.md`](14-cubist-committees-and-model-tree-forests.md) | **Superseded** | — | Cubist boosting committees, hierarchical model smoothing, and LinearForest. **Superseded by Spec 28**, which rebuilds the committee with corrected mechanics: two named smoothing modes instead of the non-existent `m0` constant, and an unweighted member average instead of the boost framing. |
| **15** | [`15-structured-tree-ensembles.md`](15-structured-tree-ensembles.md) | **Superseded** | — | Generalized Random Forests (GRF) and honest inference forests. **Superseded by Specs 28 and 31**, which split the family into bagging, boosting, meta, and Bayesian facades. GRF itself still needs its own specification, because its splitting rule needs a node-level relabeling strategy that `GainFn::Custom` cannot express. |
| **16** | [`16-copula-leaves-and-generative-forests.md`](16-copula-leaves-and-generative-forests.md) | **Planned** | — | Vine copula leaves, generative forests, and tractable probabilistic circuits. |
| **17** | [`17-panel-var-vecm-trees.md`](17-panel-var-vecm-trees.md) | **Planned** | — | Panel fixed effects, TVAR, and TVECM longitudinal econometric trees. |
| **18** | [`18-documentation-examples-and-guides-modernization.md`](18-documentation-examples-and-guides-modernization.md) | **Completed** | `v0.4.9` | Comprehensive documentation, 9-suite example consolidation, real data grounding, scikit-learn style plotting, and Mermaid architecture diagrams. |
| **19** | [`19-frontier-survival-trees-aft-cure-splines-calibration.md`](19-frontier-survival-trees-aft-cure-splines-calibration.md) | **Completed** | `v0.4.10` | Parametric AFT (Weibull/LogNormal/LogLogistic), Mixture Cure, Spline leaves, Pseudo-Value/RMST/AFT-Deviance splits, D-Calibration, and on-demand dataset downloader. |
| **20** | [`20-architecture-two-fold-control-decoupling-serialization-and-agent-skill.md`](20-architecture-two-fold-control-decoupling-serialization-and-agent-skill.md) | **Main scope implemented** | `v0.4.11` | Growth and leaf controls, solver selection, first-class cluster/anomaly modules, JSON snapshots for ModelTree, PCT and EIF, and the `mltreelib` agent skill. Other unsupervised engines remain open. |
| **21** | [`21-standalone-solvers-and-benchmarks.md`](21-standalone-solvers-and-benchmarks.md) | **Completed** | `v0.4.14` | General bounded optimizer, production CMA-ES with ask/tell and IPOP restart, standalone GLM fitting, batch AskTell (scalar & multi-objective), and realistic benchmark program. |
| **22** | [`22-expanded-clustering-distance-evaluation-and-research-algorithms.md`](22-expanded-clustering-distance-evaluation-and-research-algorithms.md) | **Proposed** | — | Shared validated distance registry, scalable density clustering and DBSCAN/OPTICS, BIRCH streaming, Gower and convex clustering, clustering evaluation, and e-means/o-means/K-NN Tree research prototypes. |
| **23** | [`23-expanded-classical-leaf-family-catalog.md`](23-expanded-classical-leaf-family-catalog.md) | **Proposed** | — | Expanded classical leaf catalog: direct Rust/Python estimators, ordinal, discriminant, Bayesian, GAM, mixture, distributional, kernel, GP, and structured families with capability-gated promotion. |
| **24** | [`24-association-rules-and-context-tree-integration.md`](24-association-rules-and-context-tree-integration.md) | **Proposed** | — | Separate Rust association-rules capability with FP-growth, Eclat, closed itemsets, associative classifiers, tree-path export, and contextual rules from Context Tree counts. |
| **25** | [`25-unified-model-surface-components-and-diagnostics.md`](25-unified-model-surface-components-and-diagnostics.md) | **Completed** | `v0.4.13` | Unified task facades, package-native lifecycle, common Objective/Learner/Controller/Diagnostics reports, consolidated PCA/SVD/ICA/NMF capability, extended solver catalog. |
| **26** | [`26-generic-global-and-constrained-optimizers.md`](26-generic-global-and-constrained-optimizers.md) | **Completed** | `v0.4.13` | Leaf solver decoupling with a generic bounded optimization seam and constraint semantics, then Random Search/LHS, L-BFGS-B, BFGS, nonlinear CG, Nelder-Mead, Powell, bounded simulated annealing, COBYLA, and basin hopping in dependency order, plus canonical benchmark program. |
| **27** | [`27-regression-model-expansion.md`](27-regression-model-expansion.md) | **Proposed** | — | Group-aware regression layer: coefficient groups, panel entity keys, and hierarchical levels on the design view; binned split evaluation for the regression criteria as the shared performance prerequisite for every ensemble family; then multivariate quantile, Huber with robust scale, expectile, NNLS, group and sparse-group lasso, negative binomial and beta GLMs, panel within-transformation and fixed-effects leaves, partial pooling, and conditional GAM work. |
| **28** | [`28-model-forest-bagging-family.md`](28-model-forest-bagging-family.md) | **Proposed** | — | `ModelForest`, the bagging family: plain bagging, random forest, ExtraTrees, the Cubist-style committee, and the linear-leaf forest. `ModelTree` members, trait-gated capabilities, exact out-of-bag machinery, common report. Family sibling of Specs 29 and 30. |
| **29** | [`29-model-boost-boosting-family.md`](29-model-boost-boosting-family.md) | **Proposed** | — | `ModelBoost`, the boosting family: Friedman's gradient boosting over the existing tree engine with second-order stages, ensemble-level monotone bounds, DART dropout boosting, linear and forest stages, and a measured bar against `perpetual (Rust)`. Family sibling of Specs 28, 30, and 32. |
| **30** | [`30-model-meta-estimators.md`](30-model-meta-estimators.md) | **Proposed** | — | `ModelMeta`, the meta-estimator family: voting, averaging, and stacking over heterogeneous named members, with out-of-fold meta-features, a shared cross-validation generator, and leakage-aware defaults. Family sibling of Specs 28 and 29. |
| **31** | [`31-ensemble-family-selection-guide.md`](31-ensemble-family-selection-guide.md) | **Proposed** | — | Cross-cutting decision framework for choosing between forest, boost, meta, and Bayesian estimators: hard constraints, data regime, cost model, and a documented default, grounded in the local 56-dataset Elo arena. Governs Specs 28, 29, 30, and 32. |
| **32** | [`32-bayesian-tree-ensembles.md`](32-bayesian-tree-ensembles.md) | **Proposed** | — | `ModelBayes`, the Bayesian family: BART's tree, leaf, and error-variance priors adopted as growth-criterion configuration first, then a Metropolis-Hastings posterior sampler and credible-interval products under the experimental namespace. Family sibling of Specs 28, 29, and 30. |
| **33** | [`33-api-consistency-defaults-and-docs-contract.md`](33-api-consistency-defaults-and-docs-contract.md) | **Partly landed** | — | One source of truth for `ModelTree` defaults, split/grow/prune/leaf knob model, fixes for documented-but-rejected options, scale-free pruning, and data-driven defaults confirmed on held-out datasets. |
| **34** | [`34-survival-and-uplift-real-data-suite.md`](34-survival-and-uplift-real-data-suite.md) | **Partly landed** | — | Curated real survival, competing-risks and uplift datasets fetched on demand with pinned checksums and licence notes; one normalised schema; reproducible results pages with uncertainty. |
| **35** | [`35-simple-object-interfaces-and-family-summaries.md`](35-simple-object-interfaces-and-family-summaries.md) | **Partly landed** (A, B and the input-boundary fixes shipped; C and D open) | — | Class-first `Dataset` and `MetricsReport`; models accept a dataset; one summary shape per family (GLM, Tobit, quantile, Huber, MARS, k-NN, survival, cluster, anomaly); causal linear leaf. |

---

## Build Priority

Ordered by what unblocks the most work and what the local benchmark says is worth building. Phases are the count in each specification's build sequence.

### Tier 0 — Unblocks everything else

| Order | Spec | Phases | Why first |
|---:|---|---:|---|
| 1 | **27** Phase 1 | 1 | Binned split evaluation for the regression criteria. Every ensemble family pays this per member, per stage, or per member per fold. Without it a 200-member forest costs over 13 minutes per fit |
| 2 | **27** Phase 0 | 1 | Group metadata on the design view. Group lasso, panel grouping, and hierarchical pooling all depend on it |

### Tier 1 — The measured win

| Order | Spec | Phases | Why |
|---:|---|---:|---|
| 3 | **28** Phases 0 to 2 | 3 | `ModelForest` skeleton, bagging, and the Cubist committee. The local board shows `ModelTree(all)` at 715.9 regression Elo against ranger's 1700.6. The tree is under-ensembled on regression, and bagging is the direct fix. This is the largest measured improvement available |
| 4 | **25** Decision 11 | **Completed** | Remove the remaining sklearn `get_params` from `python/mltreelib/_controls.py` and `crates/mltreelib-py/src/lib.rs`. Replaced with `.config()` |

### Tier 2 — The remaining families

| Order | Spec | Phases | Why |
|---:|---|---:|---|
| 5 | **29** | 7 | `ModelBoost`. Must beat `perpetual (Rust)` at 1400.1 overall on at least one of distributional leaves, ensemble-level monotone bounds, categorical handling, or missing-value routing. A plain GBDT is not worth building |
| 6 | **30** | 5 | `ModelMeta`. Voting and averaging are cheap; stacking needs the fold generator from Spec 28 Phase 0 |
| 7 | **27** Phases 2 to 7 | 6 | Quantile, Huber, expectile, NNLS, group lasso, and the count and bounded-response GLMs. Each adds a regression family the forest's leaves inherit |
| 8 | **28** Phases 5 to 7 | 3 | Wider bagging family, later coverage, and the Elo benchmark |

### Tier 3 — Cross-cutting and deferred

| Order | Spec | Phases | Why |
|---:|---|---:|---|
| 9 | **31** | 3 | The selection guide. Documentation and the agent skill entry, then the local benchmark that replaces cited tendencies with this repo's own numbers |
| 10 | **27** Phases 8 to 11 | 4 | Panel entity grouping, fixed-effects leaves, partial pooling, and conditional GAM work |
| 11 | **32** | 4 | `ModelBayes`. Priors as configuration first, then the sampler. The local board puts `bart` seventh at 1313.9 with a 25.49 s fit time, so the accuracy case is modest and the uncertainty case is the argument |
| 12 | **23** | — | The leaf catalog. Ordinal, discriminant, Bayesian, mixture, GAM, and GP leaves. Shares utilities with Spec 27, so sequence it after Tier 2 |
| 14 | **22** | — | Clustering. `cluster/` holds cltree, hdbscan_tree, pct, and urf, but no distance registry, DBSCAN, OPTICS, BIRCH, Gower, or convex clustering |
| 15 | **24** | — | Association rules. `rules.rs` is C5.0 rule extraction only; no FP-growth, Eclat, closed itemsets, or associative classifiers |
| 16 | **11** | — | GAMLSS multi-parameter leaves. No `gamlss.rs` exists |
| 17 | **16** | — | Copula leaves and generative forests. No copula code exists |
| 18 | **13** | — | Context tree weighting. `ContextAnomalyTree` and `TrieKNN` landed at v0.4.4; full CTW ensembling is still open |
| 19 | **GRF** | — | Not yet specified. Needs a node-level relabeling strategy, which `GainFn::Custom` cannot express |

### Dependency notes

- Spec 27 Phase 1 precedes Spec 28 Phase 0. The forest's fit-time expectations depend on it.
- Spec 28 Phase 0 delivers the cross-validation generator that Spec 30 Phase 2 needs.
- Spec 27 Phases 2 to 7 deliver the leaf models that Spec 28's `linear_forest` and `committee` kinds, and Spec 29's stages, build on.
- Spec 31 Phase 2 benchmarks the families from Specs 28, 29, 30, and 32, so it lands after them.
- Spec 29 needs the monotone and missing-value work from Spec 27's leaf models to clear the `perpetual (Rust)` bar.


---

## Looking Forward

The specs above describe a library that is strong on **core** (trees, leaves, ensembles). The long-term shape is three areas, with one consistent surface across all of them:

- **Data**: loading, splitting, profiling, validation, feature tools. Missing values are never imputed or recoded by data helpers.
- **Core**: trees first (split, grow, prune, leaf), then ensembles.
- **Metrics**: evaluation, calibration, fairness, and drift monitoring, so that a business question like "why did this model drift or fail?" has an answer.

Guiding rules: no scikit-learn dependency, a shrinking dependency set (build inside when possible), and early-stage honesty: APIs may change, so new features start as an issue.

### Gaps against that shape (from a read of specs 01 to 34)

| Capability | Status | Notes |
| :--- | :--- | :--- |
| Drift monitoring (PSI, CSI, KS, data and prediction drift, decay, retrain triggers) | **Not specified** | Primitives exist: a stability index (PSI/CSI) and score-KS separation in the metrics layer. No spec owns a drift API. |
| Data area: profiling, validation, splits, pipelines | **Not specified** (splits and helpers are in Spec 34) | `summarize`, `winnow`, `DatasetBuilder`, encoders and time features exist without an owning spec. |
| Save, load, versioning | **Partly** (Spec 20) | Versioned JSON, bytes, file and pickle exist. Only the tree has a runnable docs example. Model cards and a version registry are not specified. |
| Statsmodels-style summaries | **Partly** (Specs 24, 25) | Reports and per-leaf tables exist. No spec for a unified summary across tasks. |
| Calibration reports | **Partly** (Spec 19, 25, 30) | D-calibration and one-calibration exist. No cross-task calibration report. |
| Fairness reports | **Partly** (Spec 12) | A fairness criterion exists. No cross-model report. |
| Model comparison API | **Partly** (Specs 01, 31) | The arena compares models. No user-facing compare call. |
| Missing data in unsupervised models | **Open gap** (Spec 22) | Some clustering engines still coerce `None` to 0.0, which conflicts with the first-class missing-data rule. |
| CLI, interactive reports, other-language bindings | **Not specified** | Plots are static; R support is a source package only. |
| Table input at the model boundary (pandas, Arrow, Polars-style) | **In Spec 33** | A pandas frame currently fails in `fit`. |

### Housekeeping noted during the audit

- Status headers of Specs 25 and 26 say proposed while this index says completed.
- Spec 03 lists probit, tobit and tweedie as remaining, but the code has them.
- Spec 20's header says some unsupervised snapshots remain open, but they are implemented.
- Specs 21 and 26 both describe the generic bounded optimizer seam; 11 and 27 overlap on overdispersed and bounded outcomes; 20 and 25 both specify snapshot round-trip. Add cross-references.
- Spec 34 is partly landed (on-demand datasets with checksums exist), but its header says proposed.


### Candidates for the next release (added 2026-11-26)

Beyond issues 24 to 26 and `Dataset` on the cluster, anomaly and component models. Ranked by reliability payoff, then capability.

1. **Missing data in unsupervised models** (gap under Spec 22): some clustering engines coerce `None` to 0.0. Check every engine and route natively or raise.
2. **Save, load and a model card for every model** (Spec 20 follow-up): round-trip tests per family, a runnable guide, a small card (schema, defaults, version).
3. **Calibration, fairness and model-compare as `MetricsReport` tasks** (Specs 12, 19, 25).
4. **Drift monitoring** (no spec yet): a `drift` task in `MetricsReport` on the existing PSI/CSI and KS primitives.
5. **Spec 31 selection guide, phase 1** (docs only).
6. **Spec header housekeeping** (25, 26, 03, 20, 27, 34 status lines).
7. **Keyword catch-all audit**: `**params` accepts unknown keywords (`AnomalyModel(method="pca")` is silently accepted). Unknown keywords should raise.
8. For the release after: **Spec 28 `ModelForest`**, the largest measured win, after Spec 27 phase 1.

### Research ideas (not scheduled)

- **A tree as a leaf model.** Each leaf of a tree fits its own tree on the rows that reach it, so a region is treated as a universe of its own: its own features, depth, criterion and leaf family. In principle the result is one deeper tree, so the value must come from what a single tree cannot do: different complexity per region, different feature sets or leaf families per region, local missing-value routing, local regularisation. Risks: overfitting and explanation length. Related work: M5 smoothing, hierarchical mixtures of experts, cluster-then-model, Cubist rule committees. Approach if pursued: a leaf family `tree` that recurses with its own controls, a summary that nests the sub-tree, and a benchmark against a plain deeper tree at equal leaf count.
- **Data-representation analysis tools** (how the data is structured and shaped) are parked until the existing surface is complete.

**Principle: completeness and reliability of what the package already covers come before new capability.**
