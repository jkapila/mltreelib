# Practical modeling documentation audit and specification

**Date:** 2026-11-18  
**Scope:** Repository documentation, navigation, examples, Python exports and selected implementation paths. This is an audit and proposal. It does not authorize claims that a capability is production-ready or equivalent to another package.

## Goal and method

The intended reader builds and evaluates models for practical decisions. The package aims to support a complete tree modeling workflow, with particular attention to missing values, categorical features and usable tree structure. A complete guide must distinguish prediction, explanation and intervention.

This audit inventories the documentation and examples, reads the main modeling journeys, compares selected examples with their scripts and API exports, and checks the existing CI configuration. It builds on the [November 12 coverage audit](2026-11-12-modularity-coverage-audit.md). It is not a method-by-method correctness audit, a package comparison, or proof that every example runs. The code graph reported a stale index. Source files are the evidence for the findings below.

## Priority 0: fix misleading or broken journeys

1. **Forecasting leakage.** [Forecasting example](../../examples/timeseries_forecasting.py) computes a window from `sales` on the full series before the train/test split. Its documented first-day rolling means equal that day's sales. Thus the feature includes the outcome being forecast. The reported out-of-time metrics do not establish performance for a forecast made before that outcome is known. Define the forecast origin and horizon, construct features from information available at that origin, and re-run the example. State whether later test-day forecasts use observed or predicted prior outcomes.
2. **Causal identification.** [Causal guide](../../docs/examples/causal-uplift.md) says one treated and one control observation in each child guarantees identifiable treatment effects. Those counts permit a sample difference, but they do not establish a causal effect. Explain consistency, treatment assignment or conditional exchangeability, overlap, appropriate covariate timing and interference assumptions. Separate randomized and observational workflows. Explain that one observation per arm is not adequate evidence of a stable business subgroup. Do not imply that honesty alone fixes confounding. The example uses randomized synthetic treatment and known effects, not a real-world intervention evaluation.
3. **Copyable entry points.** [Quickstart](../../docs/quickstart.md) uses undefined `y_test` and `pgv`. Its corresponding [script](../../examples/quickstart.py) defines the target and treats Graphviz as optional. The [churn walkthrough](../../docs/examples/walkthrough.md) lacks its pandas import and the definitions of `xtr` and `ytr`. It also fits category mappings before a stated train/test split. Publish complete examples that split first, fit preprocessing on training data and use named features.
4. **Claims about estimation.** [Honesty guide](../../docs/examples/honesty-weights.md) states that honest leaf predictions and quantiles are unbiased out-of-sample estimates without conditions. Narrow the claim to what sample splitting does, explain its assumptions and sample-size tradeoff, and report how empty estimation leaves are handled after checking the implementation. [Categorical encoders](../../docs/examples/categorical-encoders.md) claims ordered statistics completely prevent memorization and treats shrinkage as leakage protection. Explain out-of-fold encoding, train-only fitting, evaluation leakage and the limits of each technique.
5. **Practical classification.** The churn guide uses a regression tree and a fixed 0.5 threshold as its classification baseline. A classification guide needs the supported binary and multiclass APIs, probability interpretation, class ordering, threshold and cost selection, calibration, imbalance and held-out metrics. Do not present a generic regressor score as a calibrated probability.

## Priority 1: close the workflow gaps

| Area | Current evidence | Gap and desired outcome |
|---|---|---|
| Discoverability | [Home](../../docs/index.md) and [navigation](../../mkdocs.yml) organize examples by mechanics. | Provide a task map for regression, classification, survival, uplift, clustering, anomalies and forecasting. Link each task to a runnable workflow, API and limitations. |
| Regression and classification | [Model building](../../docs/examples/model-building.md) focuses on a benchmark and knobs. [Distributional leaves](../../docs/examples/distributional-leaves.md) covers many families. | Add separate decision-oriented workflows. Cover targets, input schema, split strategy, model choice, predictions, metrics, error analysis, tuning and handoff. Explain family assumptions and prediction scales. |
| Missingness | [Missing routing](../../docs/examples/missing-routing.md) explains several mechanisms but says missing entries never raise an error. | Document contracts per model and operation, including missing targets, unseen levels, all-missing columns, missing surrogates and training versus inference. Check actual behavior before promising a universal policy. Show audit counts and monitor missingness drift. |
| Categories | The [walkthrough](../../docs/examples/walkthrough.md) shows `IndexMapper` and native typing. | Show train-only mapping, raw strings versus coded values, stable column order, unseen levels, high cardinality and category labels in reports. Compare native splits and encoders without implying that every model handles them alike. |
| Tree inspection | [Inspection](../../docs/examples/inspection.md) lists splits, leaves, paths and local breakdowns. [Rules guide](../../docs/examples/rules-summary-knn.md) describes extraction. | Provide a complete path from named input to node, leaf, condition, support, prediction and export. Specify structured fields, category codes, missing routing, linear leaf terms, rule overlap and default behavior after source checks. Separate association from causal intervention. |
| Uplift | [Causal guide](../../docs/examples/causal-uplift.md) demonstrates synthetic recovery. | Add randomized evaluation, independent holdout, subgroup arm counts, uncertainty, incremental profit and policy comparison. Explain why individual effects and PEHE are unavailable as observed labels in real data. Mark observational analysis as assumption-dependent. |
| Survival | [Survival guide](../../docs/examples/survival-trees.md) shows tuple targets and step functions. | Explain censoring assumptions, time origin, event definition, horizon and output shape. Document median when survival never crosses 0.5 after checking behavior. Add held-out censoring-aware discrimination and calibration guidance. Distinguish cumulative hazard from instantaneous hazard. |
| Clustering | [Comparison](../../docs/examples/clustering-comparison.md) uses undefined `x_matrix`, `feature_names` and `dtypes`; it ranks methods on one custom metric. | Provide runnable preparation and prediction for each supported model. State whether PCT needs target columns, how noise is labeled, whether prediction on new rows is supported, and how scaling and categorical inputs affect each method. Compare stability and business usefulness as well as the custom metric. |
| Anomaly detection | [Comparison](../../docs/examples/anomaly-comparison.md) reports scores and broad best-use claims. | Give a runnable workflow, score direction and scale, training versus streaming contracts, threshold setting, review capacity, feedback and drift. Do not treat rank agreement as anomaly accuracy or a generated narrative as a verified cause. |
| Evaluation and deployment | [Metrics and plots](../../docs/examples/metrics-plots.md) and [inspection](../../docs/examples/inspection.md) catalog outputs. | Specify leakage-safe splits, grouped or temporal holdouts, baselines, reproducibility, model persistence and versioning only where supported, serving schema, monitoring and retraining. Clearly label absent capabilities. |

## Priority 2: references, API and upkeep

- The generated [API index](../../docs/reference/api/index.md) says every public class and function appears, but the generator mapping omits `MarsRegressor`, `AnomalyTree`, `ClusterTree`, `Rule` and `RuleSet` from the root [exports](../../python/mltreelib/__init__.py). Check whether any omission is deliberate, document it, and remove the universal claim until coverage is checked. Classify `TrieKNN` by its actual task rather than assuming it is a clustering algorithm.
- Review generated API pages and regenerate them from the current extension. The earlier audit reported malformed fences; re-check the current output before changing the generator. Generated signatures do not replace input, output and error contracts.
- [Data processing](../../docs/examples/data-processing.md) and [categorical encoders](../../docs/examples/categorical-encoders.md) give different encoder counts. Derive lists from the public API and explain which encoders require targets.
- [CI](../../.github/workflows/test.yaml) runs Rust and Python tests but has no documentation build, public-API coverage check or curated documentation snippet smoke test. Add those checks when the guide changes land. Keep external datasets and long benchmarks outside the ordinary smoke set.
- Audit blog and benchmark claims separately from guides. Examples include universal comparison language and numerical claims without a reproducible dataset, split, settings and uncertainty next to the result. Avoid legal-compliance guarantees and claims of superiority that exceed evidence.
- The `llmstxt` list in [navigation config](../../mkdocs.yml) omits several modeling guides, including survival and causal uplift. Check its intended audience and update it alongside navigation.

## Specification for the next iteration

For each modeling task, publish one discoverable guide with the following sections:

1. **Decision and fit:** the business question, target, suitable method, alternatives and known unsupported cases.
2. **Data contract:** complete input schema, target definition, missing and categorical behavior, treatment or event fields where relevant, and information available at prediction time.
3. **Runnable workflow:** imports, reproducible data, train/validation/test split, train-only preprocessing, fit, predict and evaluation. State optional dependencies and data acquisition. A reader can run the example from a clean environment.
4. **Model interpretation:** output shape and scale, named tree conditions, leaf support, missing-value routing, relevant uncertainty and export options supported by the actual API.
5. **Decision and monitoring:** baseline, appropriate metrics, costs or operational thresholds, failure modes, leakage checks, drift and a handoff checklist. Explain when evidence does not justify an intervention.
6. **Theory and references:** a short explanation of the method and its assumptions, links to primary papers, and links to exact API methods and full scripts. Attribute the implementation only to the specific method it actually uses.

Every code block advertised as runnable needs a smoke check. Each guide needs a reviewed example of missing or unseen inputs where the model supports them. Examples with synthetic truth must label it, and published metrics must name the data, split and seed. Review any dataset or model persistence requirement against the real implementation before promising it.

## Proposed order and acceptance gates

1. **Trust repairs:** fix the forecasting leakage, causal and honesty claims, quickstart and churn snippets. Gate: run the published examples and inspect the rendered pages.
2. **Core workflow:** regression and classification, plus shared missingness, categoricals and structured tree inspection. Gate: a reader can train, evaluate and inspect named paths on held-out mixed data, including missing and unseen values.
3. **Decision workflows:** uplift and survival, with explicit assumptions, appropriate validation and limits. Gate: randomized uplift and censored-survival examples run and identify what their evaluations can and cannot prove.
4. **Unsupervised workflows:** clustering and anomaly detection. Gate: runnable examples show input contracts, out-of-sample behavior, score or label meaning, and operational evaluation for each covered method.
5. **Lifecycle and reference:** forecasting, persistence and monitoring where available, API coverage, generated docs and CI. Gate: docs build, checked snippets run, and exported symbols are mapped or explicitly excluded.

The user can review each phase independently. Features uncovered during documentation work belong in the roadmap or a separate implementation specification. Do not document proposed functionality as shipped.

## Product questions to investigate, not claims of absence

The audit does not establish drop-in compatibility with any named library. Compare fit and prediction signatures, parameter semantics, serialization, input containers, probability shapes, estimator integration and missing/category behavior against selected alternatives before making compatibility claims. Check structured export, category-label preservation, confidence intervals, uplift and survival metrics, and per-model missing-data behavior in source and tests. Classify each result as supported, limited or absent, then prioritize the gaps separately from documentation repairs.

## Primary references for the next pass

- Athey and Imbens, [Recursive partitioning for heterogeneous causal effects](https://www.pnas.org/doi/10.1073/pnas.1510489113) — causal trees and honest estimation.
- Kaplan and Meier, [Nonparametric estimation from incomplete observations](https://doi.org/10.1080/01621459.1958.10501452) — product-limit survival estimation and censoring assumptions.
- Breiman, Friedman, Olshen and Stone, [Classification and Regression Trees](https://doi.org/10.1201/9781315139470) — tree growth, pruning and interpretation.
- Kaufman et al., [Leakage in data mining: Formulation, detection, and avoidance](https://doi.org/10.1145/2382577.2382579) — information leakage in model development.
- scikit-learn, [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) — practical split and preprocessing examples. This is a reference, not a package dependency.

These references are starting points. Check each source against the precise claim when writing the guides.

## Proposed reader journey

Organize the site around decisions in this order:

1. **Start here:** install, a complete first fit, input conventions, and a task selector.
2. **Prepare data:** missing values, categorical types, train-only transformations, leakage and feature names.
3. **Choose a task:** regression, classification, clustering, anomaly detection, survival, uplift or forecasting. Each task starts with a decision guide and a complete small example.
4. **Evaluate:** task-specific baselines, held-out design, metrics, uncertainty and failure analysis.
5. **Understand and act:** named paths, leaf support, rules, explanations and the limits of causal interpretation.
6. **Operate:** supported persistence and input contracts, monitoring, drift and retraining. Mark missing capabilities explicitly.
7. **Reference:** model contracts, parameters, methods, examples, benchmarks and primary sources.

For clustering, proceed from a complete fit and segment profile to model selection, mixed inputs, noise and new-row behavior. Follow with stability, held-out usefulness and a reproducible comparison. For anomaly detection, proceed from score interpretation and review capacity to threshold selection, contextual explanations, feedback, drift and labeled evaluation when available. Document each model's actual support before promising shared behavior.

Every task path needs the same progression: decision, data, baseline, fit, evaluate, interpret, act, limitations and next steps. Benchmark pages need data provenance, split, preprocessing, budgets, seeds, metrics and failed runs. Use small runnable examples before longer comparisons. Add navigation and smoke checks as each path is completed.

## Progress after the first pathway pass

- **Phase 1, trust repairs: substantially complete.** The forecasting example uses prior-day outcomes, the causal and honesty claims are narrower, and the quickstart and churn snippets run. A full review of all existing claims remains.
- **Phase 2, core workflow: started.** Regression and classification now have runnable synthetic pathways. Shared missing-value and categorical contracts and structured tree export remain to be checked per model.
- **Phase 3, decision workflows: started.** Survival and uplift now have introductory pathways, but they do not yet contain independent censoring-aware survival evaluation or a randomized policy evaluation with uncertainty and costs.
- **Phase 4, unsupervised workflows: started.** Clustering and anomaly pathways use the unified `ClusterTree` and `AnomalyTree` interfaces. The remaining work is to document each supported `kind`, its inputs, missing values, prediction behavior, score or label semantics, and practical evaluation.
- **Phase 5, lifecycle and reference: pending.** Persistence, deployment, API coverage, generated references and CI checks still need source-backed review.

### Next implementation step

Build out the unified clustering and anomaly pathways first. Check every facade option against its implementation and tests. Add small runnable examples and task-specific figures, then compare stability and operational usefulness using reproducible data and budgets. Show limits and unavailable methods explicitly. Follow with shared missing-value, categorical and tree-inspection contracts. Then deepen survival and uplift evaluation before claiming complete decision workflows.

## Follow-up: model examples, data and metrics

The model examples are useful but mix task introductions, configuration catalogs and benchmarks. Keep the existing pages and links while their content is reviewed. Use the task pathways as entry points and the model-extension guide as a decision map. Consolidate repeated explanations only after checking their claims and inbound links. The knob tour belongs after a working baseline. Tobit, constraints and retraining are advanced workflows, not mandatory steps for every model. Keep benchmark results separate from instructions and label their datasets, splits and limits.

Data and metrics are not complete as end-to-end user journeys. The new preparation and evaluation pages supply a first sequence and link to existing material. Remaining work includes per-estimator missing and unseen-category contracts, leakage-safe preprocessing across folds, stable schema and persistence, task-specific metrics for survival and uplift, anomaly and clustering evaluation, subgroup and uncertainty guidance, and runnable monitoring examples. The catalog contains conflicting descriptions of some metrics signatures. Check each against source and tests before treating it as authoritative. Do not infer that the package implements every metric recommended in the guide.


## Checked contracts on 2026-11-18

Both the committed worktree build (0.4.4) and an isolated build of the current original-checkout source (0.4.5, including its uncommitted Rust and Python changes) gave the following results. The latter build used a separate snapshot, virtual environment and Cargo target. These observations do not prove all cases.

- On the healthcare training set and the 24-point age grid in `examples/constraints_check.py`, `monotone=[+1 on age]` produced no decreases with mean leaves. With linear leaves it produced two decreases: age 42 to 44 (20522.277 to 9202.407) and 58 to 60 (48176.713 to 15310.416). Treat global monotonicity for that configuration as a product gap and investigate before promising it.
- A mean-leaf ModelTree accepted raw `north`/`south` strings and `None` numeric predictors with declared types. For the repeated four-row synthetic sample in `docs/guides/data.md`, prediction for `["unknown", None]` was 9.5 while `[None, None]` was 1.5. This differs from the repository's unseen-as-missing rule. Investigate category mapping and routing as a product gap.
- For training values `["a", "a", "b", None]` and queries `["a", "unknown", None]`, `IndexMapper` returned `[0, None, None]`; one-hot returned all zeros for both unknown and missing; count encoding returned `[2, 0, None]`; target encoding with smoothing 2 returned approximately `[0.5833, 0.6667, None]`. Encoder behavior is method-specific and does not imply encoding is required for ModelTree.
- A churn example passes raw category strings and missing values directly to ModelTree. Fitted tree rules still display categorical codes. Check whether users can obtain human-readable category labels and actionable paths as structured outputs. A predictive path alone does not identify an intervention effect.


## Data, metrics and visual diagnostics follow-up

The `examples/data_metrics_pathway.py` workflow uses raw categories and missing numeric values, separate training/validation/test partitions, a training-mean regression baseline and a validation-selected decision threshold. Its visual diagnostics show missingness, residuals, precision–recall, calibration, confusion, split importance and the tree. The metrics showcase now uses native categories and a training-set baseline.

A further product gap appeared when the showcase passed a raw mixed-type row to `ModelTree.break_down`: prediction accepted the row but `break_down` raised `TypeError: must be real number, not str`. The showcase uses a separate numeric-only model for its break-down chart. Check mixed-type explanation support and structured category labels in the later contracts phase.


## Probability calibration gap

`Metrics.calibration` produces reliability bins, and `plots.plot_calibration` draws them. Brier score and log loss assess probability quality. These diagnostics do not fit or apply a post-hoc probability calibrator. The inspected public exports and model methods do not expose a fitted calibration workflow. Add probability recalibration to the roadmap, with held-out or out-of-fold fitting and independent evaluation. The algorithm and interface remain to be specified before implementation.


## Prediction contract checks

On the committed 0.4.4 worktree, a two-column mean-leaf ModelTree accepted a one-cell row and a three-cell row without an error. A numeric column supplied as a string raised `ValueError`. Python `pickle.dumps` on the fitted model raised `TypeError`. These are focused observations, not a complete deployment audit. The new prediction and schema guide checks field names in application code and records these gaps. An isolated build of the current original-checkout source reproduced the short-row, extra-row and pickle results. Define persistence and compatibility before production handoff.


## Unified clustering checks

The same `python/mltreelib/models.py` facade is present in the documentation worktree and current original checkout. Source inspection shows `ClusterTree` converts `None` to `0.0` for CLTree, HDBSCAN and URF in both fit and prediction. This conflicts with the missing-data-first contract and requires a product fix, not a documentation-only workaround.

A synthetic two-group numeric check exercised fit, labels, new-row prediction and `rules()` for all four kinds. PCT returned rules; CLTree returned bounding boxes and `None` for some rows; HDBSCAN and URF returned empty rules. The observed results and code are described in the clustering guide. PCT accepted raw strings and missing numeric inputs in a separate mixed-data example, but rendered category codes and routed an unseen category differently from a missing one.

The `examples/clustering_workflow.py` example adds independent training and review sets, cluster sizes and a label-permutation-invariant pairwise stability check. Its fixed-seed agreement was 0.788. That number is illustrative and does not establish cluster quality.


## Unified anomaly checks

`AnomalyTree` facade source was checked in the documentation worktree. HST, RCT and active paths replace `None` with `0.0` in fit and scoring, conflicting with missing-data-first behavior. EIF and DET accepted a small mixed-type input with raw categories and missing values, but that check does not establish all input contracts.

A synthetic numeric check exercised fit, `score_samples` and `predict` for EIF, DET, HST, RCT and active. Context fit accepted integer sequences, but `score_samples` raised a message referring to `surprise_score`, which is not exposed on the facade. The active facade did not expose a feedback or update method. A tied-score example with 20 identical rows and contamination 0.1 flagged all 20. The facade uses a training cutoff and `>=`, so ties can exceed the nominal fraction.

`examples/anomaly_workflow.py` fits independent reference data and ranks a separate 100-row review set with 10 planted anomalies. At a fixed budget of 10, 7 planted anomalies were reviewed. The training cutoff flagged 17. These are illustrative synthetic results, not a benchmark.


## Larger synthetic pathway checks

The initial 100-row anomaly review example and 80-row clustering review example above are historical. The examples now use at least 1,000 review or test rows. The anomaly workflow fits 2,000 reference rows and reviews 1,100, including 100 planted anomalies. It finds 75 at a review budget of 100. The clustering workflow fits 2,000 rows and assigns 1,000 held-out rows; its pairwise stability agreement is 0.750 across two fits. The mixed-data metrics example uses 2,000 training, 1,000 validation and 1,000 test rows. The six introductory modeling examples also use at least 1,000 rows per displayed evaluation or fit. These larger synthetic samples improve the demonstrations; they do not establish real-world performance or replace repeated seeds and independent datasets.


## Real data check: Bank Customer Churn

OpenML 46911, the 10,000-row TabArena Bank Customer Churn dataset already present in the original workspace, was copied to the isolated worktree as an untracked dataset. `examples/real_data_workflow.py` uses seven features, seed 2026 and 6,000/2,000/2,000 training/validation/test rows. The selected training columns have no missing values. A binomial ModelTree gives test average precision 0.5238 and Brier score 0.1215 against a training-rate baseline of 0.1639. A validation-selected threshold 0.2 gives test confusion [[1430,157],[187,226]]. PCT assigns four test groups of sizes 444, 530, 561 and 465. EIF scores 2,000 test records; churn is not an anomaly label, so anomaly accuracy is not established. This single random split is an exploratory example, not a benchmark or evidence of temporal deployment performance.


## Synthetic regression selection and real-data explanation

`examples/regression_real_and_synthetic.py` compares four ModelTree leaf/depth configurations on 3,000 synthetic training and 1,000 validation rows, then evaluates the selected setting on 1,000 held-out rows. Numeric predictors are missing in 169 test rows, and raw category labels are passed directly. Linear leaves at depth 2 achieve validation RMSE 2.624 and test RMSE 2.935 versus a training-mean baseline of 6.385. This is one seed and four configurations.

The selected setting then fits 6,000 and scores 2,000 randomly selected records from OpenML 46923 diamonds. Test RMSE is 1381.867 versus 3982.039 baseline, and MAE is 772.555 versus 3012.514. The selected training features contain no missing values. Splits use named size features. The random split and synthetic selection do not establish the optimal model or causal interpretation.


## Survival, uplift and real anomaly evaluation

`examples/survival_uplift_evaluation.py` evaluates survival on 2,000 held-out synthetic rows, including 609 censored observations. At horizons 10, 20 and 30, tree IPCW Brier scores are 0.2417, 0.2000 and 0.1226, compared with 0.2509, 0.2102 and 0.1316 for a global survival baseline. The censoring distribution is known from the simulation. These example calculations are not a public metric API.

The uplift evaluation uses 3,000 training, 1,000 validation and 2,000 independent randomized test rows. The selected policy yields estimated incremental net outcome 0.7072 per person versus -0.0478 for treat-all and 0 for treat-none. Its approximate 95% interval is [0.6438, 0.7706], conditional on the selected fraction and excluding model-selection uncertainty. PEHE 0.2347 and policy truth 0.7252 are available only because the data is synthetic.

The UCI Statlog Shuttle archive (https://archive.ics.uci.edu/dataset/148/statlog+shuttle) supplies 43,500 train and 14,500 test records. The anomaly example trains EIF on a seeded subset of 5,000 class-1 train records and treats classes 2–7 as a proxy for anomalies. On the official test split, 648 of the top 725 scored records are proxy positives: precision 0.8938, recall 0.2144, AP 0.5988, prevalence 0.2084. These are operating-state labels rather than verified anomaly incidents. The source notes that the original time order was randomized. Dataset files are ignored by Git.

### Numeric boundary displayed by DOT

A depth-one mean tree fit ten rows at zero and ten at one, with targets
zero and ten, respectively. The split threshold was zero. Predictions at
`[-0.001, 0, 0.001, 1]` were `[0, 0, 10, 10]`. DOT labeled the two branches
`< 0` and `>= 0`, which assigns equality to the other branch. This is a
verified export gap. A regression check for the fix must compare the displayed
operator with prediction, including exact equality and the missing route.
