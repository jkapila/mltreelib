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
