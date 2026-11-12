# mltreelib relevance and adoption ecosystem

**Date:** 2026-11-12  
**Decision:** Where a Rust-backed tree package can matter in the present tabular-AI market, and which adoption investments should precede agent integrations. This is a recommendation, not an implementation or a claim that mltreelib currently outperforms incumbents.

## Judgment

mltreelib has a credible **research and specialist-practitioner niche** if it becomes a reliable way to model and inspect structured, grouped, temporal, and categorical tables. It is not yet a credible general-purpose accuracy replacement for LightGBM, CatBoost, AutoGluon, or tabular foundation models. Its strongest potential product is an *honest experiment and explanation workflow*: reproduce a dataset split, fit a transparent tree, evaluate it against strong baselines under equal budgets, and report which structure the model recovered. That value survives even when another model wins predictive accuracy.

The market remains relevant. The original TabArena paper says gradient-boosted trees remain strong on practical tabular tasks, deep learners catch up under larger budgets, and foundation models excel on smaller data [1]. BeyondArena's October 2026 revision evaluates 11 models on 142 datasets and reports that traditional tree and deep learning models still dominate grouped or large settings while foundation models are competitive or strong up to 100k rows [2]. But TabPFN-3.5's November 2026 report claims stronger results on temporal/grouped data and high-cardinality features [3]. The niche is therefore **contested and moving**, not an uncontested tree advantage. All direct mltreelib superiority claims require a current, reproducible benchmark against these systems.

## What the repository can already build on

The package has a Rust core with Python bindings, typed ingestion and categorical support, grouped/time transforms, a synthetic generator with a ground-truth manifest, several tree/anomaly/clustering families, metrics, and model inspection. These are visible in the current exports, examples, and [modularity audit](2026-11-12-modularity-coverage-audit.md). The local Rust and Python suites passed there (177 and 125 tests). This proves implementation breadth and working behavior, **not** comparative quality, package discoverability, or release readiness.

The project's own [Elo board](../../docs/benchmarks/elo.md) is a useful corrective to ambitious positioning. On its 18-dataset regression field, `ModelTree(all)` is listed at 1050 Elo with 16/18 completed datasets, versus LightGBM(200) at 1470 and CatBoost(200) at 1425 with 18/18. On its classification field, `ModelTree(all)` is 698.5 log-loss Elo with 16/24 completed datasets, behind CatBoost(200) at 1440.8 with 23/24. These local rounds are not a universal ranking or a fair comparison against every tuned configuration, but they do not support a general accuracy-lead claim. The sensible public claim today is a transparent research and specialist workflow whose value must be demonstrated per use case.

The differentiator worth testing is the combination of:

1. **Structure recovery** on generated data with known categorical/group/hierarchy effects, rather than score-only model comparison.
2. **Interpretable specialist trees** (model-tree leaves, rules, contextual anomaly and clustering outputs), with inspectable decisions and CPU-friendly deployment as hypotheses to measure.
3. **Validation that respects groups and time**. BeyondArena explicitly models IID, temporal, and grouped tasks [2, 4]. A model that uses the wrong split can look good for the wrong reason; a workflow that makes the split contract visible is valuable to humans and agents.

The package should present these as **specific use cases**, not as a claim that all its algorithms are already superior. A broad algorithm menu alone is weak differentiation when mature packages already cover common prediction tasks.

## Adoption work, in priority order

### 1. Make the first ten minutes trustworthy

Publish one install path that works on supported Python versions; a small copied quickstart that runs exactly as shown; an end-to-end example for a real grouped or temporal dataset; and a clear table of supported input types, missing-value semantics, fit/predict outputs, and optional dependencies. The existing audit found a broken displayed quickstart, malformed generated API code fences, and omitted public classes [5]. Fix those before asking agents to use the docs as instructions. Add a tiny docs/example smoke gate to CI.

Also reconcile the [installation guide](../../docs/installation.md), which says “no transitive dependencies,” with the declared `pyarrow>=14` runtime dependency in [pyproject.toml](../../pyproject.toml). A newcomer should be able to trust the dependency and install story before judging model behavior.

### 2. Publish a reproducible comparison, including losses

Enter the TabArena/BeyondArena workflow as a **model or system** under the first-party harness, which describes shared validation, tuning, refitting, and leaderboard submission [4]. Show comparisons against tuned LightGBM/CatBoost and relevant tabular FMs with equal budgets, inference latency, memory, completion rate, and per-dataset results. Include generated-data structure-recovery metrics separately. A single favorable Elo number is insufficient; report where the package loses and why. This is the evidence required before positioning mltreelib as a stronger learner.

### 3. Improve ordinary interoperability

Verify and document reliable DataFrame/Arrow/Parquet input and output, stable `fit`/`predict`/`predict_proba` behavior, model persistence, and machine-readable evaluation/explanation reports; implement only the missing parts that block a demonstrated workflow. Arrow's Python documentation identifies first-class NumPy/pandas integration and Parquet support [6]; using those existing conventions lowers adoption friction. Measure copies and memory before promising zero-copy behavior. Keep optional plotting and benchmark dependencies outside the core install.

### 4. Focus the public surface and contribution story

Use the proposed data / learner / metrics ownership map, preserve top-level imports, and document a few recommended entry points rather than sending newcomers into every algorithm class. Provide a model card or method table that states task, data types, missingness, fit-time scale, output, and research maturity. Keep citations and reproducible benchmark recipes close to each method. Add a `CONTRIBUTING.md` and a small set of beginner issues only when the corresponding tests and review expectations are clear.

## Agent-installable skills: yes, after the workflow works for humans

The official Agent Skills specification defines a portable directory with `SKILL.md` and optional scripts/references; the skill's description determines when an agent loads it [7]. A skill can teach an agent **how to use mltreelib safely and repeatably**, but it cannot repair missing package behavior. It will amplify both good and broken workflows.

Start with **one** task-specific skill, tentatively `mltreelib-tabular-audit`:

- Input: a local table plus target and optional group/time columns.
- Workflow: inspect schema and missingness, choose a leakage-safe split, fit a transparent baseline with the public Python interface, compute held-out metrics and a model explanation, and state uncertainty and failure cases.
- Output: a short human-readable report and a stable JSON/CSV summary, including split definition, seed, package version, metric, runtime, missing-row counts, and artifact paths.
- Safety: never silently infer a target or group/time column, leak targets into preprocessing, or claim causal structure from predictive feature importance. Ask when those choices are material.
- Acceptance: run it against two checked-in small fixtures (ordinary IID and grouped/temporal) and compare the produced report with expected behavioral assertions. Validate the skill format with the official reference validator [7].

Only after that workflow is useful should the project consider a second skill for **ground-truth structure-recovery experiments**. A generic “use mltreelib” skill, an MCP server, or a hosted agent service adds a surface before there is a stable task contract. An MCP server becomes justified if agents need a remote dataset/model registry, long-running jobs, or shared model artifacts; local package use needs none of those.

## Decision gates

| Gate | Evidence that would change the recommendation |
|---|---|
| Broader predictor positioning | Competitive, reproducible TabArena/BeyondArena results across representative tasks under fair budgets, with acceptable runtime and completion rates. |
| Agent skill launch | A documented human workflow, stable machine-readable outputs, two fixture runs, and green quickstart/docs gates. |
| Separate installable data/metrics/learner packages | Demonstrated users who need independent installs or measurable build/footprint gains after accounting for release and ABI cost. |
| MCP or hosted integration | Proven need for remote/shared execution that a local skill and Python package cannot serve. |

## Sources and evidence limits

[1] Erickson et al., *TabArena: A Living Benchmark for Machine Learning on Tabular Data*, NeurIPS 2025, revised 2025-11-03: https://arxiv.org/abs/2506.16791  
[2] Purucker et al., *Beyond IID: How General Are Tabular Foundation Models, Really?*, NeurIPS 2026, revised 2026-10-30: https://arxiv.org/abs/2606.30410  
[3] Jäger et al., *TabPFN-3.5: Technical Report*, revised 2026-11-03: https://arxiv.org/abs/2609.17895 — vendor/coauthor claims, not independent validation of every result.  
[4] AutoGluon TabArena first-party README and BeyondArena examples (accessed 2026-11-12): https://github.com/autogluon/tabarena and https://github.com/autogluon/tabarena/tree/main/examples/beyondarena  
[5] Repository audit, 2026-11-12: [modularity and coverage](2026-11-12-modularity-coverage-audit.md).  
[6] Apache Arrow Python documentation (accessed 2026-11-12): https://arrow.apache.org/docs/python/  
[7] Agent Skills official specification (accessed 2026-11-12): https://agentskills.io/specification

External sources establish the competitive setting and the skill format. The specific product sequence is an inference from those sources and the audited repository. No external user-demand interviews, package download counts, current mltreelib leaderboard entry, or cost measurements were found or claimed here.
