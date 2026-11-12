# Modularity, tests, examples, and documentation audit

**Date:** 2026-11-12  
**Scope:** Current working tree, including uncommitted work. This is an audit and recommendation, not an implementation plan approval. No package code was changed for this audit.

## Executive decision

The **data / learners / metrics** split is a sound way to organize the project. Use it first as ownership and Python namespace boundaries within the existing distribution. Do **not** promise three independently installable or versioned packages yet: the current root package imports one PyO3 extension, and a physical split would add release, ABI, and compatibility work without evidence of a user need for separate wheels. Keep root imports working.

The existing [decomposition spec](../specs/SPEC-data-core-metrics-decomposition.md) already describes the intended split. Revise its claims about independent versioning, fast incremental compilation, standalone installation, and FFI before treating it as an execution contract. Those benefits do not follow from adding Python namespaces, and independent wheels require a separate packaging design.

## Evidence and method

- Refreshed the code-review graph against the current working tree with a **full rebuild**: 145 files parsed, 1,703 nodes, 24,840 edges, no build errors. The graph is a navigation aid; the findings below were checked in source. The graph does not establish runtime test or documentation coverage.
- Read the Rust module exports, Python root exports, packaging, API generator, docs navigation, examples, test locations, and the existing decomposition spec. This audit includes untracked `python/mltreelib/models.py`, the new wrapper tests and benchmark, and in-progress model-tree changes.
- Ran `cargo test --workspace --quiet`: **177 passed**, zero failed. Ran `uv run pytest tests/ -q`: **125 passed**, zero failed. This proves the current suites pass locally; it does not prove the public surface is uniformly tested. No benchmark or docs-site build was run.
- Inventory: **48** root Python exports, **16** Rust integration test files, **20** Python test files, **56** Python files in `examples/`, **19** example guide pages, and **7** generated per-domain API pages. These are inventory counts, not coverage percentages.

## Modularity assessment

| Domain | Current owner and evidence | Assessment |
|---|---|---|
| Data | `frame`, `schema`, `builder`, `index`, `encode`, `transform`, `timefeat`, `generator` in `crates/mltreelib-core/src/`; dataset loader in `python/mltreelib/datasets.py` | Cohesive functional area, but no public `mltreelib.data` namespace. `encode.rs` uses `nn.rs`, so a Rust data crate extraction must carry or relocate the embedding implementation deliberately. |
| Learners | `model_tree`, split/leaf/gate/prune/family modules, `cluster/`, `anomaly/`, `trie_knn`, `mars` | Already internally separated by function and model family. All still compile in one Rust crate. `winnow` is a model column-screening tool (`preselect`), although the spec places it under data. |
| Metrics | `crates/mltreelib-core/src/metrics.rs`, Python `Metrics` binding, optional `python/mltreelib/plots.py` | Good candidate for a standalone Rust crate because the metric report code is largely value-based. The current Python `Metrics` still arrives through the monolithic `_core` extension, so a namespace alone gives organization, not standalone installation. |
| Bindings | `crates/mltreelib-py/src/lib.rs` and a flat `python/mltreelib/__init__.py` | One extension and root facade couple build and distribution. Keep this compatibility seam while clarifying ownership. |

**Recommended dependency direction:** data types and transforms have no learner dependency; learners consume data; metrics consume predictions and targets as values, without importing learners. The Python binding may depend on all three. Do not put model-specific split screening or tree diagnostics into the data module merely because they accept columns.

## Test uniformity

**Strengths.** The Rust and Python suites both pass. Rust has domain-oriented integration files including `builder.rs`, `encoders.rs`, `features.rs`, `metrics.rs`, `model_tree.rs`, `anomaly.rs`, and clustering-specific files. Missing-data and deterministic behavior are exercised in `tests/test_dataset.py` and Rust tests. The new unified wrappers have a Python test file.

**Gaps to address before a package split:**

1. Python data, encoders, transforms, metrics, demo execution, and a model-tree smoke test are concentrated in [tests/test_dataset.py](../../tests/test_dataset.py). The metrics Python binding has one broad test there, while [crates/mltreelib-core/tests/metrics.rs](../../crates/mltreelib-core/tests/metrics.rs) has 20 focused Rust tests. Keep the Rust oracle, then add focused Python interface checks for the metric families and plots that users actually call. File names alone are not proof of missing behavior.
2. [tests/test_unified_trees.py](../../tests/test_unified_trees.py) checks EIF, DET, HST, RCT, PCT, HDBSCAN, and URF paths, but not every advertised wrapper kind or the key error and missing-value contracts. The wrappers are new Python behavior rather than thin re-exports; they deserve tests for each supported dispatch path and the threshold contract. Avoid duplicate numerical tests already covered in Rust.
3. The CI workflow runs Rust core tests and Python tests across 3.10–3.14, but has no docs build, generated-reference freshness check, or runnable example gate ([test.yaml](../../.github/workflows/test.yaml)). Those are the main ways the observed documentation drift could reach users despite green suites.

## Example and documentation coverage

The example inventory is broad: data, encoding, metrics, tree building, missing routing, clustering, anomaly, and benchmarking all have scripts. The 19 docs example pages are curated guides, while many of the 56 Python example files are benchmark drivers or helpers; a one-to-one page requirement would create noise. A **small smoke set** of user-facing examples is the useful gate: quickstart, data, metrics, one clustering model, and one anomaly model. Keep large benchmark jobs out of ordinary CI.

Concrete defects and drift:

1. [docs/quickstart.md](../../docs/quickstart.md) uses `y_test` without defining it, whereas [examples/quickstart.py](../../examples/quickstart.py) does define it. The displayed sample cannot run as copied. Its `pgv` use likewise omits the example script's optional import/fallback.
2. The [API generator](../../examples/gen_api_doc.py) appends the opening code fence and code on adjacent strings without line breaks. The generated [ModelTree page](../../docs/reference/api/model-tree.md) contains ` ```pythonclass ModelTree...``` ` and analogous malformed fences. A literal scan finds **172** ` ```python... ` instances across generated API pages. Fix the generator, regenerate, and build the site; hand-editing generated pages will recur.
3. The generator's hard-coded `PAGES` list omits the public `MarsRegressor`, `AnomalyTree`, and `ClusterTree`. A symbol-presence scan across documentation found these three absent, though all 48 root exports appear in at least one test and one example. Symbol presence is a weak measure; it does not establish usable explanations or examples.
4. The generated [API index](../../docs/reference/api/index.md) says every class and function is documented, but the omissions above disprove it. The root exports and `PAGES` list can drift independently; add a check that each public root symbol is either mapped to a page or deliberately excluded.
5. [README.md](../../README.md) still displays a 0.3.0 badge while Cargo and Python package metadata say 0.3.3. The docs and source therefore need a small consistency pass before a structural move.

## Recommended sequence for the next agent

1. **Repair the evidence surface first.** Fix the API generator's fences, include the three missing public classes, regenerate docs, correct the quickstart, and add a lightweight generated-doc / example smoke gate. Verify the docs build. These are observable defects and can be reviewed as a separate change.
2. **Define one ownership map.** Amend the existing spec with precise symbol placement and dependency direction, including `winnow`, `nn`/embedding encoders, `plots`, generator, and the new unified wrappers. Decide whether the Python learner namespace is `mltreelib.tree` with `cluster`/`anomaly` children or a broader `mltreelib.models`; do not create both as competing top-level public interfaces. Preserve all root imports.
3. **Create Python namespaces within the existing distribution.** Use thin re-exports for data, learner, and metrics domains, and test import identity, root compatibility, and representative end-to-end behavior. This improves discoverability without a Cargo or wheel migration.
4. **Extract Rust crates only if measured benefit remains.** Metrics is the least coupled candidate. Data extraction follows after resolving the encoder/`nn` placement and shared frame contract. The learner crate stays the consumer. Measure clean and incremental build time and check binary size before claiming faster builds or lighter deployment. Separate installable Python distributions and independent versions are a later product decision, not a consequence of this refactor.

## Risks and scope limits

This is a source and passing-suite audit, not a mutation or full method-by-method coverage study. It did not run the docs builder, plotting extra, external benchmark datasets, or the example corpus. The working tree has extensive pre-existing changes; the next agent should use this note as a baseline and re-check status before editing. The existing spec is a proposal and should not be treated as already implemented.
