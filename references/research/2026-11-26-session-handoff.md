# mltreelib Handoff — PR #1 Validation & PR #2 (Item 6 + Next Version) Plan

- **Date:** 2026-11-26
- **Repo:** `/mnt/d/Programs/mltreelib`
- **Branch:** `spec-33-34/api-consistency-and-data-suite` (HEAD `b330193`, not pushed, no PR yet)
- **Current Version:** `0.4.15` (do not bump until PR #1 is cut/merged per maintainer workflow)

> **Update (end of 2026-11-26):** HEAD is now the Spec 35 status commit after `b330193`; a second verification pass of `b330193` is green (802 passed, 4 skipped). The PR body is drafted in `/tmp/mltreelib-pr-description.md` (copy it into the pull request; closing keywords for #16-#18, #20-#22, #27-#35, references for #19, #23-#26, #36). Issues 28 to 36 were opened today from the sanity pass and two Haiku probes; they stay open until the PR merges.

---

## 1. Current State & Verification (PR #1 Ready)

All work planned for PR #1 is committed and verified on `spec-33-34/api-consistency-and-data-suite`:

- **Key commits on branch (newest first):**
  - `b330193` — `fix: resolve verification issues #34, #35, and #36(a-f)` (`competing_risks_report` NaN filtering, unconditional `group_col`/`id_col`/`protected_attribute` validation, `flchain` cache alias + SHA-256 cache check, `dtypes=` hint, `leaf_samples` guard, doc fixes)
  - `ad22a9f` — `feat(metrics): add competing_risks to MetricsReport and migrate survival docs` (plus `#32c` `id_col`/`leaf_samples` sections in `docs/guides/survival.md` and `docs/guides/uplift.md`)
  - `9ec6d47` — `fix(input): validate table extraction, predict columns, missing y, and id_col (#30-#33)`
  - `8b4bd4e` — `docs: fix anomaly ruff warnings, regenerate data API docs and study figures`
  - `c89474b` — `test(solvers): make cvx_reference independent CVXPY MIP/pruning solvers`
  - `29999ba` — `feat(datasets): repair SurvSet registry and add Spec 34 survival/uplift loaders`
  - `e237108` — `docs: Spec 35 bug register for the model input boundary (issues 30-33)` + Rust `clippy` fixes
  - `171c8d0`, `83b3e17`, `e44426c`, `27915aa`, `660a7d5`, `f30a421` — Docs/examples migration to `Dataset` & `MetricsReport`, issue `#28` API doc generator fix

- **Dataset packaging confirmation:**
  - `pyproject.toml` (`[tool.maturin]`) bundles zero datasets in wheels or sdists. Datasets live in `datasets/` (untracked) or `~/.cache/mltreelib/datasets/` (`MLTREELIB_DATA_DIR`), fetched only on explicit `load_dataset` / `load_survival_dataset` / `load_uplift_dataset` calls, blocked when `MLTREELIB_OFFLINE=1`.

- **Verified gates (all passed after final edit on `b330193`):**
  - `cargo clippy --workspace --no-deps -- -D warnings` -> 0 warnings
  - `cargo test -p mltreelib-core` -> 0 failures
  - `UV_OFFLINE=1 uv run maturin develop --release` -> installed `mltreelib-0.4.15`
  - `uv run --no-sync pytest tests -q` -> `802 passed, 4 skipped` (`polars` skipped in base venv; live network test skipped unless `MLTREELIB_NETWORK_TESTS=1`)
  - `uv run --no-sync python scripts/docs/run_snippets.py --check` -> passed
  - `uv run --no-sync python scripts/docs/check_nav.py` -> passed
  - `uv run --no-sync python scripts/docs/check_benchmarks.py` -> `ok: 35 entries, 2 published`
  - `uv run --no-sync zensical build` -> `No issues found`
  - Pre-commit (`ruff`, `ty`, `cargo clippy`) -> passed

- **Optional extra checks for final PR #1 sign-off:**
  - `uv run --with polars --with pyarrow pytest tests/test_sanity_basics.py tests/test_table_input.py -q`
  - `uv run --offline --with cvxpy pytest tests/test_solver_oct.py tests/test_solver_prune_and_rules.py -q`
  - Optional live network check: `MLTREELIB_NETWORK_TESTS=1 uv run --no-sync pytest tests/test_dataset_registry.py -q`

---

## 2. What Was Intentionally Left Out of PR #1

1. **Work-order Item 6** (`specs/35-simple-object-interfaces-and-family-summaries.md` Parts C & D, issues `#24`, `#25`, `#26`, plus `Dataset` input on `ClusterModel`, `AnomalyModel`, `ComponentModel`).
2. **Spec 34 multi-dataset results pages** (cross-dataset survival/competing-risks and uplift comparison pages using the new `load_survival_dataset` / `load_uplift_dataset` loaders).
3. **Spec 33 Section F long-run / cosmetic remainders:**
   - `examples/guides/hpo_shapes.py` end-to-end Optuna run (up to 10 min per study x 21 studies).
   - `docs/guides/metrics-plots.md:124` still calls `Metrics().stability_index(...)` (PSI drift is not a `MetricsReport` task).
   - `#36(g)`: 393 `ruff` lints in `tests/`, `scripts/`, `examples/` (pre-commit scopes `ruff` to `python/mltreelib/`).
4. **Work-order Item 10** (Elo benchmark upgrade + scale-free `cp` default search).

---

## 3. Deep Spec Study: Recommended Scope for PR #2 (Next Version Bump)

PR #2 is centered on **Item 6** (a major user-facing explainability and API-completion release). Combining the open items from `specs/35`, `specs/34`, `specs/33`, `specs/25`, and `specs/27` gives a cohesive release plan:

### Tier A — Core of PR #2 (Must Land Together)
1. **Issue `#24` — Per-family leaf summaries, `ModelTree.report()`, and conformance test (`specs/35` Part C):**
   - `ModelTree.report()` returning the 4-section `ModelReport` (`Objective`, `Learner`, `Controls`, `Diagnostics`, plus `Inference` when mathematically valid).
   - `ModelTree.summary()` / `leaf_summary(id)` for **all** leaf/family combinations (GLM binomial/probit/Poisson/Gamma/Tweedie/Tobit/NegBin/Beta with covariance from IRLS/normal equations where unpenalised, explicit note when regularised/bounded; Quantile with `tau` + SE; Huber with scale; MARS with basis terms; k-NN with neighbour table; Survival with curve table + median; Softmax; Heteroscedastic).
   - **Conformance test:** fits every accepted `(leaf, family)` combination and fails on any stub string (e.g., `"summary covariance not computed"`, `"Mean leaf model"` for non-mean leaves).
   - Fix `.pyi` stub drift noted in `specs/35` (`leaf_stats` and non-existent method stub).
2. **Issue `#25` — `ClusterModel` & `AnomalyModel` `.report()` and `.summary()` (`specs/35` Part C, `specs/25`):**
   - `ClusterModel`: cluster profile table (size, share, per-feature mean/spread/missing, prototype) + validity, stability, balance, and noise rate; meaningful `rules()` for every engine (or explicit message when unsupported).
   - `AnomalyModel`: top-cases review queue (rank, ID, score, threshold, percentile, leading features) + keep GritBot case summaries.
3. **Issue `#26` — Causal linear leaf (`specs/35` Part D, `specs/09`):**
   - Implement covariate-adjusted / treatment-interaction linear model inside causal leaves when `leaf="linear"` under `gain="causal"`, preserving honest sample splitting.
   - Update `docs/guides/uplift.md` once `leaf="linear"` differs from `leaf="mean"`.
4. **Issue `#23` completion — `Dataset` input on unsupervised facades (`python/mltreelib/cluster.py`, `anomaly.py`, `components.py`):**
   - Accept `Dataset` in `fit` / `predict` / `score_samples` / `transform`.
   - Carry over `#30`–`#33` input-boundary guarantees (column-width checks, `polars`/`pyarrow` via `to_rows`, 1-D errors) so every model facade shares one contract.

### Tier B — High-Synergy Additions (Recommended to Club into PR #2)
1. **Spec 34 Cross-Dataset Results Pages (`specs/34`):**
   - Build one reproducible survival/competing-risks benchmark script + guide section (`flchain`, `pbc`, `mgus2`, `gbsg2`, `veteran`, `colon`, `rott2`) and one uplift script + guide section (`hillstrom_email` subsampled, `nsw`, `jobs_ii` with non-compliance `assignment`/`receipt`).
   - Show tree (using `#24` survival summaries & `#26` causal linear leaf) vs no-tree vs linear vs ensemble, with repeated splits + bootstrap CIs + per-dataset failure modes.
2. **Item 10 Part A — Scale-Free `cp` Default Search & Elo Upgrade (`specs/33`, `specs/01`):**
   - Because changing defaults (`cp` scale-free pruning) is a visible behaviour change (per `specs/33` Decisions Recorded), landing it with Item 6 avoids two back-to-back behaviour-changing minor releases.
   - Upgrade Elo arena (`benchmarks/arena/`) with Bradley-Terry CIs, XGBoost, tuned peers, KNN anchor; confirm new `cp` defaults on the held-out Spec 34 survival/uplift suite.
3. **Small Cleanups Worth Bundling:**
   - **Issue `#19`:** Document (and decide rename of) `CatBoostEncoder` prior parameter (`specs/35` progress table).
   - **Guides update (`specs/35`):** Demonstrate `ModelTree.summary()` and `ModelTree.report()` across regression, classification, survival, clustering, and anomaly guides with real captured output.

### Tier C — Optional Speed Prerequisite (Only If Time/Scope Permits)
- **`specs/27` Phase 0–1 (Group metadata + Binned histogram splits for regression criteria):**
  - Needed before `specs/28` (`ModelForest`), `specs/29` (`ModelBoost`), and `specs/30` (`ModelMeta`).
  - Keep **Specs 28–30 (Ensembles)** and **Specs 22–24 / 11 / 16 / drift API** out of PR #2 — save them for PR #3 ("Ensembles").

---

## 4. Key Reference Paths (Do Not Duplicate)

- Workflow & rules: `AGENTS.md`, `contributing/definition-of-done.md`, `contributing/api-conventions.md`, `contributing/testing.md`
- Specs: `specs/README.md`, `specs/35-simple-object-interfaces-and-family-summaries.md`, `specs/34-survival-and-uplift-real-data-suite.md`, `specs/33-api-consistency-defaults-and-docs-contract.md`, `specs/27-regression-model-expansion.md`
- Handoff history: `references/research/2026-11-26-session-handoff.md`

---

## 5. Suggested Skills for Next Session

Call `read_file` on these `SKILL.md` paths (or invoke via skill protocol) at the start of the next session:
- `task-observer` (`/mnt/d/Programs/mltreelib/.agents/skills/task-observer/SKILL.md`) — mandatory first-batch scan per `AGENTS.md`.
- `code-review` (`/mnt/d/Programs/mltreelib/.agents/skills/code-review/SKILL.md`) — to audit PR #1 (`main..HEAD`) before pushing/opening the PR.
- `verify-and-stop` (`/home/jkapila/.agents/skills/verify-and-stop/SKILL.md`) — when running final PR #1 validation gates.
- `lean-build` (`/home/jkapila/.agents/skills/lean-build/SKILL.md`) — when implementing Item 6 (`#24`–`#26`) in PR #2 to keep scope tight against the spec.
- `tdd` (`/mnt/d/Programs/mltreelib/.agents/skills/tdd/SKILL.md`) — for writing the `#24` conformance test and `#26` causal linear leaf tests first.

---

## 6. Decisions made on 2026-11-26 (carry forward)

- **Per-family defaults.** `ModelTree` 6 / 15 / 0.5; `ClusterModel`, `RuleModel` and `AnomalyModel` `min_rows` 20; `AnomalyModel` `max_depth` 10. The parity test covers `ModelTree` (Rust, constructor, `LeafControl`); `tests/test_sanity_basics.py` holds the per-family table. Docs must say where a family differs (the anomaly page does not yet).
- **`MetricsReport` missing values.** It evaluates predictions after the fact, so it scores complete rows and shows `n_missing` (`on_missing="drop"`, default); `on_missing="error"` is the strict mode.
- **`predict` with fewer columns than `fit`:** warn and name the columns (knob `on_missing_columns="warn"|"error"|"allow"`); more columns raises.
- **`y` with None or NaN in `fit`:** rows not trained on, warn with the count (`y_missing="warn"|"error"`). Training on missing `y` is a feature request.
- **`id_col`, `group_col`, `protected_attribute`:** unresolved names raise `ValueError`; `predict` and `leaf_samples` accept the full-width or the id-excluded layout.
- WOE sign convention stays (ln event share over non-event share). **Every leaf family needs a summary table**, with a conformance test. Completeness and reliability before new capability. "Tree as a leaf" is research only; the unnamed pandas package is parked. `cvxpy` is test-only (`uv run --with cvxpy`). `pygraphviz` is docs-group only.

## 7. Still open after PR 1 (all recorded in Spec 35 registers)

Low severity, one small PR or bundled into PR 2:
- `leaf_samples(x_without_id, id_col=0)` without `col_names` silently uses a feature as ids (should raise or warn).
- `protected_col` given a name raises `TypeError` (takes an index): accept a name or raise `ValueError`.
- `docs/concepts/components-and-rotations.md` ~219: the block says "API gap", but `AnomalyModel(kind="pca")` is accepted and `method=` is silently swallowed by `**params` (that keyword catch-all is itself a trap: unknown keywords should raise).
- One-shot `Metrics()` left in `docs/guides/metrics-plots.md:124` (`stability_index`); decide whether a `drift` task in `MetricsReport` replaces it.
- Ruff on `tests`, `scripts`, `examples`: 393 errors, outside the pre-commit scope; baseline or clean up.
- Not verified: online loading of `pbc`, `colon`, `jobs_ii`, `nsw` with an empty cache; polars string-column hint tested with pandas only. Run once with `MLTREELIB_NETWORK_TESTS=1` and an empty `MLTREELIB_DATA_DIR`.
- Open question for the maintainer: string class labels. Today they raise "encode labels to integers"; support them (keep the label map, return labels from `predict`) or keep the error.
- Do the survival tasks need a threshold and cost like uplift (a horizon plus a risk cut for a policy table)? Not decided.

## 8. Further candidate specs for a strong next release (added 2026-11-26; also in `specs/README.md`)

Ranked by reliability payoff first, capability second. All are existing specs or index gaps, so each needs an issue first.

1. **Missing data in unsupervised models** (index gap under Spec 22): the README notes some clustering engines coerce `None` to 0.0, against the hard rule. Check every engine of `ClusterModel`, `AnomalyModel`, `ComponentModel` with missing cells and either route natively or raise. Today only "no crash" is checked, not the semantics. Rides naturally with `Dataset` on the facades.
2. **Save, load and model card for every model** (Spec 20 follow-up): only the tree has a runnable example. Add round-trip tests (byte-identical) per family and facade, a runnable guide, and a small model card (data schema, defaults, version).
3. **Calibration, fairness and model-compare as `MetricsReport` tasks** (Specs 12, 19, 25): primitives exist (D-calibration, a fairness criterion, the arena); no cross-task report. Fits the "metrics" third of the package.
4. **Drift monitoring** (no spec): PSI/CSI and score-KS primitives exist (`Metrics().stability_index`). A `drift` task in `MetricsReport` plus a short spec. Replaces the one-shot `Metrics()` left in the docs.
5. **Spec 31 selection guide, phase 1** (docs only): which family for which data. Cheap, and it makes the single-tree-first position explicit.
6. **Spec housekeeping:** Specs 25, 26 headers say proposed (index says completed); Spec 03, 20 and 34 headers are stale; Spec 27 is marked proposed although phases 0 to 7 shipped in 0.4.15; Spec 33 and 35 need "partly landed".
7. **Keyword catch-all audit:** `**params` swallows unknown keywords (`AnomalyModel(method="pca")` is accepted). Make unknown keywords raise on every facade; add the check to the sanity matrix.
8. Bigger, for PR 3: **Spec 28 `ModelForest`** is the largest measured win (`ModelTree(all)` regression Elo 716 vs ranger 1701), after Spec 27 phase 1; Specs 29 and 30 follow. Keep out of PR 2.

## 9. How to run verification cheaply (what worked today)

- Haiku verifier with a written brief, report-only, one subprocess per probe, `timeout 60`, and record each exit code itself (exit 137 or a timeout is a FAIL, never "no output"). Review every claim before relaying it: two of today's flags (`protected_col`, the stale components note) needed a second look.
- The sanity matrix lives in `tests/test_sanity_basics.py`; extend it with each new bug (it already covers defaults per family, `on_missing`, edge sizes, determinism, save/load, tables).
- Real `polars` and `pyarrow` only through `uv run --with polars --with pyarrow pytest ...`.

## Rules to remember (all in AGENTS.md and contributing/)

- No scikit-learn ever; shrink dependencies; build inside the package. Data helpers never impute, drop, recode or reorder missing values. Defaults identical on every surface. Reproducible to about 12 digits; compare with relative tolerance, never string equality.
- Done means code, tests and the existing docs and examples that show it, with generated output; no parallel new pages; explain each snippet in plain words; helper code belongs in the library, not the guide.
- Issue first for features; bugs get an issue and a `Fixes #N` commit. Verify commits with `git log`, not a filtered commit output (pre-commit runs ruff, ty and clippy repo-wide).
- Briefing Haiku agents: lowest-cost capable agent, disjoint files, the mandatory brief in `contributing/agents.md`; review every report; stop an agent that re-sends its report. Read-only agents must not run scripts that write tracked files.

## Practical notes

- `CLAUDE.md` is gitignored in this repo; the binding rules are in `AGENTS.md`.
- GitHub's GraphQL API was flaky: retry `gh issue create` up to four times and check `gh issue list` before retrying to avoid duplicates.
- Task Observer workspace is `$HOME/.claude`; the scan lists a due starter-set reconciliation and observations 13 to 14 (commit verification, missed session-start scan) are open; 15 is actioned.
- Examples needing extra packages run with `uv run --with <package> python <script>` (see the gallery's Run column).
