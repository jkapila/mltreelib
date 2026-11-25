# Session handoff, 2026-11-26

Read this first in the next session, then `AGENTS.md` and `contributing/definition-of-done.md`. Rules live in the repository, not in a session's memory.

## State

- **Branch:** `spec-33-34/api-consistency-and-data-suite`, about 17 commits ahead of `main` (`cad208a`). **Not pushed, no pull request yet, on purpose:** the maintainer decided that the docs rework and the open bugs (issues 24 to 27) are fixed first, then one pull request and one new version, so the picture is complete before review. Working tree clean at the time of writing.
- **Checks last run green:** `uv run pytest tests -q` (729 passed, 1 skipped), `cargo test -p mltreelib-core --release`, `scripts/docs/run_snippets.py --check`, `check_nav.py`, `check_benchmarks.py`, `uv run zensical build`, 0 broken links, sitemap 67 URLs. Re-run them before trusting this note.
- **Docs site:** `main` is deployed; the site was never submitted to Google Search Console, so structure can still change freely. This branch is not deployed.
- **After any Rust change** run `uv run maturin develop --release` before pytest; `uv sync --group dev --group docs` needs system Graphviz (`graphviz libgraphviz-dev`).

## What landed on the branch (and the GitHub issue each fixes)

| Issue | What |
|---|---|
| 16, 20 | Docs rewritten to run on real output (snippet runner `<!-- run -->`); one set of defaults on every surface (depth 6, min rows 15, ridge 0.5) with a parity test; saved models carry no timings; reproducibility statement (about 12 significant digits) |
| 17 | `fit`/`predict` accept pandas, Arrow-style, dict-of-columns tables by duck typing (`_tables.py`); pandas NA/NaT and infinities are treated as missing at the model boundary |
| 18 | Deterministic mob/ctree categorical gate statistics (ordered maps) |
| 19 | CatBoostEncoder docs example fixed (prior multiplies the global mean) |
| 21 | `mltreelib.data` package: profile, split, association, binning (greedy and exact dynamic programme), scaling, accessors |
| 22 | `MetricsReport(task, ...)` one-shot printable evaluation |
| 23 | `Dataset` class; `ModelTree.fit(dataset)` and `predict(dataset)` |
| (none) | Scale-free `cp` pruning (`prune="cp", cp=0.01`), docs group with pygraphviz, docs restructure (Concepts, Blog, Guides, Examples, Benchmarks, Reference), numbered steps with Next lines, README landing page, specs 33 to 35, contributor guides |

## Open issues

24 per-family summaries and `ModelTree.report()`; 25 cluster and anomaly report and summary; 26 causal linear leaf (`leaf="linear"` equals `leaf="mean"` under the causal gain); 27 `Metrics` NaN hangs, panics and silent recodes (`MetricsReport` refuses missing values outside regression and label classification until fixed).

## Next steps, in order

1. **Uplift policy table** in `MetricsReport` (selected group, everyone, no one, each with a 95% interval) plus a library helper that picks the threshold on validation data; uplift-by-decile table. The uplift guide currently hand-codes this in a 15-line function.
2. **Rework the regression and uplift guides** on `Dataset` and `MetricsReport`: regression at depth 4 (linear leaves lower RMSE about 5% there; depth 2 hides it), print `summary()` and one `leaf_summary(0)`; plain-language explanation before each snippet and what to read after it; state that the causal leaf is a difference of means whatever the leaf setting.
3. **Move the remaining ~40 pages and scripts** (about 124 lines) from the helper functions to `Dataset` and `MetricsReport` in one pass; the lists are in the agent reports of 2026-11-25 (search for `xy(`, `select_rows(`, `train_test_split(` and `Metrics()`).
4. Fix `Metrics` missing-value handling (27), then drop the refusals in `MetricsReport`.
5. Per-family summaries and cluster/anomaly reports (24, 25), causal linear leaf (26).
6. Let `ClusterModel`, `AnomalyModel`, `ComponentModel` accept a `Dataset`.
7. **Decided but not yet done:** repair the dead SurvSet registry entries (OpenML ids 44101, 44103 to 44107 return 404; 43840 is a different dataset) by repointing to the Rdatasets mirror, keeping the subset description as secondary detail. Remaining Spec 34 items: core survival and uplift dataset suite (GBSG2, PBC, mgus2, NSW, JOBS II, one subsampled large uplift set) with checksums and licence notes, normalised loaders, offline mode.
8. **Elo upgrade** (`benchmarks/arena/elo_bt.py` exists; published Elo pages still show the old sequential ratings and must not show "rank 9"). Plan: fix the protocol first (splits, repeats from a variance pilot, budget, KNN as the anchor), add XGBoost and tuned peers, separate default and tuned leagues, add a single-tree league, dataset manifest with checksums, then switch the pages. Then the better-defaults search (default pruning is a no-op on dollar-scale targets, so make `cp` the default scale-free), confirmed on held-out datasets (CTR23 and CC18, deduplicated by name).
9. Backlog in Spec 33 section F: stale figures, typed tables with no run block, `hpo_shapes.py` never run end to end, other API reference pages stale against the generator, `to_json` timing check.
10. Decisions made on 2026-11-26: no pull request or version until the docs rework and bugs are done; `pygraphviz` stays in the `docs` group only (not `dev`); `cvxpy` is an optional extra (`pip install "mltreelib[cvx]"`) used only by the three exact reference solvers in `solvers.py` (`solve_sparse_rules_cvx`, `solve_knapsack_pruning_cvx`, `solve_oct_mip_cvx`), which cross-check the native Rust solvers; `DatasetBuilder` opt-in drop/impute stays, default allow. Still unanswered from the maintainer: which sign convention for weight of evidence (the code uses ln(event share / non-event share), opposite of many scorecard tools), which model family summary the maintainer once asked for (the stub says only Gaussian linear has a table), the name of the pandas data-preparation package the maintainer used before.

## Decisions added late on 2026-11-26

- WOE sign convention stays as shipped (ln event share over non-event share); documented.
- **Every leaf family needs a summary table**, no exception (mean, median, linear and parametric families, quantile, Huber, MARS, k-NN, survival, Tobit, softmax, heteroscedastic, anything added later). Spec 35 now requires a conformance test over all accepted leaf and family combinations that fails on a stub. Issue 24 carries it.
- Completeness and reliability before new capability: no new analysis tooling now (the unnamed pandas package the maintainer once used is parked).
- Research idea recorded in `specs/README.md`: a tree as the leaf model of a tree (each leaf fits its own tree on its rows). Not scheduled.

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
