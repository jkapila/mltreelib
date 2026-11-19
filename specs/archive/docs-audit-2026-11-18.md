# mltreelib documentation audit, 2026-11-18

## Summary

1. The README has about 10 API calls that do not exist (`leaf_models`, `export_rules`, `predict_with_neighbors`, `predict_uplift`, `ExtendedIsolationTree.score_samples`, `GritBot(min_cases=...)`, `detect_anomalies`, and several Rule attributes). Users who copy it will get `AttributeError` or `TypeError`.
2. The version is inconsistent: `python/mltreelib/__init__.py` says `0.4.5`, while pyproject, Cargo and PyPI say `0.4.8`.
3. The install page claims "one wheel, no transitive dependencies", but the package requires `pyarrow>=14`. The quickstart output is stale, and the live site still says `uv sync` and "51 datasets".
4. The model scope is clear in code. Survival, uplift, Tobit, Tweedie and heteroscedastic leaves all exist and have guides. Regression discontinuity, panel models and instrumental variables do not exist anywhere in code or docs.
5. The published site has no broken nav entries and PyPI has 0.4.8. Problems are internal `agents/` pages published by accident, two broken image links, two stub pages and missing project-level pages (changelog, contributing, citation, licence).

## Method and limits

- Read: `README.md`, `pyproject.toml`, `Cargo.toml`, `mkdocs.yml`, `.github/workflows/{deploy,test,release}.yaml`, `docs/**` (all pages), `python/mltreelib/*.py`, `_core.pyi`, and the PyO3 bindings `crates/mltreelib-py/src/lib.rs`.
- Ran against the existing `.venv` (Python 3.12, `_core.cpython-312` built 2026-11-18 21:46, newer than `lib.rs` at 19:51). No Rust compile was done.
- Ran the README code blocks (`/tmp/claude-1000/.../scratchpad/readme_run.py`), the docs quickstart block (`docs/quickstart.md`), `examples/quickstart.py`, and a fit test of every documented `gain` and `leaf_family` string.
- Fetched `https://jkapila.github.io/mltreelib/` and subpages, and `https://pypi.org/pypi/mltreelib/json`, via WebFetch.
- Not checked: the other 60+ example scripts, the R package (not in repo), the benchmark CSVs, and the deploy commit that the live site was built from. The live site cannot be mapped to a commit from outside.
- Git: 8 files in `docs/guides/` and `examples/real_survival_uplift.py` have uncommitted edits, and `examples/real_tree_studies.py` is untracked. Findings on those files may already be fixed locally.
- Nothing in the repo was edited. The only file written is this report.

## Models: exist in code vs documented

| Workflow / model | In code (Python API) | Documented | Notes |
|---|---|---|---|
| Regression: linear / mean / median / quantile / huber leaves (`ModelTree`) | Yes | Yes (`guides/regression.md`, README) | Core |
| Classification: binomial logit, probit, softmax (`leaf_family`) | Yes | Yes (`guides/classification.md`) | |
| Counts and GLM-type: Poisson, Gamma, Tweedie | Yes (`leaf_family`) | Partly (blog, `advanced-modeling.md`) | No dedicated guide |
| Censored: Tobit (`leaf_family="tobit"`, `gain="tobit"`) | Yes | Yes (`advanced-modeling.md`, blog, `examples/tobit-regression.md`) | |
| Heteroscedastic / distributional leaves | Yes | Yes (`examples/distributional-leaves.md`) | |
| Survival: KM / Nelson-Aalen leaves, `predict(kind="survival"/"hazard")`, `gain="logrank"`, `"martingale"` | Yes | Yes (`guides/survival.md`, `examples/survival-trees.md`) | Guide has no assumption section (censoring) |
| Competing risks | No | No | `specs/10-survival-and-competing-risks-trees.md` only |
| Cox proportional hazards | No | Mentioned in one example comment only | Not a tree model |
| Uplift / causal trees: `gain="causal"`/`"uplift"`, `fit(X, (y, t))`, `predict(kind="uplift")` | Yes | Yes (`guides/uplift.md`, blog) | README uses `predict_uplift`, which does not exist |
| Honest trees (`honesty=True`) | Yes (`LeafControl`/`ModelTree`) | Yes (`reference/model.md`, `examples/honesty-weights.md`) | |
| Regression discontinuity (RDD) | No | No | Not in README, docs, code or roadmap |
| Panel regression (fixed effects, DiD) | No | No | `docs/` "panel" hits are plotting or time-series only |
| Panel VAR / VECM trees | No | No | `specs/17-panel-var-vecm-trees.md` only |
| Instrumental variables | No | No | |
| Synthetic control | No | `examples/two-controls-and-0d.md` (a synthetic-control style example) | Not an estimator class |
| Cubist (linear leaves + neighbour adjustment) | Yes (`predict_adjusted`, `extract_rules`) | Yes (`examples/rules-summary-knn.md`) | README names are wrong |
| C5.0-style rules (`RuleSet`, `Rule`) | Yes (via `extract_rules`) | Yes | README attribute names are wrong |
| MARS (`MarsRegressor`, and `leaf="mars"`) | Yes | Not verified (no dedicated page found in nav) | Grep `docs/` for `Mars` before claiming coverage |
| Clustering: HDBSCAN, URF, CLTree, PCT, TrieKNN | Yes | Yes (`guides/clustering.md`, `reference/api/cluster.md`) | 12 classes have no Rust doc comment |
| Anomaly: DET, EIF, HST, RCT, AAD, CAT, GritBot (+ `AnomalyTree` facade) | Yes | Yes (`guides/anomaly.md`, `reference/api/anomaly.md`) | README names are wrong |
| Encoders | 14 classes | Docs say 18, README says 12 | See finding H5 |
| Synthetic generator `generate()` | Yes | Yes | |

Verdict: the code covers survival, uplift, censored/distributional, clustering, anomaly and rule-based workflows. RDD, panel and IV are absent. The owner's goal (one package for many econometric workflows) needs a scope page that says which of those are planned, not implied.

## Findings, ranked

### Critical

**C1. README API examples call methods that do not exist.** Verified by running the blocks against the installed build.
- `README.md:86` `model.leaf_models()` → not present. Real: `ModelTree.leaves()` / `leaf_stats()` / `leaf_summary(id)`.
- `README.md:98` `model.export_rules(prune=True, confidence=0.25)` → `AttributeError`. Real: `extract_rules(x=None, y=None, prune=True, cf=0.25)`. This line is also in `README.md:44`.
- `README.md:107` `predict_with_neighbors(X, k=5, neighbors_weight=0.5)` → `AttributeError`. Real: `predict_adjusted(x, neighbors=5)`. No `neighbors_weight` argument exists. Also `README.md:45`.
- `README.md:138` `causal_tree.predict_uplift(X_new)` → `AttributeError`. Real: `predict(X, kind="uplift")`.
- `README.md:100-103` `ruleset.n_rules`, `rule.antecedent`, `rule.consequent`, `rule.error_rate` → not present. Real `Rule` attributes: `conditions`, `confidence`, `errors`, `cover`, `prediction`, `id`, `leaf_id`; `RuleSet` has `rules`, `default_prediction`, `default_confidence`.
- `README.md:175` `ExtendedIsolationTree.score_samples` → `AttributeError`. Real: `anomaly_score(x)`. (`score_samples` exists only on the Python facade `AnomalyTree`.)
- `README.md:178-180` `GritBot(min_cases=10, filter_percent=1.0)` → `TypeError`; `grit.detect_anomalies` → not present. Real constructor: `min_items`, `max_norm`, `sig_threshold`, `max_depth`, `check_cols`; method: `scan`.
- Fix: rewrite the README section 2-5 blocks from the names listed above, and add a CI job that runs README and docs code blocks (for example with `pytest --doctest-md` or a small `exec` test). Doing so would have caught this.

**C2. Version mismatch.** `python/mltreelib/__init__.py` (last line) sets `__version__ = "0.4.5"`. `pyproject.toml:5`, `Cargo.toml` (`[workspace.package]`) and PyPI all say `0.4.8`. Runtime `mltreelib.__version__` therefore reports `0.4.5`. Fix: read `importlib.metadata.version("mltreelib")` or bump the constant in the same release commit, and add a test that compares it to `pyproject.toml`.

### High

**H1. Install page is factually wrong about dependencies.** `docs/installation.md:3-4` says "one wheel, no transitive dependencies". `pyproject.toml:13` lists `dependencies = ["pyarrow>=14"]`, and the installed metadata shows it. Fix: "one wheel; pulls `pyarrow` (required)". Also check the `pip install` line is still correct (PyPI 0.4.8 exists, so it is).

**H2. Quickstart output is stale.** `docs/quickstart.md` shows a hard-coded output block. Running the same code today gives predictions `[0.25, 2.09, 4.21, 6.15, 8.09, 15.17, 17.14, 19.11]` (docs: `[0.16, 2.18, 4.17, 6.17, 8.12, 15.12, 17.14, 19.16]`), RMSE `0.161` (docs `0.153`), and a 5-node tree (docs show 10 nodes, with split `x0<2`). Fix: generate the output from the code (`mkdocs`/`zensical` snippet include of a script run), or delete the printed output and say what to look for.

**H3. Dataset count conflicts.** `docs/index.md:12` and the live home page say "51 TabArena-v0.1 benchmark datasets". `README.md` (TabArena section) says "18 benchmark datasets" for regression and "24" for classification. Fix: pick one number with its source (`datasets/MANIFEST.md` or `benchmarks/`) and use it in index, README and benchmark pages.

**H4. Encoder count conflicts.** `docs/installation.md` "What you get" says "18 encoders". `README.md` says "12". The bindings expose 14 encoder classes (`OneHot, Hashing, Target, CatBoost, Tfidf, WoE, Embedding, Count, LeaveOneOut, MEstimator, BaseN, Sum, Quantile, JamesStein`; `crates/mltreelib-py/src/lib.rs`). Fix: use 14 (or the exact count after review).

**H5. Internal agent pages are published.** `docs/agents/{docs-generation,domain,issue-tracker,triage-labels}.md` are not in the nav but are built into `site/agents/` and live at `https://jkapila.github.io/mltreelib/agents/domain/` (confirmed by WebFetch: "How the engineering skills should consume this repo's domain documentation"). Fix: move these out of `docs/` (for example to `.agents/` or `specs/`), or add them to `exclude_docs`.

**H6. Model-tree gain docstring is incomplete.** `crates/mltreelib-py/src/lib.rs:2888-2894` lists gains `sse, deviance, friedman_mse, absolute_error, huber, poisson, entropy, gini, gain_ratio, sdr, twoing, callable`. Fitting all of these, plus `tweedie, heteroscedastic, tobit, cv, logrank, martingale, causal, uplift`, succeeds. The README and the survival/uplift/distributional docs use them, but the signature docs do not. Fix: add all accepted gains to the docstring and to `docs/reference/api/model-tree.md`.

**H7. Stub type file misleads.** `python/mltreelib/_core.pyi` types most classes as `_BaseExtensionClass` with `__getattr__ -> Any`. For `ModelTree` it lists 13 methods, but the real class has 28 (missing `leaves` variants, `profile`, `shapley`, `break_down`, `permutation_importance`, `predict_adjusted`, `extract_rules`, `retrain`, `quantile`, `cdf`, `leaf_samples`, `apply`, and others). Fix: generate the stub with `pyo3-stubgen` or hand-complete it, and list the true signatures.

### Medium

**M1. Broken image links.** `docs/examples/data-processing.md:67` and `:114` use `assets/variance_shares.png` and `assets/decomposition.png`. From `docs/examples/`, these resolve to `docs/examples/assets/`, which does not exist. Fix: `../assets/...`.

**M2. Stub pages outside the nav.** `docs/examples/visualization.md` (13 words; its live page says "moved to Visualize data...") and `docs/reference/plots.md` (13 words). Neither is in `mkdocs.yml` nav. Fix: delete both, or add a redirect with `mkdocs-redirects`.

**M3. Uplift theory is incomplete.** `docs/guides/uplift.md` cites Athey & Imbens (2016) but never states the identifying assumptions. Uses "appropriate treatment assignment" and "A leaf comparison alone does not remove confounding" but does not name unconfoundedness or overlap as conditions. Survival guide (`docs/guides/survival.md`) has no assumption list for independent censoring. Fix: add a short "Assumptions" section to each (randomized or unconfounded treatment, overlap, non-informative censoring, proportional-hazard-free leaves), and note what the tree does not guarantee.

**M4. No project-level pages for a new user.** Absent: `CHANGELOG`, `CONTRIBUTING`, `CITATION.cff` (or a BibTeX block), a docs licence page (`LICENSE` exists at repo root, Apache-2.0), and a scope page. `docs/index.md:66` sends contributors to the repo's README and AGENTS.md. Fix: add `changelog.md` (seed from the commit log), `contributing.md`, a `citation` section, and a `scope.md` that lists supported workflows, planned ones (RDD, panel, from `specs/`) and non-goals.

**M5. Docstrings: 12 of 36 PyO3 classes have no doc comment.** `crates/mltreelib-py/src/lib.rs` `#[pyclass]` without `///` above it: `GritBot, DensityEstimationTree, ExtendedIsolationTree, HalfSpaceTree, RandomCutTree, ActiveAnomalyTree, ContextAnomalyTree, TrieKNN, PredictiveClusteringTree, CLTree, HDBSCAN, UnsupervisedRandomForest`. These are the anomaly and clustering families, and they are exactly where the README is wrong. Fix: add a docstring to each with parameters, returns and one example.

**M6. Install page versus pyproject classifier list.** README badge lists Python 3.10 to 3.15; CI matrix is 3.10 to 3.14; pyproject classifiers stop at 3.13. Fix: make these agree.

**M7. Broken guide link check not repeated.** `docs/guides/index.md:17` links `../examples/index.md`. That file exists now, but it is not in the nav, and it was not in my first directory listing, so the tree changed during the audit. Re-run the nav and link check after the uncommitted edits land.

### Low

**L1. Docs toolchain is ambiguous.** Deploy (`.github/workflows/deploy.yaml`) runs `zensical build`. `mkdocs.yml` is the only config, and it uses mkdocs-only plugins (`llmstxt`, `minify`, `include-markdown`). `site/llms.txt` exists, so Zensical reads this file, but the README and `pyproject` (`dev` group has zensical, no mkdocs) do not say so. Fix: a one-line note in `CONTRIBUTING` on how to build docs locally.

**L2. Mixed "R frontend" language.** `pyproject.toml:4` description and README say "Python and R frontends" / "R bindings underway". No R package is in the repo, and the roadmap says "R Frontend Package: CRAN package ... [ ]". Fix: say "planned".

**L3. Blog and index framing.** `docs/index.md` lists "Featured Technical Articles" that include a credit-scoring case study with "Regulatory Compliance" framing. Fine for a blog, but there is no disclaimer that the examples use synthetic or public data. Low risk.

## Installation check

- `pip install mltreelib` works: PyPI JSON shows `info.version = "0.4.8"`.
- `release.yaml` builds Linux (x86_64, aarch64), macOS (arm64, x86_64), Windows (x86_64) and sdist, and publishes on `v*` tags. So wheel coverage is correct per config. I did not verify the wheel list on PyPI because the JSON response was truncated.
- Source build instructions: `uv sync` then `uv run maturin develop --release` (README) and `uv run maturin develop` (`docs/installation.md`). Both are valid for a maturin + PyO3 project with `manifest-path` and `module-name = "mltreelib._core"` in `pyproject.toml`. The `--release` flag is missing from the installation page, so the two differ.
- Unfitted print behaviour: `docs/installation.md` says `print(ModelTree())` raises `ValueError: ModelTree is not fitted`. This matches the code (the constructor succeeds, `__repr__` raises).

## Live site vs repo

- Nav: all 66 nav entries resolve to files on disk. No nav entry is missing.
- Pages present on disk and absent from nav: `docs/agents/*` (4), `docs/examples/visualization.md`, `docs/reference/plots.md`.
- Live pages checked: home (matches `docs/index.md` text including the 51-dataset claim and `uv sync` install), `guides/uplift/` (matches local), `examples/visualization/` (matches the stub), `agents/domain/` (published, see H5).
- Not verified: whether the live build is current with the last push. The live home shows no version. Local `git status` has 8 modified guides and 1 untracked example that the live site may not reflect.
- Broken internal links in `docs/`: 2 (M1). External links were not checked.

## Quickstart and example results

| Block | Result |
|---|---|
| `docs/quickstart.md` code | Runs. Output differs from the documented block (H2). |
| `examples/quickstart.py` | Runs, exit 0. |
| README 1 (`ModelTree(...)` fit, predict, quantile, summary) | Runs. |
| README 1 `leaf_models()` | `AttributeError` (C1). |
| README 2 `extract_rules(prune, cf)` | Runs. README's `export_rules` fails (C1). |
| README 2 `predict_adjusted` | Runs. README's `predict_with_neighbors` fails (C1). |
| README 3 `gain="tweedie"`, `"heteroscedastic"`, `"causal"`, `"tobit"`, `"cv"`, `"logrank"`, `"martingale"` | All accepted (fit succeeds). Docs correct; signature docstring incomplete (H6). |
| README 3 `predict_uplift` | `AttributeError` (C1). `predict(kind="uplift")` works. |
| README 4 `HDBSCAN(min_cluster_size, min_samples).fit`, `UnsupervisedRandomForest(...).fit(X, n_clusters=4)` | Run. |
| README 5 `ExtendedIsolationTree.score_samples` | `AttributeError` (C1). |
| README 5 `GritBot(min_cases, filter_percent)` | `TypeError` (C1). |
| README 6 `generate(... tail_behavior="bathtub" ...)` | Runs. Returns `columns, target, labels, truth`. |

Not run: the examples under `examples/` other than `quickstart.py`, which need datasets or long fits. The Rust compile was skipped, so the Rust-side tests were not run.

## Top 5 fixes to do first

1. Rewrite the README API blocks (C1) to match the real names and signatures, and add a CI job that runs README and docs Python blocks.
2. Fix the version (C2): derive `__version__` from package metadata or bump it in the release commit.
3. Correct the install page's dependency claim (H1) and make the quickstart output generated, not hard-coded (H2).
4. Move `docs/agents/` out of the published docs tree (H5), and delete or redirect the two stub pages (M2).
5. Add the missing project pages and a scope page (M4), starting with a scope statement that says survival, uplift, Tobit and clustering are implemented, and RDD, panel and IV are not.
