# Spec 33: API Consistency, Tuned Defaults and the Docs Contract

Status: proposed

Scope boundary: this spec owns the public `ModelTree` parameter surface (the split, grow, prune and leaf knob model), the single source of truth for defaults, the data-driven choice of better defaults, and the contract that keeps reference documentation equal to the code. It does not add new model families (Specs 28 to 32) and does not change the Elo protocol (separate spec).

## Problem Statement

A documentation audit before the first public indexing found that the documented API and the real API disagree in ways a user cannot detect until a snippet fails.

1. **Three different defaults.** The live `ModelTree()` constructor, the Rust parameter struct and the reference page each state different values for depth and the minimum rows per split. A fourth set appears in a concepts page, together with a different pruning default. A user cannot tell what a bare `ModelTree()` does, and nobody can tune a default that is not defined in one place.
2. **Documented options that raise.** About fifteen documented option values are rejected at runtime, for example a gain named "variance", a split evaluation named "exact", a pruning method named "cost_complexity", a categorical order named "natural". Thirty constructor parameters are missing from the reference page.
3. **A pruning example that never worked.** A guide used a scikit-learn style `ccp_alpha`, which the library has never accepted. The real knobs are `prune` and `cp_alpha`, and `cp_alpha` is an absolute per-leaf penalty in squared-error units, so its effect depends on the target's scale. The page also claimed holdout cross-validation, which does not exist.
4. **Knobs in the wrong place or named twice.** `monotone`, `tree_monotone` and `leaf_monotone` coexist and the growth control rejects one of them. The ridge penalty defaults to one value on the model and a different one on the leaf control, so constructing a leaf control silently changes the fit. A named preset silently overrides explicit arguments. Several controls and aliases name the same concept.
5. **Documented behaviour that is not built.** `max_leaf_nodes` is described as best-first growth but is a depth-first cap. There are no multiway splits. Pruning has one alpha and no path, no cross-validated selection and no reduced-error or minimum-error variant.
6. **Defaults are untuned for competition.** The shipped defaults were chosen by small grids on a few datasets. The library is benchmarked against a broad field, and the default configuration is what most users and every default-versus-default comparison will see.

## Solution

One definition of every default, generated into the reference documentation; a knob model that matches how the maintainer thinks about a decision tree (split, grow, prune, leaf, plus standard knobs); a short list of corrected or removed options; and a better default configuration chosen by search on one group of datasets and confirmed on datasets that played no part in the search.

The user's view: `ModelTree()` does one documented thing, every option in the reference works, every pruning knob is explained in terms of what it does to the tree, and the default model is measurably better on data the search never saw.

## User Stories

1. As a data scientist, I want `ModelTree()` with no arguments to behave exactly as the documentation says, so that I can reason about my results.
2. As a data scientist, I want every option value in the reference page to be accepted by the constructor, so that copying from the docs never raises.
3. As a data scientist, I want unknown or removed parameters to raise an error that suggests the right name, so that typos are cheap.
4. As a maintainer, I want each default defined once, so that the code, the reference page and the guides cannot drift apart.
5. As a maintainer, I want the reference defaults table generated from the live constructor, so that a changed default shows up in the docs without manual editing.
6. As a maintainer, I want a test that fails when a documented option is rejected, so that a documentation edit cannot ship a broken example.
7. As a maintainer, I want a test that fails when a constructor parameter is undocumented, so that the reference stays complete.
8. As a user tuning pruning, I want a scale-free way to set pruning strength, so that the same value means the same thing on a target measured in dollars and one measured in grams.
9. As a user, I want the absolute per-leaf penalty to remain available and clearly described, so that existing models are reproducible.
10. As a user, I want to know whether pruning picked its strength by validation, so that I can trust or override it.
11. As a user, I want a cross-validated choice of pruning strength, so that I do not have to guess one.
12. As a user, I want a weakest-link pruning path, so that I can inspect how the tree shrinks as the penalty grows.
13. As a user, I want reduced-error pruning against a holdout, so that I can use the simple method business reviewers already know.
14. As a user, I want growth to stop at a maximum number of leaves by always splitting the best candidate first, so that a small leaf budget yields the most useful small tree.
15. As a user, I want the documentation to say plainly that splits are binary, so that I do not look for a multiway option that does not exist.
16. As a user, I want one name per concept for the monotone constraint, so that I know where to put it.
17. As a user, I want the same ridge default whether I pass a leaf control or not, so that adding a control does not change my model.
18. As a user, I want a preset that conflicts with my explicit arguments to raise, so that I notice the conflict.
19. As a user, I want minimum-sample knobs explained as standard knobs independent of growth and leaf, so that I know they exist whatever leaf I choose.
20. As a user, I want missing-value handling described in one section even though it spans two knobs, so that I can find it.
21. As a reviewer at a company, I want a single page that maps every knob to split, grow, prune, leaf or standard, so that I can audit what a model does.
22. As a benchmark reader, I want the default model to be reported separately from a tuned model, so that I know which numbers need effort.
23. As a benchmark reader, I want the chosen default to be validated on datasets excluded from the search, so that the improvement is not benchmark overfitting.
24. As a maintainer, I want the default search to record every trial, so that the choice can be audited and rerun.
25. As a maintainer, I want exact-coefficient tests to pin their parameters explicitly, so that a default change never breaks them.
26. As an upgrading user, I want a changelog entry listing every default that changed and its old value, so that I can pin the old behaviour.
27. As an upgrading user, I want a way to request the previous defaults by name, so that old results stay reproducible.
28. As a maintainer, I want documentation snippets to use only options that the live constructor accepts, so that the docs build can fail on a bad example.
29. As a user, I want to pass a pandas or Arrow table straight to `fit`, `predict` and the other task models, with missing values in any form, so that I never hand-convert NaN to `None` before the model sees the data.
30. As a maintainer, I want that table input handled by duck typing at the model's input boundary, so that no pandas, Arrow or scikit-learn import is needed to run the library.

## Implementation Decisions

- **Knob model.** The public parameter surface is organised into split (how a node is divided and what the candidate cuts are), grow (when and in what order the tree grows), prune (how a grown tree is cut back), leaf (what is fitted in a terminal node) and standard (sample sizes, weights, seeds, honesty, missing-value routing, constraints). Every parameter belongs to exactly one group, and the reference page is organised by those groups. This matches the two-control architecture: growth control covers split, grow and prune; leaf control covers leaf; standard knobs belong to neither.
- **Single source of truth for defaults.** The constructor's defaults are the contract. The Rust core default struct is aligned to them, and the reference page's default column is generated from the live constructor at documentation build time. Hand-written default values in guides are replaced by references to the generated table or by snippets that print the real defaults.
- **Reference completeness gate.** The documentation build compares the constructor's parameter list and accepted option values with the reference page. Missing parameters, extra parameters and documented option values that the constructor rejects fail the build.
- **Option corrections.** Each documented option that raises is either made to work, if it names a capability that exists under a different spelling, or removed from the docs. Removals and renames are listed in the changelog. No alias is added without a reason, because aliases are what created the current duplication.
- **Pruning.** `prune` and `cp_alpha` keep their meaning. The docs state that `cp_alpha` is an absolute per-leaf penalty in squared-error or deviance units. A scale-free option is added, expressed relative to the root error, with the absolute form retained for reproducibility. Weakest-link path output and cross-validated selection are added as a second step. Reduced-error pruning against a holdout is added as a user-visible alternative. The false cross-validation claim in the docs is removed immediately, before any of these ship.
- **Growth order.** Either best-first growth is built so `max_leaf_nodes` does what the docs say, or the docs are corrected to describe a depth-first cap. The decision is made explicitly. Best-first is the recommended choice, because a small leaf budget is the main way to get a small, explainable tree.
- **Splits are binary.** The docs say so. Multiway splits are recorded as a possible future capability and are not part of this spec.
- **Placement fixes.** One canonical name is kept for the monotone constraint at the tree level and one at the leaf level, and the growth control accepts the tree-level one. The ridge default is identical on the model and on the leaf control. A preset that conflicts with an explicit argument raises. Treatment, event and cause inputs move from the constructor to fit-time arguments where practical, with a deprecation path.
- **Undocumented parameters.** Each of the thirty missing parameters is documented, or deleted if it has no users.
- **Better defaults, search design.** The search covers the knobs that matter for a single tree: depth, minimum rows, pruning strength, leaf type, ridge penalty, split precision, and the split-evaluation mode. Datasets are divided into a search group and a confirmation group with no dataset in both; datasets that appear in more than one benchmark suite under different identifiers are assigned to the same group by name, shape and checksum. The objective is the average normalized score or rank against the same reference models used by the benchmark, with a fit-time penalty so a slow configuration cannot win by a small margin. The search uses cross-validation inside each dataset and a fixed seed. Regression and classification may receive different defaults. A size-dependent default (for example, minimum rows as a function of row count) is considered as an alternative to a single constant.
- **Acceptance of new defaults.** A new default is adopted only if it improves the confirmation group by more than the benchmark's confidence interval, does not lose on the search group, and does not slow the fit time beyond a stated factor. The previous defaults remain reachable by name.
- **Input boundary.** The model accepts a column-oriented table from common sources (pandas, Arrow, Polars-style objects, dict of columns, NumPy, nested lists) without importing those libraries: detection is by behaviour, not by type. Missing values in any form (NaN, None, null, NA) are accepted as they are and routed natively. Today a pandas frame passed to `fit` raises `Can't extract str to Vec`, which is why snippets convert by hand. Missing data is never imputed or recoded by the library's data helpers; only the model's own handling (surrogate splits, learned missing direction) treats it.
- **Dependency policy.** No scikit-learn dependency, ever; the sklearn-shaped API is a convention only. The package aims to shrink its dependency set. Anything that can be built inside the library is built inside it.
- **Documentation contract.** Every parameter named in a guide must be accepted by the live constructor. The runnable-snippet gate already in the docs build is the enforcement mechanism for snippets; a separate check covers inline parameter names and option values.

## Testing Decisions

- A good test checks externally visible behaviour: what the constructor accepts, what it rejects, what a documented example prints. It does not inspect internals.
- The highest seam is the public constructor plus the reference documentation. One test enumerates the constructor's parameters and accepted values and compares them with the reference page. A second test runs a bare `ModelTree()` and asserts the documented defaults.
- Pruning behaviour is tested on small deterministic datasets: a split is kept only when its gain exceeds the penalty; the scale-free option is invariant when the target is multiplied by a constant; the pruning path is monotone in the penalty; reduced-error pruning never increases holdout error on the pruning set.
- Best-first growth, if built, is tested by comparing the tree at a leaf budget against the best tree reachable by exhaustive search on a small problem, and by checking that the budget is never exceeded.
- Default selection is tested by a reproducibility test: the search with a fixed seed on a fixed small dataset group returns the same configuration.
- Prior art: the existing parameter-validation tests, the runnable-snippet tests in the docs build, and the default-search benchmark harness.

## Audit Findings (verified)

The documentation pass printed results that looked like defects. Each was reproduced or traced; outcomes:

1. **Non-deterministic gate statistics: real, fixed (issue 18).** The categorical significance statistics for MOB and ctree varied in the last digits between identical runs, because floats were summed in the iteration order of a randomly seeded hash map. The level grouping now uses an ordered map, and a test fits the same model repeatedly and compares statistics exactly. Still open: `to_json` embeds wall-clock profiling fields, so serialised models are not byte-identical between runs; the reproducibility contract must either exclude or remove them.
2. **Target encoder range: not an encoder bug (issue 19).** `prior` multiplies the global mean and enters only the numerator, so for a 0/1 target the output lies in `[0, max(1, prior * mean)]`. The docs example copied `5.0` from a smoothing weight. The example is corrected; the parameter still needs documentation, a clearer name and an out-of-scale warning.
3. **Monotone constraint had no effect: not a bug.** The documented example data already satisfied the constraint, so the constrained tree equals the unconstrained one; the opposite direction changes the result. Follow-ups: the example should use data where the constraint binds, and one internal check on fitted leaf boundaries reads the flat `monotone` value instead of the effective tree-level direction (unverified impact; add a test that sets only the tree-level constraint).
4. **Unassigned points in density clustering: open.** CLTree left 232 of 980 points unassigned when only 80 were uniform noise. Not yet traced.
5. **Survival calibration failure: open.** D-calibration rejected the documented synthetic competing-risks cohort. Not yet traced.
6. **Oblique splits not chosen: open.** The oblique-split example picked an axis-aligned split. Not yet traced.
7. **Documented capabilities that do not exist: open.** Subspace-reconstruction anomaly scoring and a PCA method on the anomaly model are described in a guide but not implemented. Implement or remove the claims.
8. **Pandas input: real, fixed (issue 17).** A pandas frame passed to `fit` failed. Models now accept a list of rows, NumPy, a dict of columns, a data-frame-like object or an Arrow-style table, by duck typing and without importing those packages. Missing values are untouched; only the pandas NA and NaT singletons, which cannot be converted to numbers, become `None` at the input boundary.

## Documentation Stabilisation Backlog

Items left after the first pass of the documentation audit. They are worked one at a time, each checked by a person before the next, so that the site is stable before anything new is added. Rule for every item: what a page shows must be what the code prints with the same data and seed.

**A. Figures that embed old numbers or timings (regenerate from the real run, or remove):**
- Solver benchmark Pareto and optimization trajectory figures.
- Evaluation page figures: residuals, precision-recall, calibration, confusion, tree, importance.
- Regression real-versus-synthetic figure and the real diamonds leaves figure.
- Time-series decomposition figure.
- Constraints monotone-comparison figure.
- Core growth-knob grid figure.
- Variance-shares figure on the data-routing page.
- The missing-routing APS figure was removed from its page; decide whether to regenerate it.
- Rule: every figure is produced by a script that is named on the page and rerun when its numbers change.

**B. Page sections that still cannot run top to bottom:**
- Leaf retraining page: its only code block has no setup and prints nothing; its timing tables are unverified.
- Regression-methods benchmark page: the quantile-crossing block has no producer for its inputs; the full suite block is slow; the tables with fit times are unverified.
- Survival-trees page: the real-data narratives and the benchmark table are not reproduced by any block; the DOT export block writes files and is not opted in.
- Solvers page: the real-data tuning table with a wall-time column is not reproduced.
- Components page: the rotation, whitening and NMF tables and the cognitive-battery text are hand-written; the subspace anomaly block needs capabilities that do not exist.
- Distributional-leaves page: the usage block lacks setup; the real-dataset claims are unverified.
- Survival and uplift guides: any block that still depends on an earlier block's names.

**C. Prose that was only checked for running, not for claims:**
- Guides: data, visualization, model inspection, serving contract, advanced modeling, classification, clustering, anomaly.
- Each hand-written number or claim is either tied to printed output or removed.

**D. Benchmark pages:**
- The Elo pages still show the old sequential ratings and the old rank headline. They switch to the order-independent rating with intervals and the baseline anchor in one step, together with any new entrants, so the published numbers change once.
- The "Why mltreelib" page is rewritten in the maintainer's own words about why the project exists.

**F. Found during the helper pass (typed content with no generator, or figures that may be stale):**
- Typed tables and prose with no run block: the components page (variance, covariance and NMF tables, the API-gap block), the piecewise-linear comparison table, solvers section 5 (wall-clock column), uplift sections 4 and 5, survival section 4, regression section 4, clustering sections 4 and 5, the uplift blog (20-seed table, population effect figures), the zero-imputation blog (sections 1, 4, 5), the two-stage survival blog (sections 4 onward), the credit-scoring blog (sections 2, 4, 5 and the rule-set tail).
- Blog numbers that came from example scripts whose splits are now seeded helper splits: two-stage survival (C-index, integrated Brier score, cohort sizes), causal uplift (Qini, AUUC, 20-seed table), credit scoring (cost, queue, AUROC, Brier and the 20-seed table). The credit post also had two numbers that matched no run before this work. Each is either regenerated from the script and tied to a run block, or removed.
- Figures that may embed old numbers: real-survival, real-uplift, real-credit, real-diamonds-leaves, real-wine-clustering, causal uplift figures, the growth-knob grid, the solver Pareto figure, and the quickstart tree-versus-flat figure. Re-render with the owning scripts.
- The hyperparameter-search example (`hpo_shapes.py`) was switched to the seeded split helper but not run end to end (each study can take up to 10 minutes); run it once and record the result.
- API reference pages other than the data page are stale against the generator (old blurb links, duplicate class names, a dropped note); regenerate and review the whole set in one change.

**E. Generated files:**
- Regenerate the model-readable index files and the sitemap after every docs change, and fail the build if they are stale.

## Decisions Recorded

- **Defaults are identical on every surface.** The Rust core parameter struct, the Python constructor and the leaf control all use the same values: depth 6, minimum rows 15, ridge penalty 0.5. A parity test compares the core defaults with the constructor and the leaf control, so a drift fails the build.
- **Serialised models carry no wall-clock timings.** Per-node timing diagnostics stay in memory only. A model refitted on identical inputs on the same machine saves to byte-identical output; total fit time is a diagnostic.
- **Numerical contract.** Results are reproducible to about 12 significant digits on the same platform, build, BLAS and thread settings; no bit-identical guarantee across platforms, BLAS, thread counts or versions. Stated in the serving-contract guide and tested at a relative tolerance of 1e-12.
- **Release gating.** No pull request and no version until the documentation rework and the open bugs (issues 24 to 27) are fixed, so the review sees the complete picture; then one pull request and one version.
- **Optional dependencies.** `pygraphviz` belongs to the `docs` dependency group only (docs builds need system Graphviz). `cvxpy` is the optional `cvx` extra; it backs three exact reference solvers (sparse rule selection, ensemble pruning, an optimal stump by mixed-integer programming) that cross-check the native solvers, and nothing in the core needs it.
- **Missing values.** NaN and `None` pass through; infinities and the pandas NA and NaT singletons are treated as missing at the model input boundary. Data helpers never alter missing values.

## Out of Scope

- Multiway splits.
- New leaf families or ensembles (Specs 28 to 32).
- The Elo benchmark protocol and any new entrants (separate spec).
- Per-dataset hyperparameter tuning as a user feature (the existing optimizer covers it).
- Changing the Rust numerical kernels.

## Further Notes

- Origin: the audit found that the pruning example never worked, so it was a documentation error, not a lost parameter. The history shows the docs introduced it in a consolidation commit one day before the audit.
- Default changes are visible behaviour changes. They go in a minor release with an explicit changelog table, and the arena publishes default-versus-default results for both the old and the new defaults.
- The confirmation group is where the broader suites (see the datasets spec) are most useful, because they were not used to choose the defaults.
