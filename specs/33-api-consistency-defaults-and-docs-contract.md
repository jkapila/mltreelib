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
- **Documentation contract.** Every parameter named in a guide must be accepted by the live constructor. The runnable-snippet gate already in the docs build is the enforcement mechanism for snippets; a separate check covers inline parameter names and option values.

## Testing Decisions

- A good test checks externally visible behaviour: what the constructor accepts, what it rejects, what a documented example prints. It does not inspect internals.
- The highest seam is the public constructor plus the reference documentation. One test enumerates the constructor's parameters and accepted values and compares them with the reference page. A second test runs a bare `ModelTree()` and asserts the documented defaults.
- Pruning behaviour is tested on small deterministic datasets: a split is kept only when its gain exceeds the penalty; the scale-free option is invariant when the target is multiplied by a constant; the pruning path is monotone in the penalty; reduced-error pruning never increases holdout error on the pruning set.
- Best-first growth, if built, is tested by comparing the tree at a leaf budget against the best tree reachable by exhaustive search on a small problem, and by checking that the budget is never exceeded.
- Default selection is tested by a reproducibility test: the search with a fixed seed on a fixed small dataset group returns the same configuration.
- Prior art: the existing parameter-validation tests, the runnable-snippet tests in the docs build, and the default-search benchmark harness.

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
