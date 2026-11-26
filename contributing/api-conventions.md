# API conventions

## One surface, four choices

A tree is built from four choices. Every public parameter belongs to exactly one group, and the reference page is organised by them:

- **split**: how a node is divided (gain criteria, thresholds, categorical handling, oblique splits, constraints).
- **grow**: when and in what order the tree grows (depth, minimum rows, leaf budget, significance gates).
- **prune**: how a grown tree is cut back (`prune`, `cp_alpha`, scale-free `cp`).
- **leaf**: what is fitted in a terminal node (mean, linear, quantile, distributional, survival and so on).

Plus standard knobs independent of growth and leaf: seeds, sample sizes and weights, honesty, missing-value routing. Growth knobs live in `GrowthControl`, leaf knobs in `LeafControl`; the flat constructor accepts both.

## Names

- Umbrella entry points are `ModelTree`, `ClusterModel`, `AnomalyModel`, `ComponentModel`, `RuleModel`. Docs and examples use these, imported from top-level `mltreelib`, never the backward-compatible aliases (`AnomalyTree`, `ClusterTree`) or submodule paths.
- One canonical name per concept. Do not add an alias without a reason; aliases created today's duplication. A rename is a public API change: open an issue and record it in the changelog.
- Data helpers live in `mltreelib.data`; dataset loading and fetching in `mltreelib.datasets`.

## Defaults

- **One set of defaults per model family on every surface** (Rust core, Python constructor, controls). Families may differ: `ModelTree` 6 / 15 / 0.5; `ClusterModel` and `RuleModel` `min_rows` 20; `AnomalyModel` `max_depth` 10, `min_rows` 20. `tests/test_sanity_basics.py` holds the table; say so in the docs where a family differs. Changing a default means changing all of them and the parity test. Defaults are a behaviour change: list them in the changelog.
- The reference table of defaults is generated from the live constructor, not typed by hand.

## Missing data

- `None` and NaN are missing. The model routes them natively (surrogate splits, learned missing direction); infinities and the pandas NA and NaT singletons are treated as missing at the model input boundary.
- **Data helpers never impute, drop, recode or reorder.** They count (missing profile, per-row counts), flag (outliers), or pass through. Opt-in policies that change data must default to "allow" (no treatment).

## Inputs and outputs

- Models accept a list of rows, a NumPy array, a dict of columns, or a data-frame or Arrow-style table, by duck typing, without importing those packages.
- Helpers take and return plain Python types (lists, dicts, floats, strings), never mutate their inputs, and never depend on global random state; randomness takes an explicit seed.

## Dependencies

- Zero runtime dependencies. **No scikit-learn, ever**; the `fit`/`predict` shape is a convention only. Shrink the dependency set; build inside the package when possible. Optional extras (parquet reading, plotting, hyperparameter search) are lazy imports with a clear error, never required for core functionality.

## Reproducibility

Results are reproducible to about 12 significant digits on the same platform, build, BLAS and thread settings. Saved models carry no wall-clock timings, so identical fits save byte-identical files. See the serving-contract guide.
