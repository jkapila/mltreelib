# Spec 34: Survival and Uplift Real-Data Suite

Status: proposed

Scope boundary: this spec owns the set of real survival, competing-risks and uplift datasets used in the documentation and benchmarks, how they are fetched on demand, and the elaborated, reproducible results pages built on them. It does not change the regression and classification arena (separate spec) and does not add model families.

## Problem Statement

The two task families the library is most distinctive at, survival and uplift, have almost no real public data behind them in the documentation.

1. **One dataset each.** Survival guides rest on a single clinical dataset and uplift guides on a single email-marketing experiment. A reader cannot tell whether results generalise, and a single dataset cannot show how the models behave with different censoring rates, competing events, treatment ratios or missing values.
2. **The big benchmark has no survival or uplift tasks.** The tabular arena used for regression and classification contains neither, so these families have no comparative evidence at all.
3. **Fetching was broken.** The registry entries for a set of survival datasets pointed at identifiers that return errors or at different data (one entry labelled as a clinical dataset is a financial index). Docs snippets read untracked local files, so a reader who installs the package cannot reproduce them.
4. **Sources are unstable and licences unclear.** The easiest mirrors regenerate files when upstream packages update, so a pinned checksum can break. Several well-known uplift datasets are competition data with unknown terms, one is non-commercial and share-alike, and others have no stated licence.
5. **Results are thin.** Existing pages show one run on one dataset. They do not show the model against meaningful baselines for these tasks, uncertainty across datasets, or failure modes such as tied times, event coding conventions, grouped observations or non-compliance.

## Solution

A small, curated suite of real datasets per task family, each fetchable on demand with a pinned checksum and a recorded licence note; a loader contract that normalises each dataset's quirks into one documented schema; and a results section that shows, for each family, how a single tree compares with simple and strong baselines across the suite, with uncertainty, with the data's failure modes called out.

The user's view: `load_dataset` can fetch any suite dataset by name, the survival and uplift guides show several datasets side by side, and every number on those pages can be regenerated from a command with the same data and seed.

## User Stories

1. As a reader of the survival guide, I want to see the same workflow on several real datasets, so that I can judge whether the method generalises.
2. As a reader of the uplift guide, I want more than one randomized experiment, so that I can see how policy value varies with the treatment ratio and outcome type.
3. As a user, I want to fetch any suite dataset by name with one call, so that I can reproduce a documented result without hunting for files.
4. As a user, I want the first download to verify a checksum, so that I know I have the data the docs used.
5. As a user, I want a clear message when an upstream file changed, so that I know why the numbers differ instead of getting a silent difference.
6. As a user, I want the dataset's licence and source printed before the first download, so that I can decide whether I am allowed to use it.
7. As a user in a restricted environment, I want an offline mode that fails clearly, so that a script never reaches the network by surprise.
8. As a maintainer, I want nothing from this suite bundled in the package, so that the wheel stays small and no redistribution question arises.
9. As a maintainer, I want broken registry entries removed or repointed, so that a listed dataset always loads.
10. As a maintainer, I want a test that every registry entry resolves and matches its checksum, so that broken entries cannot linger.
11. As a data scientist, I want every survival dataset exposed with the same schema (time, event, features, optional group), so that one script runs over the whole suite.
12. As a data scientist, I want event coding conventions normalised and documented per dataset, so that a coding mismatch cannot silently invert a result.
13. As a data scientist, I want competing-risk datasets to expose the cause of each event, so that I can fit cause-specific and subdistribution models.
14. As a data scientist, I want datasets with grouped observations (two eyes per patient, repeated records) flagged and grouped in cross-validation, so that results are not leaky.
15. As a data scientist, I want long-format datasets collapsed to one row per subject by a documented rule, so that I know what the tree sees.
16. As a data scientist, I want datasets with time-dependent covariates excluded or marked, so that a naive tree cannot leak the future.
17. As a data scientist, I want every uplift dataset exposed with the same schema (treatment, outcome, features), so that one script runs over the whole suite.
18. As a data scientist, I want randomized and observational data kept apart and labelled, so that I do not read a causal effect off confounded data.
19. As a data scientist, I want non-compliance datasets to expose both assignment and receipt, so that intent-to-treat and per-protocol analyses are both possible.
20. As a data scientist, I want large uplift datasets available as a documented, seeded subsample, so that a single tree runs in seconds.
21. As a reader, I want the single tree compared with a baseline with no tree, a simple linear model, and a strong ensemble for each task, so that I can see what the tree gains and loses.
22. As a reader, I want results shown with confidence intervals across datasets and repeated splits, so that I can tell real differences from noise.
23. As a reader, I want evaluation metrics appropriate to each task: concordance and integrated Brier score for survival, cumulative incidence for competing risks, Qini or policy value for uplift.
24. As a reader, I want each dataset's failure modes described next to its result, so that I learn when the method struggles.
25. As a reader, I want every table on the results pages generated from a command, so that the page and the numbers cannot disagree.
26. As a maintainer, I want the suite's checksums recorded with the date they were taken, so that a mirror regeneration is diagnosable.
27. As a maintainer, I want a hash-mismatch fallback or a clear version note, so that a regenerated mirror does not break every user.
28. As a licence-conscious user, I want datasets with restrictive or unknown terms clearly marked and excluded from the default suite, so that the default is safe to use commercially.
29. As a benchmark reader, I want survival and uplift results reported under the same seed, split and confidence-interval protocol as the regression and classification arena, so that the numbers are comparable in spirit.
30. As a maintainer, I want the suite to be usable as a confirmation group when choosing default parameters, so that defaults are validated on data not used in the search.

## Implementation Decisions

- **Suite membership.** Survival core: a clean single-event clinical dataset with no missing values and a randomized-trial dataset with competing events and many missing biomarkers, plus a third dataset with competing progression and death. A smaller randomized cancer trial and a larger medium-size relapse dataset are optional extras. Uplift core: a small randomized job-training experiment, a randomized psychological-intervention trial with non-compliance and a binary outcome, and a large randomized advertising or retail experiment used only as a documented subsample. The existing clinical survival dataset and the existing email experiment stay. Candidate datasets and their properties were checked by download; final membership is confirmed against the licence column below before adoption.
- **Not in the default suite.** Competition and hackathon uplift datasets with unknown terms, a non-commercial share-alike advertising dataset, observational datasets presented as causal, semi-synthetic benchmarks without a licence, and datasets with time-dependent covariates or clustered recurrent events that need dedicated methods. These may be listed as optional, user-fetched extras with their licence printed.
- **Licence policy.** Nothing is bundled. Every registry entry carries a source and a licence note. Datasets with restrictive or unknown terms are excluded from the default suite and from any page that implies free reuse. The documentation states provenance and licence next to each dataset.
- **Fetch contract.** Datasets resolve through the existing dataset API under stable names. Downloads go to a temporary file, are verified against a pinned checksum, then renamed. Each registry entry records the checksum with the date it was taken. A mismatch produces a clear error naming the expected and actual checksum and pointing at the upstream source, and an escape hatch allows an explicit, logged override for users who accept a changed upstream file. An offline environment variable forbids network access and fails with a clear message. Nothing downloads at import time.
- **Stable sources.** Where a mirror regenerates files, the registry records the version or date in the note and prefers a source that does not regenerate. The checksum is treated as a record of the data used in the docs, not a promise about upstream.
- **Normalised schema.** A loader layer returns each dataset in one schema per task family. Survival: duration, event indicator coded consistently with the coding convention documented per dataset, optional cause, optional subject group, features. Uplift: treatment indicator, outcome, optional assignment-versus-receipt pair, features, and a randomized-or-observational label. The raw columns remain available. Quirks are normalised in the loader and documented on the dataset's page: event coding that differs between datasets, string event labels, long-format records collapsed per subject by a stated rule, unnamed index columns, and unusual missing-value tokens.
- **Splitting rules.** Cross-validation respects groups (patient, eye, subject). Competing-risks datasets keep every cause in each fold. Uplift splits stratify on treatment so both arms appear in every fold.
- **Registry repair.** The broken entries in the existing registry are removed or repointed to working sources, and each remaining entry is verified by a test.
- **Results design.** For each family, one results page runs the same pipeline over every suite dataset. Survival reports concordance, integrated Brier score and, for competing risks, cumulative-incidence error, for a no-tree baseline, a regularised linear model, the single tree with survival leaves, and one strong ensemble reference. Uplift reports Qini and expected policy value for a no-model baseline (treat everyone or no one), a simple two-model approach, the single causal tree, and one strong reference. Results use repeated splits and a bootstrap across datasets and splits for the intervals, following the same method as the arena. Each page lists, per dataset, what is unusual about it and where the tree does worse.
- **Reproducibility.** A single command regenerates the tables and figures with fixed seeds, and the documentation shows the command's real printed output. Results pages state the data checksums used and the date they were taken.
- **Scaling.** Large datasets are subsampled by a stated seeded rule and the documentation says so; the full file is never required for a documented result.
- **Relation to defaults.** The suite is a natural held-out confirmation group when choosing default parameters, because it was not used in the default search.

## Testing Decisions

- A good test checks behaviour a user can observe: that a named dataset loads, has the documented schema, row counts and event rates, and that a corrupted download is rejected. It does not inspect how the downloader is written.
- The highest seam is the public dataset loader. Tests cover: every registry entry resolves and matches its checksum when the network is allowed, and is skipped otherwise; a tampered file raises the checksum error; the offline setting prevents network access; each normalised schema contains the documented columns; event coding is normalised so the event rate matches the documented value; groups are present where documented; the seeded subsample is reproducible.
- Pipeline tests run on tiny synthetic datasets that mimic each quirk (competing causes, grouped records, non-compliance) and assert metric values on cases with known answers, for example a perfectly separating tree and a null effect.
- Results pages are covered by the existing runnable-snippet gate in the documentation build, so a page whose printed output differs from a rerun fails the check.
- Prior art: the existing dataset registry tests, the survival and causal tests in the model test suite, and the documentation snippet runner.

## Data Preparation Helpers

The documentation audit found that the same small data-preparation steps are written out by hand in many snippets: about 36 manual split indexings (`int(n * 0.8)` and slices), about 27 pandas conversions of missing values to `None`, and about 9 list comprehensions that rebuild a feature matrix from a dict of columns. There is no train, validation and test split helper anywhere in the package. Hand-written versions differ slightly between pages, which makes results harder to reproduce and the snippets longer than the idea they teach.

**User stories**

31. As a user, I want one call that splits a dataset into train, validation and test sets with a seed, so that I do not write index arithmetic.
32. As a user, I want splits that can be stratified on the target or on a treatment flag, so that every fold contains both arms and every class.
33. As a user, I want splits that respect a group column (patient, eye, subject, store), so that related rows never leak across sets.
34. As a user, I want an ordered split by time, so that forecasting and temporal checks do not shuffle the future into training.
35. As a user, I want the same split from the same seed on every machine, so that documented numbers can be reproduced.
36. As a user, I want to take a loaded dataset and get the feature matrix, target and column names in one call, so that I do not rebuild them with a comprehension.
37. As a user, I want missing values to stay exactly as they are in every data helper, so that the model sees the real missingness pattern and nothing is imputed, dropped or recoded behind my back.
38. As a user, I want survival datasets returned as duration, event and features, and uplift datasets as treatment, outcome and features, so that I do not index columns by hand.
39. As a user, I want repeated and cross-validated splits from the same helper, so that the arena and the guides share one splitting rule.
40. As a maintainer, I want every documented snippet to use these helpers, so that a change in splitting behaviour is made in one place.

**Implementation decisions**

- The helpers live in the dataset module (the data layer that already loads, caches and describes datasets), not in the model classes. The model stays free of any data-splitting dependency.
- The split helper accepts a loaded dataset, a plain dict of columns, or array-like inputs, and returns the same kinds of objects it was given. It supports a seed, fractions or counts, stratification on a named column, grouping on a named column, and a time-ordered mode. It never shuffles in the ordered mode.
- Missing values are an integral part of the data. No helper imputes, drops, normalises or recodes them; they pass through unchanged. Accepting missing values in any representation (NaN, None, null, Arrow or pandas missing) is the model's job at its input boundary, together with surrogate-based routing, and is specified in Spec 33.
- Survival and uplift loaders return the normalised schema from this spec, with helper accessors for the time and event columns or the treatment and outcome columns.
- The benchmark arena adopts the same splitting helper so that documented splits and arena splits are the same rule.
- The helpers add no dependency on scikit-learn or any other third-party package. Anything that can be built inside the library is built inside it; the package aims to reduce its dependencies, not add to them.

**Testing decisions**

- Test behaviour: sizes add up, a stratified split preserves class and arm proportions within a stated tolerance, grouped splits never share a group across sets, ordered splits are strictly ordered in time, the same seed gives identical indices, and missing values are untouched by every helper (counts and positions are identical before and after).
- Prior art: the existing dataset and encoder tests, and the existing snippet gate, which will fail if a page still uses a hand-written split after the helper exists.

## Data Helper API Contract (release scope: P1 and P2)

All helpers live in the data module (`mltreelib.data`), are pure Python with the standard library only (the package has zero runtime dependencies), and take a dict of columns (as returned by `load_dataset`) or any table the model input boundary accepts. They never modify their inputs and never alter missing values. "Missing" means `None` or NaN; infinities are counted separately as `n_inf` and ignored by statistics. Any helper that must exclude rows to compute a statistic reports how many rows it used.

**Profiling**
- `describe(columns)`: one entry per column, like a data-frame describe but aware of missing values. Counts: rows, missing, percent missing, infinities, distinct. Numeric statistics are computed for every column whose observed values are numbers: mean, standard deviation, min, max, chosen percentiles (including quartiles), IQR, MAD, coefficient of variation, trimmed mean and trimmed standard deviation (default trim 10% per end), skewness, excess kurtosis, outlier count and percent by the Tukey rule, and counts of zero, negative and positive values. Categorical columns report the mode and top levels with counts. A `kind` hint (numeric, categorical, or ordinal candidate) is reported, never enforced; the user can override it per column, including marking a column ordinal. Integer columns with fewer than six distinct values are reported as ordinal candidates, because they are often category codes or ordered ratings (such as a 1 to 5 scale), and an ordered level is useful for monotone constraints. A flag chooses whether infinities are also counted as missing (default: reported separately, ignored by statistics).
- `missing_profile(columns)`: per-column missing counts and percents, per-row missing counts, number of complete rows, missing-pattern counts (which set of columns is missing together, with row counts), and a co-missing count matrix.
- `missing_vs_target(columns, y)`: per column, the target mean when the column is missing versus observed, with both counts; rows whose target is missing are counted and left out of the means.
- `infer_types(columns)`: kind and number of levels per column, using the same rule as the dataset builder.
- `near_zero_variance(columns)`: columns flagged by the caret rule (frequency ratio of the two most common values and percent unique), computed on observed values.
- `constant_columns(columns)`, `duplicate_rows(columns)` (groups of row indices), `duplicate_columns(columns)` (groups of names). Missing compares equal to missing.
- `class_balance(y)`: counts, proportions, imbalance ratio, missing count.
- `outlier_flags(values, method, k)`: a list of True, False or None (None for missing). IQR and MAD methods. Flags only; nothing is removed.
- `schema_capture(columns)` and `schema_validate(columns, schema)`: per-column kind, levels or range, and missing count; validation returns a list of violations and treats missing as allowed unless asked otherwise.

**Splitting** (all return index lists only; deterministic for a given seed on every platform)
- `train_test_split(n, test_size, seed, stratify, groups, time_order)` and a three-way variant for validation.
- `kfold(n, k, seed, stratify, groups, time_order)`: stratified and grouped folds; with a time order, expanding-window folds.
- `take(columns, indices)`: subset a dict of columns by row indices.

**Association and screening**
- `correlation_matrix(columns, method)`: Pearson or Spearman on pairwise-complete rows, with the pair counts; `correlation_filter(result, cutoff)`: columns to drop by the greedy caret rule.
- `cramers_v_matrix(columns)`: bias-corrected Cramer's V with chi-square p-values for categorical columns.
- `vif(columns)`: variance inflation factors on complete cases, with the number of rows used.
- `leakage_hints(columns, y)`: single-feature predictive score per column, flagged above a threshold. Flags only.

**Binning and information value**
- `binner(values, ...)`: numeric binning by quantile or equal width with a minimum observations per bin; returns edges, labels, counts and a bin index per row, with missing rows getting `None`.
- `iv_bins(values, y, ...)`: bins for a binary target with events, non-events, weight of evidence and information value per bin, a separate missing bin that is never dropped, and prior smoothing for zero cells (reported, not dropped). Optional monotone constraint on the event rate, enforced by merging adjacent bins (pool-adjacent-violators), and bounds on the number of bins and the minimum bin fraction. `iv_levels(values, y)` does the same for categorical columns.

**Scaling and transforms** (fit, transform and inverse transform; missing passes through unchanged)
- log1p, power, mean-difference, median-difference, Box-Cox and Yeo-Johnson (lambda fitted on observed values), alongside the existing scalers.

**Not included in this release (later):** drift, variable-voting and variable-clustering selectors, stratified sampling, class weights, mutual information, Kendall tau.

**Existing dataset-builder options.** The opt-in drop and impute policies stay, with "allow" (no treatment) as the default; new helpers do not copy that pattern.

## Decisions Recorded

- **Weight-of-evidence sign convention stays** as shipped: WOE = ln(event share / non-event share), documented in the reference page. It is the public convention for now; changing it would be a breaking change and needs an issue.
- **No further data-analysis tooling in this release.** The maintainer asked for completeness and reliability of what exists before adding capability (for example tools that describe how the data is represented); those are parked.

## Out of Scope

- Observational causal inference methods and any claim of causal effect from observational data.
- Time-dependent covariates, recurrent events with frailty, and multi-state models.
- Bundling any dataset in the package.
- The regression and classification arena protocol and entrants.
- Semi-synthetic benchmark families, unless a licence is confirmed.
- Training or evaluating foundation models.

## Further Notes

- Verification found that several candidate sources work and several do not. The mirrors used most often for survival datasets regenerate files when the upstream package updates, so the pinned checksum is a record, not a guarantee. Competition-hosted uplift data comes with unknown terms and is large; the advertising dataset with a clear licence is non-commercial and share-alike. The registry therefore keeps licence and source notes next to each checksum.
- Two of the datasets most often cited for uplift were not found at a stable, verifiable source. They are not part of this spec until a source is confirmed.
- A survival code gotcha worth documenting once, in the loader: several datasets code an event as 1 for death and others as 2 for death, and one trial codes three outcomes.
