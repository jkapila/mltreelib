# Spec 35: Simple Object Interfaces and Per-Family Summaries

Status: partly landed (parts A and B shipped; C, D and the policy table open)

Scope boundary: this spec owns the class-first surface for data (`Dataset`), evaluation (`MetricsReport`), model hand-off (`fit` and `predict` accepting a dataset), and the printed summaries every model family should produce. It does not change split criteria or leaf mathematics, except for the one gap named in part D. It builds on Spec 25 (model surface and reports) and Spec 34 (data helpers), and finishes items those specs promised.

## Problem Statement

A documentation audit and a read of the code found that the library works but asks users to remember too much, and that most families cannot show what they learned.

1. **Too many functions to thread together.** Preparing data takes `xy`, `train_test_split`, `take`, `select_rows`, `describe` and more, each returning something that has to be passed to the next call and then unpacked into `fit`. The maintainer finds this as heavy as the scikit-learn style the package is meant to replace.
2. **Metrics are a class used like a function.** `Metrics().regression(y, p)["rmse"]` appears 64 times in docs and examples and returns an unformatted dict. There is no one-shot, printable view of a model's performance for a task.
3. **Summaries exist for one family.** Of 47 working leaf and family combinations, only a Gaussian linear leaf prints a coefficient table with standard errors and p-values. Binomial, probit, Poisson, gamma, Tweedie, Tobit, negative binomial and beta linear leaves print a stub ("Linear model: intercept=-1.5357, 3 coefficients (summary covariance not computed)"). Quantile and Huber leaves print "Mean leaf model" although the value is a quantile. MARS and k-NN leaves, softmax and survival leaves show no basis table, neighbour table or curve.
4. **The summary is not demonstrated.** `ModelTree.summary()` appears in no guide or example. The rpart-style printout of a tree appears only in the quickstart.
5. **Cluster and anomaly models report almost nothing.** `ClusterModel.report` holds only the number of clusters, `cluster_profiles` is never populated, `rules()` is empty for two engines, and neither facade has a `summary()`. Spec 22 promised clustering validity, stability and noise-rate reporting and Spec 23 promised a four-section report per leaf; neither exists.
6. **Causal leaves ignore the leaf type.** With the causal gain, a linear leaf and a mean leaf give identical results, because the leaf estimate is always a difference of means. A comparison of leaf types for heterogeneous effects, which the uplift guide implies, is impossible.
7. **Guides carry helper code the library should own.** The uplift guide defines a 15-line function to compute policy value with a confidence interval, and the depth and leaf settings in several examples are too small to show the method, which makes results look unconvincing.

## Solution

Classes that hold state and print themselves, with the existing functions kept underneath; one summary shape that every family fills in as far as its mathematics allows; and a model that accepts the dataset object directly.

The user's view:

- `ds = Dataset.load("diamonds", target="price")`, then `ds.describe()`, `ds.missing()`, `train, test = ds.split(test_size=0.25, seed=2026)`.
- `model.fit(train)` and `model.predict(test)` with no unpacking.
- `print(MetricsReport("regression", test.y, model.predict(test)))` for the full view.
- `print(model.summary())` for a readable table of what each leaf learned, whatever the family.

## User Stories

1. As a data scientist, I want one object that holds my columns and their roles, so that I do not thread five function results into `fit`.
2. As a data scientist, I want to mark target, time, event, treatment, outcome, groups and weights once, so that every later call knows them.
3. As a data scientist, I want `describe`, `missing`, `checks`, `associations` and `iv` as methods of the dataset, so that profiling reads as one workflow.
4. As a data scientist, I want `split` to return datasets, so that I can fit and score without index bookkeeping.
5. As a data scientist, I want stratified, grouped and time-ordered splits by column name, so that I do not build label lists by hand.
6. As a data scientist, I want `model.fit(train)` and `model.predict(test)` to accept the dataset directly, so that the roles and column types travel with the data.
7. As a data scientist, I want a printed text overview of a dataset, so that I see rows, features, kinds and missing counts without calling anything.
8. As a data scientist, I want missing values left untouched by every dataset method, so that the model sees the real pattern.
9. As an evaluator, I want one call that takes the task name and the values and prints the full metric view, so that I do not pick among 29 methods.
10. As an evaluator, I want regression reports to show errors, error deciles and optional per-group tables, so that I see where the model is weak.
11. As an evaluator, I want classification reports to show the confusion matrix, per-class precision, recall and F1, discrimination and calibration, so that I judge more than accuracy.
12. As an evaluator, I want time-series, survival and uplift reports in the same shape, so that every task reads alike.
13. As an evaluator, I want an uplift report with a policy table (treat selected, treat all, treat none) with confidence intervals, so that I do not hand-code it.
14. As an evaluator, I want the report as text, Markdown, dict and JSON, so that I can print it, paste it or store it.
15. As a statistician, I want a coefficient table with standard errors, z or t values, p-values and confidence intervals for every parametric linear leaf (binomial, probit, Poisson, gamma, Tweedie, negative binomial, beta, Tobit), so that I can troubleshoot them like statsmodels output.
16. As a statistician, I want the table to say why inference is unavailable when it is (regularised fit, bounded solver), instead of omitting it or inventing p-values.
17. As a statistician, I want a quantile leaf to be labelled as a quantile, with its level and a bootstrap or asymptotic standard error labelled as such.
18. As a statistician, I want a Huber leaf to show robust coefficients and the scale estimate.
19. As an analyst, I want a MARS leaf to show its basis terms (term, hinge, knot, coefficient), so that I can read the shape.
20. As an analyst, I want a k-NN leaf to show its neighbours for a row (rank, row id, distance, target, weight), so that I can see why a prediction was made.
21. As a survival analyst, I want a survival leaf to show the survival curve table (time, survival, interval, at risk, events) and the median, so that I can read each leaf.
22. As a reviewer, I want a model-level report with sections for objective, learner, controls, diagnostics and optional inference, so that I audit a tree like any other facade.
23. As a reviewer, I want the report to state which controls and defaults were used, so that a result can be reproduced.
24. As an analyst, I want cluster models to report a profile table (size, share, per-feature mean, spread and missing, prototype) and validity, stability, balance and noise rate, so that I can judge a segmentation.
25. As an analyst, I want anomaly models to report a top-cases table (rank, row id, score, threshold, percentile, top features), so that I get a review queue.
26. As an analyst, I want `summary()` on cluster and anomaly models that prints the same content as their report, so that every family answers the same question the same way.
27. As an analyst, I want a leaf-model option for heterogeneous effects inside a causal leaf, so that I can compare mean and linear causal leaves.
28. As a learner, I want every guide to say in plain words what each snippet does and what to look for, so that I do not need prior knowledge.
29. As a learner, I want example settings (tree depth, leaf type) chosen so that the method's benefit shows without special tuning, with the setting stated.
30. As a maintainer, I want one text renderer shared by all reports, so that tables look the same everywhere.
31. As a data scientist, I want **every leaf family to have a summary table, without exception** (mean, median, linear and every parametric family, quantile, Huber, MARS, k-NN, survival, Tobit, softmax, heteroscedastic, and any family added later), so that no fitted leaf is a black box.
32. As a maintainer, I want a conformance test that fits every accepted leaf and family combination and fails when its summary is a stub, so that a new leaf family cannot ship without a summary.

## Implementation Decisions

- **Class first, functions underneath.** `Dataset` and `MetricsReport` delegate to the existing helpers and metrics; they do not duplicate logic. A thin function alias is allowed, the class is the documented path.
- **Dataset.** Immutable. Holds columns and roles (target, features, exclude, time and event, treatment and outcome, groups, weights, name). `Dataset.load(name, ...)` wraps dataset loading. Methods for profiling delegate to the profile, association and binning helpers. `split` and `kfold` return datasets; the stratify, group and time options accept column names, `True` (meaning the role's natural column: target, treatment for uplift, event for survival) or row order for time. A dataset exposes `fit_args()` and `predict_args()` so that existing code works unchanged, and the model's `fit` and `predict` additionally accept a dataset (follow-up change in the model input boundary; the target then need not be passed separately). The dataset never changes values, imputes, drops or recodes missing values, and never normalises event codes.
- **MetricsReport.** Class with a task argument and task-specific named inputs. Composes the existing metric methods and adds only presentation. Presents scalars grouped under headings and tables for deciles, confusion matrix, per-class results, calibration bins, by-group results and, for uplift, a policy table with confidence intervals (selected group, everyone, no one) computed in the library. Output forms: text, Markdown, dict, JSON; scalar access by key. Shows missing counts; never imputes or drops silently.
- **One summary shape per family.** Every fitted `ModelTree` gains `report()` returning the shared four-section report (objective, learner, controls, diagnostics, plus inference when valid), and `summary()` prints the per-leaf detail. Per-leaf content by family: parametric linear leaves show estimate, standard error, statistic, p-value and interval for coefficients, with family, link, dispersion, deviance and sample size; the covariance comes from the fitted solver (the iteratively reweighted path must return it), and when it cannot (a penalised or bounded fit) the table says so and shows estimates only. Quantile leaves state the level; Huber leaves show the robust scale; MARS shows basis terms; k-NN shows neighbours on request for a row; survival leaves show the curve table and median; causal leaves show the effect estimate with its interval. No p-values are shown unless the mathematics supports them (Spec 25 decisions 5 and 6).
- **No leaf family without a summary (hard rule).** The summary of a leaf always shows what the leaf learned: its model terms (coefficients, basis terms, neighbours, curve points, quantile level) with uncertainty where valid, its support, and its fit quality. A conformance test enumerates every accepted combination of leaf and family and asserts that the summary is not a stub (no "summary covariance not computed" text, no "Mean leaf model" for a non-mean leaf). Adding a leaf family means adding its summary in the same change.
- **Cluster and anomaly.** Both gain `summary()` and a populated `report()`: clustering reports the profile table plus validity, stability, size balance and noise rate (Spec 22 stories); anomaly reports the top-cases table with scores, thresholds and the leading features, and keeps the GritBot case summaries. `rules()` returns something meaningful for every cluster engine or states that the engine has no rules.
- **Causal leaf.** Add a linear option for causal leaves that models the effect inside a leaf as a function of covariates (a treatment term and treatment interactions), with the honest-estimation rule unchanged. Until then the docs state that the causal leaf is a difference of means whatever the leaf setting is.
- **Teaching defaults.** Guides use depth settings that let the method show its benefit (a depth of 4 for the diamonds regression comparison, where linear leaves lower the error by about five percent without other tuning) and state why the setting was chosen.
- **Plain-language rule.** Every guide snippet is introduced in plain words and followed by what to read in the output (a contributor rule).

## Testing Decisions

- A good test checks printed and returned behaviour a user sees, not wiring. Snapshot a small, seeded text report per task and per family. Numbers in a report equal the numbers the underlying metric or fit returns.
- Highest seam: the public classes and `summary()`/`report()`. Cross-check the coefficient table against an independent computation on a seeded dataset (for example a logistic fit with known coefficients) at 1e-6, and check that inference is withheld, with a stated reason, for fits where it is not valid.
- Missing-value tests on every new class: counts shown, nothing dropped, input unchanged.
- Determinism: same seed gives the same split and the same report text.
- Prior art: the profile, split and access tests, the Gaussian summary snapshot, and the report tests from Spec 25.

## Progress and Open Work

Tracked on GitHub. Branch: `spec-33-34/api-consistency-and-data-suite`.

| Issue | What | State |
|---|---|---|
| 16 | Docs showed code that did not run; stale defaults and snippets | fixed (docs pass); figures and typed tables remain (Spec 33 backlog, section F) |
| 17 | Pandas frame rejected by `fit` | fixed (tables by duck typing) |
| 18 | Non-deterministic categorical gate statistics | fixed (ordered maps) |
| 19 | `CatBoostEncoder` example prior | fixed (docs); parameter still to be documented and possibly renamed |
| 20 | Defaults differed between constructor, Rust core and `LeafControl`; timings in saved models | fixed; parity test added |
| 21 | `mltreelib.data` helpers | done (profile, split, association, binning with exact mode, scaling, accessors) |
| 22 | `MetricsReport` | done (uplift policy table, `uplift_threshold`, and `uplift_by_decile` landed; uplift guide migration queued) |
| 23 | `Dataset` class; `fit`/`predict` accept a dataset | done; clustering and anomaly facades do not accept a dataset yet |
| 24 | Per-family summaries and `ModelTree.report()` | open (part C) |
| 25 | `ClusterModel` and `AnomalyModel` report and summary | open (part C) |
| 26 | Causal linear leaf; `leaf="linear"` equals `leaf="mean"` under the causal gain | open (part D) |
| 27 | `Metrics` hangs, panics or silently recodes NaN inputs | open bug; `MetricsReport` refuses missing values outside regression and label classification until fixed |

**Open work in order:**
1. Uplift policy table (selected group, everyone, no one, each with a confidence interval) and a library helper that chooses the threshold on validation data, inside `MetricsReport`; uplift-by-decile table.
2. Rework the regression guide (depth 4, where linear leaves lower the error by about five percent; print `summary()` and one `leaf_summary`) and the uplift guide (use `Dataset`, `MetricsReport`, the policy table; explain each snippet in plain words; state that a causal leaf is a difference of means whatever the leaf setting).
3. Move the remaining docs pages and example scripts (about 40 files, 124 lines) from the helper functions to `Dataset` and `MetricsReport`.
4. Fix `Metrics` missing-value handling (issue 27), then remove the `MetricsReport` refusals.
5. Parts C and D (issues 24 to 26), ending with demonstrations in the guides.
6. Let `ClusterModel`, `AnomalyModel` and the other facades accept a dataset.

## Out of Scope

- New split criteria or leaf families, apart from the causal linear option.
- A graphical equation view of leaves (parked; feasible later through the graph export).
- Automatic model selection from a report.
- Changing the numbers a metric returns.

## Further Notes

- Origin: findings from the documentation audit and the helper pass, mostly in the uplift and regression guides.
- Parts A (Dataset) and B (MetricsReport) are in progress under issues 22 and 23. Part C (family summaries, cluster and anomaly) and part D (causal linear leaf) need their own issues before work starts.
- Existing doc and stub drift to fix on the way: the type stub for `leaf_stats` and a stub entry for a method that does not exist.
