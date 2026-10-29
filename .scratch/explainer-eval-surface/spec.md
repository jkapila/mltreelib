# Model explanation & evaluation surface: break-down attribution, Shapley values, importance methods, and metric curves

Status: ready-for-agent

## Problem Statement

A user can fit an mltreelib `ModelTree`, read its structure (rpart-faithful text, graphviz diagram), and score test rows — but cannot yet **explain an individual prediction** ("why did this customer get this price?"), cannot compare **feature importance methods** beyond raw split gains, and cannot evaluate **probability quality** (calibration, ROC/PR/lift) the way practitioners expect from a single-model toolkit. Explainer workflows (DALEX break-down plots, Shapley attribution) are standard for interrogating one model; without them, and without a structured API reference (everything is crammed into one "Data & Model" page), the package is not yet a complete single-tree toolkit.

## Solution

A layered explanation & evaluation surface across the shipped `ModelTree` and `Metrics` APIs:

- **Break-down attribution** for one row (DALEX-style): the prediction decomposed along its decision path into per-feature contributions, plus a waterfall plot.
- **Shapley values** for one row (exact tree Shapley), and mean-|φ| **Shapley feature importance** over a dataset.
- **Gain-share importance** (rpart variable importance — already landed) and **permutation importance** as the model-agnostic check.
- **Evaluation curves** in `Metrics`: calibration (reliability) bins, ROC + AUC, precision–recall + average precision, cumulative gain/lift — returned as data so any plotting stack works.
- **More plotting options** in the examples/docs layer: break-down waterfall, importance bars, calibration/ROC/PR panels, partial dependence — matplotlib as a dev extra, the library stays plotting-free.
- **API reference in three sections — Data, Model, Metrics** — explaining the full surface (every parameter, method, and return shape).

## User Stories

1. As a data scientist, I want to decompose a single prediction into per-feature contributions (break-down), so that I can explain to a stakeholder why this row received this prediction.
2. As a data scientist, I want the break-down contributions to telescope exactly — baseline + sum of contributions + residual = prediction — so my explanation is complete and auditable.
3. As a data scientist, I want the break-down baseline to be the training mean, so contributions read as "what moved this row away from the average".
4. As a data scientist, I want break-down steps in decision-path order, so I can narrate the route the row took through the tree.
5. As a data scientist, I want break-down steps attributed to named columns (not indices), so the output is directly readable and plottable.
6. As a data scientist, I want the within-leaf model adjustment reported as a separate residual term, so linear-leaf predictions still explain cleanly.
7. As a data scientist, I want break-down for rows with missing cells to follow the same learned routing as prediction, so the explanation matches the prediction exactly.
8. As a data scientist, I want exact Shapley values for a prediction (tree Shapley), so feature credit is order-independent and fairly allocated among interacting features.
9. As a data scientist, I want the Shapley output to include the base (expected) value and satisfy efficiency — base + sum of values = prediction — so the attribution is complete.
10. As a data scientist, I want mean-|φ| Shapley feature importance over a dataset of rows, so I can rank features on attribution strength, not just split frequency.
11. As a data scientist, I want features unused by the tree to receive exactly zero Shapley value, so the attribution is honest.
12. As a data scientist, I want Shapley values that are deterministic run-to-run, so reports and tests are reproducible.
13. As an ML engineer, I want gain-share importance (rpart's variable importance) as the cheap default, so I can rank features without extra passes over data.
14. As an ML engineer, I want the per-split gain visible in the split listing, so I can audit how importance accumulates.
15. As an ML engineer, I want permutation importance on a validation set, so I can get a model-agnostic importance estimate tied to actual predictive performance.
16. As an ML engineer, I want all importance methods to share one output convention (named columns, sorted descending), so I can compare methods side by side.
17. As a data scientist, I want a calibration (reliability) curve — mean predicted score vs observed frequency per bin — so I can tell whether predicted probabilities can be trusted.
18. As a data scientist, I want an ROC curve with AUC, so I can measure discrimination at any threshold.
19. As a data scientist, I want a precision–recall curve with average precision, so I can evaluate on imbalanced targets where ROC overstates quality.
20. As a data scientist, I want a cumulative gain (lift) curve, so I can size targeting campaigns ("top decile captures X% of churners").
21. As a data scientist, I want curves returned as plain numeric series (points, counts, summary scalars), so I can plot them with any tool and diff them in tests.
22. As a data scientist, I want curve computation to handle ties and edge bins predictably and be documented, so numbers are stable across runs and machines.
23. As a user, I want a break-down waterfall plot for one prediction, so I can drop an explanation straight into a deck.
24. As a user, I want importance bar charts (gain and Shapley) in the visualization example, so I can see which method to reach for.
25. As a user, I want calibration and ROC/PR panels in the visualization example, so evaluation is one copy-paste away.
26. As a user, I want a partial dependence plot for a chosen feature, so I can see the model's average response along it.
27. As a user, I want plotting to stay optional (matplotlib as a dev extra; the library returns data only), so the core install stays light.
28. As a docs reader, I want the API reference split into **Data**, **Model**, and **Metrics** sections, so I can find every class and function by layer.
29. As a docs reader, I want each API section to cover the full surface — every constructor parameter, method, and return shape — so the reference is authoritative.
30. As a docs reader, I want the new attribution and curve APIs shown with runnable examples and real output, so I can verify behavior before adopting it.
31. As a package user, I want unfitted models to refuse explanation calls with a clear error, so mistakes surface immediately.
32. As a package user, I want attribution and curve calls to accept the same row/target representations as fit/predict (including missing cells), so nothing needs reshaping.
33. As a maintainer, I want behavioral invariants (telescoping, Shapley efficiency, curve sanity) pinned by tests, so future solver changes cannot silently break explanations.
34. As a maintainer, I want the Shapley implementation grounded in the canonical tree-Shapley algorithm (as implemented by the xgboost/shap reference), so our numbers agree with the ecosystem rather than folklore.

## Implementation Decisions

- **Break-down semantics** (API contract, from design):

  ```
  break_down(row) -> { baseline: float, steps: [(column_name, contribution) ...],
                       residual: float, prediction: float }
  invariant: baseline + sum(contributions) + residual == prediction
  ```

  Baseline = training-target mean; each split on the row's route contributes
  (child node mean − parent node mean) to the split's column, in path order;
  residual = the within-leaf model's adjustment (identically zero for mean
  leaves). Routing uses the exact prediction routing rule (inclusive threshold
  comparison; missing values follow the split's learned missing direction).

- **Shapley**: exact path-dependent tree Shapley per the canonical algorithm implemented in xgboost's tree-shap module (extraction in progress; formulas — path extension fractions from node cover counts, per-leaf phi updates, unwinding weights — will follow that source). Base value is the cover-weighted expected model output. Efficiency (base + Σφ = f(x)) is the acceptance invariant. Complexity target O(leaves · depth²) per row.

- **Gain importance** (landed): each split carries its goodness-of-split gain; feature importance = each column's share of total split gain (rpart's variable importance convention), shares sum to 1, sorted descending.

- **Permutation importance**: per-column score degradation on provided rows when the column is shuffled, deterministic seed, report `(column, degradation)` with the same sorted convention.

- **Curves** live in the core metrics module alongside the existing regression/classification/grouped reports, following the existing "report structure returned, Python dict mapping" pattern. Algorithms: rank-based sweeps over scores sorted descending; positive class = 1; AUC by trapezoid over the empirical ROC; average precision by step-wise precision sum (sklearn convention); calibration by equal-width bins over [0, 1] with edge-bin clipping and empty bins omitted; cumulative gain from the same sweep (population fraction vs capture rate). Ties handled by stable ordering.

- **Metrics API additions**: `calibration(y_true, scores, n_bins=...)`, `roc_curve(y_true, scores)`, `pr_curve(y_true, scores)`, `gain_curve(y_true, scores)` — returning plain structures of numeric series + summary scalars (auc, average_precision). Hard-label classification reporting stays as is.

- **Plotting** stays outside the library: the examples/docs layer renders figures with matplotlib (guarded dev extra), tree diagrams continue to use graphviz via pygraphviz from DOT output. New figure types: break-down waterfall, importance bars (gain/Shapley), calibration panel, ROC/PR panel, partial dependence.

- **API reference**: restructured into three pages — **Data** (encoders, scalers, builders, index mapping, winnowing, dataset helpers, feature transforms), **Model** (`ModelTree` parameters, fit/predict semantics, constraints, inspection, attribution), **Metrics** (regression/classification/grouped reports, curves) — covering the full surface including everything added here. Navigation and the llmstxt page list updated; the old combined page is replaced.

- **Conventions honored** (repo law): Rust core is the source of truth for all algorithms; the Python layer mirrors; missing data is first-class end to end (None/NaN semantics unchanged); outputs are deterministic (no wall-clock or unordered iteration); no heavy dependencies in the core.

## Testing Decisions

- A good test pins **external behavior and invariants only** — never internal structure, iteration order of internals, or solver plumbing (repo rule: "tests are behavior, not wiring").
- **Two seams** (confirmed with the user, matching the repo's standing rule that every capability lands with Rust core tests *and* Python API pytest):
  1. **Rust core integration tests** — behavioral invariants: break-down telescoping on mean and linear leaves (including rows with missing cells); Shapley efficiency and symmetry on a symmetric tree; zero contribution for unused columns; curve correctness on deterministic fixtures (AUC = 1.0 for perfect ranking, 0.0 for reversed; average precision = 1.0 for a perfect ranker; calibration bins of a perfect forecaster have observed rate ≈ mean score); permutation importance ranks the signal column first.
  2. **Python API pytest** — user-visible surface: return shapes and key names, column-name attribution, sorted-output conventions, unfitted refusal, missing-value rows accepted identically to predict.
- Prior art to follow: the existing model-tree behavior tests (gold-bytes printer test, importance-ranking test, monotone/interaction property tests, DOT export test) and the existing Python surface tests.
- **No pixel/image tests** for plots — figures are verified by running the example and inspecting rendered output (existing repo practice); the tested contract is the numeric data the plots consume.

## Out of Scope

- Logistic/GLM leaf models and the leaf-solver overhaul (IRLS/Newton, improved coordinate descent) — separate spec; the statsmodels internals extraction is already in flight for it.
- Cost-complexity pruning, cross-validation tables, C5-style rules export, model serialization, prediction intervals, forests/ensembles — roadmap items, each its own spec.
- Calibration *recalibration* (Platt/isotonic fitting) — only measuring calibration is in scope.
- Interactive/JS charting; static figures and data-returning APIs only.
- Elo benchmark/arena changes.

## Further Notes

- Two research extractions are feeding this work: the canonical tree-Shapley algorithm (xgboost implementation) grounds the Shapley decision above; the statsmodels extraction grounds the follow-up solver spec.
- Gain-share importance already ships (split gains + importance accessor); this spec extends from that landed state.
- The repo rule "one multi-core job at a time" applies to test/example execution schedules while the benchmark arena runs.
