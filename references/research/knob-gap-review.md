# Knob-surface gap review — rpart / ctree / Cubist / C5.0 / sklearn vs ModelTree

Survey completed 2026-10-29 (first pass by the assistant from cloned sources
in `references/repos/`; codex cross-check queued for after its usage window
opens). **Defaults rebake explicitly deferred until this inventory is
consumed.**

## 1. Merged knob table by theme

| theme | rpart | ctree (partykit) | Cubist | C5.0 | sklearn tree | **ModelTree (ours)** |
|---|---|---|---|---|---|---|
| split criterion | anova/entropy/gini/poisson (`method`, `parms`) | teststat quadratic/maximum + `splitflavour`, `splitstat` | M5-style (SDR gain) | gain ratio | squared_error/absolute_error/poisson; gini/entropy/log_loss | fitted-leaf SSE gain (mean or linear leaves) |
| split search | exhaustive per feature | `mtry`, `maxvar`, `splittry`, `lookahead`, `intersplit`, `stump`, `multiway` | exhaustive | exhaustive | `splitter` best/random, `max_features` | exhaustive (+ rayon parallel scan, deterministic ties) |
| min node size | `minsplit` 20, `minbucket` ≈ 7 | `minsplit`, `minbucket`, `minprob` | internal min cases | `minCases` 2 | `min_samples_split` 2, `min_samples_leaf` 1, `min_weight_fraction_leaf` | `min_rows` 10 (count-based) |
| growth bound | `maxdepth` 30 | `maxdepth` | none (rules=100 bounds) | none | **unbounded** + `max_leaf_nodes` (leaf cap) | **`max_depth` 4 (hard cap)** |
| pruning | **`cp` 0.01 + `xval` 10 (cptable/prune)** | `mincriterion` significance gate, `alpha` | rule pruning, `unbiased` | `CF` 0.25, `noGlobalPruning`, `earlyStopping` | **`ccp_alpha` + cost_complexity_pruning_path** | `min_gain` only — **no pruner** |
| randomness | — | `nresample` (MonteCarlo tests) | `sample` | `sample`, boosting `trials` | `random_state`, `splitter=random`, `max_features` | deterministic by design |
| ensembles/instance | — | — | **`committees` 20, `neighbors` 5, `composite`** | `trials` boosting, `rules` rulesets | (forests live elsewhere) | not yet (ModelForest roadmap) |
| constraints | — | — | — | — | **`monotonic_cst`** | **`monotone` + `interaction_groups`** |
| missing data | **surrogate splits** (`usesurrogate`, `maxsurrogate`) | `MIA` (missing in attributes), `maxsurrogate`, `numsurrogate` | na.fail-ish | NA branch handling | native routing | **learned directions + NA-indicator leaf terms** |
| weights | `weights`, `cost` (split costs), `parms` (loss matrix) | `caseweights`, `majority` | — | `costs` (cost matrix) | `class_weight`, `min_weight_fraction_leaf` | `fit(w)` row weights |
| output/explain | cptable, `model`, `y` retention | `saveinfo`, `cores`, `applyfun`, ordinal `scores` | rules + committees + composite print | rulesets, `bands` | `export_text/graphviz`, pruning path | rpart-format print, `to_dot`, gain importance, break_down/Shapley |

## 2. Gaps that matter

1. **Cost-complexity pruning + cross-validation** (rpart `cp`/`xval`, sklearn
   `ccp_alpha`) — the biggest classical gap; also the missing companion to a
   deep grow phase.
2. **Randomized subspaces / `mtry`** (rpart-ish `mtry`, sklearn
   `max_features`/`splitter=random`) — cheap; synergy with future forests.
3. **Committees + neighbors** (Cubist: boosting committees + kNN residual
   adjustment + composite blend) — the board's committee multiplier argues
   for it; aligns with the ModelForest roadmap.
4. **Statistical split tests** (ctree `teststat`/`testtype`/`mincriterion`,
   Bonferroni/multiplicity) — significance-gated splits vs pure gain.
5. **C5.0 knob family** — `CF`, `fuzzyThreshold`, `noGlobalPruning`,
   `earlyStopping`, `bands`, boosting `trials`, rulesets (ties to the pending
   C5-style rules printer).
6. **Cost matrices** (rpart `cost`/`parms`, C5.0 `costs`) — cost-sensitive
   splits/decisions.
7. **Weight-aware leaf bounds** (sklearn `min_weight_fraction_leaf`) — our
   `min_rows` counts rows only.

## 3. What we have that they don't (keep)

- `monotone` **and** `interaction_groups` (only sklearn has monotonic_cst, and
  nobody has interaction groups).
- Winnowing as a first-class API (`winnow()` + `min_relevance`); C5.0 buries
  `winnow` inside fitting.
- Learned missing directions + NA-indicator leaf terms (lighter than rpart's
  surrogate splits; ctree's MIA is the other modern answer).
- Ridge/elastic leaf regularization (`l1`/`l2`) — nobody exposes this.
- Attribution: `break_down` (telescoping contract) + gain importance +
  (Shapley in flight); DOT/graphviz output.
- Deterministic parallel scan with pinned tie-breaking (`cores` is the
  nearest ctree analog, but no determinism contract).

## Appendix: complete knob inventories (2026-10-29)

### sklearn tree (grounded via get_params + docs)

- criterion: {"squared_error","absolute_error","poisson"} (friedman_mse
  deprecated); classifier adds {"gini","entropy","log_loss"}
- splitter: {"best","random"}; random_state
- max_depth: None = unlimited (bounded only by leaf-size knobs)
- min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0
- max_features: int|float|"sqrt"|"log2"|None — random subspace per split
- max_leaf_nodes: None — best-first growth with a LEAF CAP (the alternative to
  a depth ceiling; LightGBM bounds the same way via num_leaves=31)
- min_impurity_decrease=0.0 (== our min_gain)
- ccp_alpha=0.0 + cost_complexity_pruning_path() — weakest-link pruning as a
  path API (vs rpart's cp table + xval; two reference semantics)
- monotonic_cst: +1/-1/0 per feature (sklearn >= 1.4)
- class_weight (classifier)

### R fit functions (knobs beyond the control objects)

- rpart(): method (anova/class/...), parms (loss matrix, split priors), cost
  (per-split cost vector), weights, subset, model/x/y retention
- ctree_control: teststat, splitstat, splittest, testtype, pargs, nmax, alpha,
  mincriterion, logmincriterion, minsplit, minbucket, minprob, stump, maxvar,
  lookahead, **MIA** (missing incorporated in attributes), nresample, tol,
  maxsurrogate, numsurrogate, mtry, maxdepth, multiway, splittry, intersplit,
  majority, caseweights, applyfun, **cores**, saveinfo, update, splitflavour
- ctree(): ytrafo, scores (ordinal), converged, cluster, doFit
- cubist(): committees (default 20), neighbors (default 5) on the fit call;
  cubistControl: unbiased, rules (100), extrapolation, sample, seed, label,
  strip_time_stamps
- C5.0(): trials, rules, costs (cost matrix); C5.0Control: subset, bands,
  winnow, noGlobalPruning, CF (0.25), minCases (2), fuzzyThreshold, sample,
  seed, earlyStopping, label

### Default-depth question (recorded, NOT actioned)

sklearn unlimited / LightGBM unlimited (num_leaves=31) / rpart 30 + pruning /
Cubist & C5.0 uncapped / **ours 4 (hard)**. Our own diamonds tour shows depth
12 + min_rows 500 beats depth 8 + min_rows 50 (555.6 vs 577.0): depth is not
the villain, leaf size is. Re-bake of defaults is **deferred by request** —
decide after the pruning + leaf-cap knobs land.

---

# Appendix 2: MARS (R earth), CatBoost, grf knobs (2026-10-29)

## MARS — earth (earth.fit(), since earth() is a ... wrapper)

x, y, weights, wp, subset, na.action, offset, **pmethod** ("backward"|
"none"|"seqrep"|"exhaustive"), keepxy, trace, **glm** (GLM family!), degree,
**penalty**, **nk** (max terms), thresh, minspan, endspan, newvar.penalty,
fast.k, fast.beta, **linpreds** (force terms linear), **allowed**
(interaction whitelist), **nprune**, Object, Scale.y, Adjust.endspan,
Auto.linpreds, Force.weights, Use.beta.cache, Force.xtx.prune, Get.leverages,
Exhaustive.tol.

Notes: pmethod chooses the pruning SEARCH; nk/nprune are the term budgets;
minspan/endspan control knot spacing; weights AND wp (per-column weights) are
first-class. This validates the MarsRegressor v1 knob set in
references/research/mars-port-plan.md (degree, max_terms=nk,
max_pruned_terms=nprune, penalty, thresh, minspan, endspan).

## CatBoost (resolved defaults from a fitted model)

iterations 1000, learning_rate auto (~0.022), depth 6, l2_leaf_reg 3,
random_strength 1, bagging_temperature (Bayesian), border_count 254,
grow_policy SymmetricTree (| Depthwise | Lossguide), min_data_in_leaf 1,
max_leaves 64, one_hot_max_size, od_type/od_wait (overfitting detector),
sampling_frequency PerTree, max_ctr_complexity (categorical combinations),
boosting_type Plain, leaf_estimation_iterations 1, score_function, langevin,
posterior_sampling.

Notes: border_count=254 is the binned-scan world (our roadmap item);
grow_policy/max_leaves is the leaf-cap philosophy again; max_ctr_complexity =
the categorical-combinations trick our preselect roadmap targets; od_type =
cheap overfitting detector worth stealing.

## grf (regression_forest; ll_regression_forest adds the ll-split family)

X, Y, num.trees, sample.weights, clusters, equalize.cluster.weights,
sample.fraction, mtry, min.node.size, **honesty, honesty.fraction,
honesty.prune.leaves**, alpha, imbalance.penalty, ci.group.size,
**tune.parameters** (+ tune.num.trees/reps/draws), compute.oob.predictions,
num.threads, seed. Local-linear extras: enable.ll.split,
ll.split.weight.penalty, ll.split.lambda, ll.split.variables, ll.split.cutoff.

Notes: the honesty triple = sample-splitting fairness (needed for our future
forest); alpha/imbalance.penalty = imbalance-aware splitting (weight-aware);
ll.split.* is the mechanism behind grf_llf's board lead (1576.5) — the
local-linear split criterion is a candidate upgrade for our linear leaves.

---

# Knob coverage check (2026-10-30 — pre-commit gate)

Every knob in the inventories above, mapped to the ModelTree surface as of
the module split. Legend: **BUILT** = ships today; **#n** = tracked in the
spec; **ROADMAP** = forests/boosting/MARS phase; **SKIP** = deliberate
non-goal.

| family | reference knobs | ours | status |
|---|---|---|---|
| split criterion | rpart `method`/`parms`, ctree `teststat`/`splitstat`/`splitflavour`, C5 gain ratio, sklearn `criterion`, M5 SDR | `gain`: sse, friedman_mse, absolute_error, huber, poisson, entropy, gini, gain_ratio, sdr, twoing + callable | **BUILT** |
| split search | sklearn `splitter`, ctree `mtry`/`maxvar`/`splittry`/`lookahead`/`intersplit` | exhaustive scan, `split_precision` grid, `col_sample`+`seed` | **BUILT**; splittry/lookahead/intersplit = **SKIP** (partykit search heuristics) |
| categorical | rpart/C4.5 level ordering, CatBoost `max_ctr_complexity` | `cat_order`, `cat_search`, levels-as-is splits | **BUILT**; combinations → **#3** EFB |
| min node size | rpart `minsplit`/`minbucket`, ctree `minsplit`/`minbucket`/`minprob`, sklearn `min_samples_*`/`min_weight_fraction_leaf`, grf `min.node.size`, CatBoost `min_data_in_leaf` | `min_rows` (count) | PARTIAL → **#3** minsplit/minbucket/minprob |
| growth bound | rpart `maxdepth`, sklearn `max_depth` + **`max_leaf_nodes`**, CatBoost `max_leaves`/`grow_policy`, LightGBM `num_leaves` | `max_depth` | **GAP FOUND: leaf-cap untracked** → added to **#3** (2026-10-30) |
| pruning | rpart `cp`+cptable+`xval`, sklearn `ccp_alpha`+path, C5 `CF`/`noGlobalPruning`/`earlyStopping`, cubist rule pruning | `prune` none/bic/mdl (encoding cost), `min_gain` | **#3** pruning family (cp/xval-vs-ccp_alpha decision, C5 pessimistic, cost matrix) |
| significance stopping | ctree `mincriterion`/`alpha`/`testtype`, `nresample` | `split_test=chow`, `split_alpha`, `bonferroni` | **BUILT** (nresample/MonteCarlo = **SKIP**) |
| randomness | ctree `nresample`, cubist/C5 `sample`, sklearn `random_state`/`splitter=random` | `seed`, `col_sample` (deterministic) | row subsampling → **ROADMAP** forests; splitter=random + sample = **SKIP** |
| ensembles / instance | cubist `committees`/`neighbors`/`composite`, C5 `trials`/`rules` | — | **ROADMAP** #5 (forests & committees), #6 (rulesets) |
| constraints | sklearn `monotonic_cst` | `monotone` + `interaction_groups` | **BUILT** (lead) |
| missing data | rpart `usesurrogate`/`maxsurrogate`, ctree **MIA**/`maxsurrogate`/`numsurrogate` | learned `missing_left` + NA-indicator leaf terms | PARTIAL → **#3** NA surrogates (count knob lands there; MIA-style handling is already ours) |
| weights | rpart `weights`/`cost`/`parms` loss, ctree `caseweights`/`majority`, C5 `costs`, sklearn `class_weight` | `w` first-class | cost/loss matrix → **#3** cost-matrix; rpart per-variable `cost` = **SKIP** |
| leaf model | cubist smoothing, MARS `glm`/`linpreds`/`Scale.y` | `leaf` linear/mean/median/quantile/huber, `l1`/`l2`, `tau` | **#4** leaf_family (binomial/poisson → probit/tobit/tweedie/softmax, `predict_proba`) |
| boosting / calibration | CatBoost `od_type`/`od_wait`, sklearn `early_stopping` | — | **ROADMAP** (od-style detector noted) |
| MARS budgets | earth `pmethod`/`nk`/`nprune`/`penalty`/`thresh`/`minspan`/`endspan`/`newvar.penalty`/`fast.k`/`fast.beta`/`allowed`/`linpreds` | MarsRegressor v1 set (validated vs `earth.fit` formals) | **ROADMAP** MARS port (oracle captured) |
| binned scan | CatBoost `border_count`/`grow_policy`, LightGBM histograms | `split_precision` grid (approximation) | **ROADMAP** (EFB + binned-scan notes) |
| output / explain | rpart cptable/print, C5 `bands`, sklearn `export_text`/graphviz | rpart-pinned print, `to_dot`, `feature_importances`, `break_down`, `training_log`, `leaf_digest`+`quantile` | `bands`/intervals → **#5** predict API |
| local-linear splits | grf `ll.split.*` family | — | **ROADMAP** (candidate criterion upgrade — the grf_llf 1576.5 mechanism) |

Findings:
1. **One untracked gap: leaf-cap (`max_leaf_nodes`-style).** The
   default-depth decision above already requires it ("decide after the
   pruning + leaf-cap knobs land") yet issue #3's knob list omitted it —
   now added there. Without it the defaults rebake has only depth to vary.
2. Everything else maps to BUILT / a spec / a deliberate SKIP — the
   coverage is complete.
3. The main table's "ModelTree (ours)" column is a 2026-10-29 snapshot
   ("no pruner" etc.); this section supersedes it for current state.
