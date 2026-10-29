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
