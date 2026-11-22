# Forest Methods Landscape and the `ModelForest` Unified API

**Date:** 2026-11-22
**Scope:** Random forests, ExtraTrees, ranger-compatible mechanics, Generalized Random Forests and honest splitting, Cubist/M5 model-tree committees, gradient boosting, forest-based interpretability, competing library API surfaces, and modern tabular challengers.
**Decision informed:** Which forest kinds belong in one `ModelForest` facade, how Cubist-style committees differ from GRF-style forests from ordinary random forests, and what the shared API contract must guarantee.
**Confidence:** High for classical mechanics, repository fit, and API surfaces. Medium for recent benchmark orderings. Explicitly marked gaps where cited papers could not be verified.
**Code changes:** None. Research and specification only.

**Implementation spec:** `specs/28-model-forest-unified-api.md` holds the `ModelForest` facade, the trait table, the kind-specific mechanics, the Cubist-versus-GRF-versus-random-forest differentiation, the build sequence, and the phase gates. It supersedes the scope of Specs 14 and 15.

## 1. Executive Summary

`mltreelib` has a mature single-tree core. `crates/mltreelib-core/src/model_tree.rs` already provides CART growth, linear leaves with `l1`/`l2` penalties, monotone constraints on splits and leaf coefficients, interaction groups, per-node column subsampling (`col_sample`), a seeded RNG, `honesty` sample splitting, `alpha` child-balance, `imbalance_penalty`, categorical partition search, missing-value routing, and `predict_adjusted` for k-NN residual correction.

What it lacks is the ensemble layer. There is no `ensemble` module. No bagging, no boosting, no committee, no forest weights, no OOB machinery.

This note surveyed four method families and produced one conclusion:

> **A forest is not one algorithm. It is a shared structure — a collection of trees, a sampling policy, an aggregation rule, and a leaf estimator — with four distinct kinds that share that structure but differ in what they are for.**

The four kinds, and what separates them:

| Kind | Sampling | Trees grown | Leaf estimator | Aggregation | Inference |
|---|---|---|---|---|---|
| **Random forest** (`rf`) | Bootstrap with replacement | Independent, in parallel | Constant mean or class distribution | Plain average | None. OOB error is a diagnostic with known optimism bias |
| **ExtraTrees** (`extra_trees`) | Full sample or bootstrap | Independent, in parallel | Constant | Plain average | None |
| **Cubist committee** (`committee`) | Full sample, case-weighted | **Sequential**, each on relabelled targets | Ridge/linear model | Unweighted average of members | None |
| **GRF forest** (`grf`) | Subsample without replacement, honest halves | Independent, in parallel | Local estimating equation or local linear model | Forest weights solve a local equation | Asymptotic only, with conditions |

The critical distinctions, stated up front because everything else depends on them:

1. **Cubist committee is not boosting.** Members are trained sequentially on relabelled targets $y' = 2y - \hat y_{m-1}$, but the final prediction is an **unweighted average**, not a shrunk sum. There is no learning rate and no row resampling. The CRAN `Cubist` vignette states this explicitly: "Unlike traditional boosting, stage weights for each committee are not used to average the predictions from each model tree; the final prediction is a simple average of the predictions from each model tree."

2. **GRF is not a random forest with a better split rule.** In GRF the tree predictions are never averaged. The forest defines weights $\alpha_i(x)$ over training points, and the estimator is the solution of a **weighted estimating equation**. For least squares this happens to be algebraically identical to Breiman's average. For anything else it is not. That is the entire conceptual content of GRF.

3. **Honesty is orthogonal to doubly-robustness.** Honesty fixes *adaptive bias* from choosing splits on the same outcomes that are averaged into leaves. Double robustness fixes *confounding bias*. Neither substitutes for the other. `grf` uses both. They must be independent switches in the API.

4. **The `m0` smoothing constant in existing repository material is a mis-statement.** No `m0` or `M0` identifier exists anywhere in the Cubist GPL source. Two distinct smoothing algorithms exist and must be named separately: Quinlan's M5 backup smoothing $p' = (n p + k q)/(n+k)$ with $k = 15$, and Cubist's variance-minimising coefficient shrink against the parent model.

5. **No mature published API unifies RF, ExtraTrees, GBDT, and Cubist under one `kind` switch.** Precedent exists for generic `fit`/`predict`, trait-based capability dispatch (Julia MLJ), and three-axis specialization (tidymodels `parsnip`). The right design combines those three patterns.

## 2. Current Repository State

### 2.1 What already exists and is reusable

`crates/mltreelib-core/src/model_tree.rs`, `ModelTreeParams`:

- `honesty: bool` and `honesty_fraction: f64` (default `false`, `0.5`). Seeded partition of rows into structure rows (drive split search, gate, pruning) and estimation rows (fit leaf models). A node whose estimation rows fall under `min_rows` falls back to structure rows, and the training log records it.
- `col_sample: Option<f64>` — per-node random column subset, the random-forest `mtry` mechanism.
- `alpha: f64` — grf-style minimum child weight fraction.
- `imbalance_penalty: f64` — grf-style `imbalance_penalty * (1/n_left + 1/n_right)` gain subtraction.
- `tree_monotone`, `leaf_monotone`, `monotone` — split-direction and leaf-coefficient-direction constraints.
- `interaction_groups: Vec<Vec<usize>>` — LightGBM-style feature groups.
- `l1`, `l2` — leaf coefficient penalties (intercept unpenalized).
- `seed: u64` — deterministic per-node column subset selection.
- `predict_adjusted` — k-NN residual correction path, the hook Cubist's composite model needs.
- `LeafKind` — `Linear`, `Mean`, `Median`, `Quantile(f64)`, `Huber`, `Mars`, `Prob`, `Odds`, `LogOdds`, `Knn`, plus family-parameterised variants.
- `LeafSolver` — `Irls`, `Lbfgs`, `Ccd`, `CmaEs`, `Shade`, `NaturalGradient`, `GeneticAlgorithm`.

`crates/mltreelib-core/src/cluster/urf.rs` has `UnsupervisedProximityForest` with `n_trees`, `max_features`, per-tree `ProximityTree`, and leaf-assignment matrices. The data structures for a forest already exist once, in the clustering module, and are not shared with anything else.

`crates/mltreelib-core/src/anomaly/` has `aad_tree.rs`, `context_tree.rs`, `det.rs`, `eif_tree.rs`, `gritbot.rs`, `hs_tree.rs`, `rc_tree.rs`. These are anomaly engines, not forests.

### 2.2 What does not exist

- No `ensemble` module.
- No bootstrap or subsample sampler.
- No per-tree seed derivation, so no reproducibility contract for parallel tree building.
- No in-bag count tracking.
- No OOB prediction or OOB error.
- No committee or boosting loop.
- No smoothing constant, no SD-based SDR split rule, no extrapolation clipping.
- No forest weights, no local estimating-equation solver, no infinitesimal jackknife, no bootstrap of little bags.
- No GRF gradient-based splitting rule.
- No `ModelForest` facade.

### 2.3 Existing repository research that this note extends

- `references/research/2026-11-17-ranger-and-grf-forest-mechanisms.md` — ranger internals, probability forests, Fisher target-ordered categorical splits, GRF kernel weights, honest partitioning, gradient-based splitting. This note supersedes its speculative parts and corrects its unattributed claims.
- `references/research/ours-vs-c5-cubist.md` — structural comparison against C5.0 and Cubist, with the RuleQuest Unix tutorial as primary evidence.
- `references/research/model-tree-survey.md` — M5/M5P/Cubist survey.
- `references/research/2026-11-19-comprehensive-splitting-criteria-survey.md` — split-rule catalogue including GRF gradient splitting.
- `specs/14-cubist-committees-and-model-tree-forests.md` — Cubist committees and LinearForest.
- `specs/15-structured-tree-ensembles.md` — honest forests, ModelTreeBoost, stacking.

## 3. Source-Correction Audit

Several attributions in common secondary sources, and in this repository's earlier material, do not survive verification. They are listed so they are not carried forward.

| Attributed claim | Verification | Use instead |
|---|---|---|
| Wager (2021), "Estimation using the adaptive forest", JRSS-B | Does not exist. Not in Crossref, OpenAlex, arXiv, or Wager's own publication list. The phrase "adaptive forest" occurs zero times in the 8 Sep 2026 draft of Wager's causal inference book. | GRF §2 (adaptive neighborhood metric), Wager & Walther (2015) on adaptive-forest consistency, Huang, Khowaja & Härdle (2021, 2022) on uniform inference |
| "An honest assessment of honesty" (Athey, Wager & Wilks 2019) | Does not exist under that title anywhere. | Hou & Fernández-Loría (ISR 2026, arXiv:2506.13107); arXiv:2509.11381; Athey & Imbens (2016) |
| Xu (2021), "Empirical Monte Carlo approximation to the error distribution of the adaptive forest" | Does not exist under that title. | Wager, Hastie & Efron (2014) JMLR 15:1625–1651; Zhou, Athey & Wager (2022) on the infinitesimal jackknife and the Ranger's correction |
| Xu, Athey & Wager (2021), "Generalized random forests for survival" | Does not exist under that title. | Cui, Kosorok, Sverdrup, Wager & Zhu (2023) JRSS-B; Xu, Ignatiadis, Sverdrup, Fleming, Wager & Shah (2023) handbook chapter |
| "GRF, AoS 47(2) 1148–1178" attributed to Athey & Wager | That paper is Athey, Tibshirani & Wager (2019). Athey & Wager's causal-forest paper is *Observational Studies* 5:36–51 (2019). | Keep the attribution correct |
| Local linear forest attributed to Athey, Tibshirani & Wager | Local linear forest is Friedberg, Tibshirani, Athey & Wager, *JCGS* 30(2), 2021 | — |
| "The `grf` R/Python package" | `grf-labs/grf` ships an R package and a C++ core only. Repo root has no Python directory. Upstream issue #257 "Any python implementations?" is open. The closest Python wrapper is third-party `skgrf`. | Document the R/C++ reality |
| "Pang et al. 2016" for Mondrian forests | No matching canonical paper. | Lakshminarayanan, Roy & Teh (2014) arXiv:1406.2673; (2016) PMLR v51 |
| "Khanna, Liu & Loh 2019" for oblivious RF | Exact paper not found. | Liu & Loh (2019), DOI 10.3390/app9050898, on candidate set size in an oblivious equi-depth setting; Popov et al. arXiv:1909.06312 for ODT mechanics |
| "Greenwell 2017, Obstructed Trees" | No primary source with that title. Greenwell 2017 is about partial-dependence plots. | Greenwell 2017 DOI 10.32614/RJ-2017-016; Greenwell, Boehmke & McCarthy 2018 arXiv:1805.04755; Loecher 2023 DOI 10.1007/s10182-023-00479-7 |
| "Louppe et al. 2014, A note on the bias of OOB estimates" | No standalone paper with that title. | Louppe 2014 dissertation arXiv:1407.7502; Janitza & Hornung 2018 DOI 10.1371/journal.pone.0201904; Mitchell 2011 DOI 10.4236/ojs.2011.13024 |
| "Mallick et al. 2024, Why do tree-based models still outperform deep learning: A review" | Not verified. | Grinsztajn et al. 2022; McElfresh et al. 2023; TabArena 2025 |
| "Ishwaran, Kagal & Lankford 2008" | Wrong author list. | Ishwaran, Kogalur, Blackstone & Lauer 2008, DOI 10.1214/08-AOAS169 |
| "Brezinski, Friedman & Sperga (2004) adaptive gradient boosting" | No resolvable record in Crossref, OpenAlex, or arXiv. The NGBoost paper contains no such citation. | Exclude until the original is located. For "adaptive" boosting semantics use Freund & Schapire (1997) |
| "Yin et al. 2019 / Chen et al. 2019" monotone-constraint GBDT papers | Not verifiable. arXiv 1905.10197 resolves to an unrelated mathematics paper. | XGBoost monotone tutorial; LightGBM `monotone_constraints` docs; Koklev, arXiv:2512.17945 (multi-dataset benchmark) |
| "Lubliner et al. 2019" for EBM | Not verified. arXiv 1909.12965 resolves to an unrelated chemistry paper. | Nori et al. 2019, arXiv:1909.09223; https://interpret.ml/docs/ |
| NGBoost attributed to "Zhao et al. 2021" | Wrong canonical attribution. | Duan, Avati, Ding, Thai, Basu & Ng, arXiv:1910.03225, PMLR v119 |
| Bühlmann & Yu (2003) DOI ending `000109` | Does not resolve. | DOI 10.1198/016214503000125 |
| `grf` / `EconML` written in Rust | False. `econml`'s forest core is Cython (`econml/grf/_criterion.pyx`, `econml/tree/_tree.pyx`, `_splitter.pyx`). No `Cargo.toml`, `rust/`, or `_rust/` in the repo tree. | Semantics matter, not code. This is good news for a Rust port |
| CRAN `Cubist` maintained by David Pawley | Not listed as maintainer or author. CRAN lists Max Kuhn; credits include Kuhn, Weston, Keefer, Coulter, Quinlan, RuleQuest Research. | https://cran.r-project.org/package=Cubist, https://github.com/topepo/Cubist |
| `grf` `sample.size`, `imbalance.limit`, `num.oob.trees`, `debiased`, `ortho.estimator`, `ortho`, `clustering` exist as arguments | None of these exist in the official API. | Use the verified `causal_forest()` signature |

Two further hygiene notes. The arXiv mirror reachable from this environment returns different papers for several ids (1710.03020 resolves to a neutrino-astronomy paper, 1905.10197 to a mathematics paper on Seshadri constants, 1909.12965 to a chemistry paper). Every id below was re-checked by title and authors. And a separate warning: `references/research/2026-11-17-ranger-and-grf-forest-mechanisms.md` attributes the Fisher target-ordered categorical split theorem to "Fisher (1958)"; the actual result is Fisher (1958) for the *discriminant* ordering and the target-ordering argument is due to permutation-invariance work, so the attribution should be softened.

## 4. Random Forest Mechanics

### 4.1 Breiman (2001)

Training, per tree $b = 1, \ldots, B$:

1. Bootstrap: draw $n$ row indices with replacement from $1, \ldots, n$. The resulting multiset is $I_b$.
2. Grow a full, unpruned tree on rows indexed by $I_b$.
3. At each nonempty node, draw `mtry` predictors without replacement from the $p$ predictors. Evaluate all admissible CART splits on those candidates. Choose the split maximising local impurity reduction — Gini for classification, response variance reduction for regression.
4. Partition and recurse. Stop when the node is empty, a predictor is constant, the response is constant, or the minimum-node-size rule fires.
5. Store the tree.

No post-growth pruning is part of Breiman RF. Prediction: classification returns the plurality class across trees; regression returns the arithmetic mean of tree predictions; a probability forest averages per-tree class-probability vectors.

The error bound depends on tree **strength** $s$ (expected classification margin of one tree) and **correlation** $\rho$ (residual-margin correlation between trees):

$$PE^*(\text{forest}) \le \rho \frac{1-s}{s^2} \quad \text{(binary classification)}, \qquad PE^*(\text{forest}) \le \rho\, PE^*(\text{tree}) \quad \text{(regression)}.$$

Raising `mtry` raises strength but also correlation; lowering it does the reverse. That trade-off is the whole reason `mtry` is the most consequential RF hyperparameter.

Sources: Breiman (2001) DOI 10.1023/A:1010933404324; Breiman & Cutler, *Using Random Forests v4.0*, https://www.stat.berkeley.edu/~breiman/Using_random_forests_v4.0.pdf.

### 4.2 Out-of-bag estimation, and its reliability

For a bootstrap of size $n$,

$$P(i \text{ absent from one bootstrap}) = (1 - 1/n)^n \to e^{-1} \approx 0.368.$$

Each row is OOB for roughly 36.8% of trees in a large forest. The OOB prediction aggregates the trees in which row $i$ was absent; the OOB error compares that to the observed response.

**Calling OOB error "unbiased" is too strong, and this must not be written into the API docs as fact.**

- Finite-forest OOB error tends to exceed full-forest error, because ensemble error falls as trees are added and each OOB prediction averages only about 36.8% of the trees.
- Janitza & Hornung (2018) found **parameter-dependent overestimation** of the true error, worst with balanced classes, small $n$, many predictors, weak predictor-response effects, and low predictor correlation. Because the overestimation depends on `mtry`, selecting `mtry` by OOB error inherits the bias.
- Mitchell (2011) independently demonstrated OOB overestimation in small-$n$, large-$p$ settings.

Practical consequence: expose OOB error as a diagnostic and a cheap tuning metric. Use repeated cross-validation or a holdout for final performance claims. The API report must say so.

Sources: Breiman (2001) OOB estimation note, https://www.stat.berkeley.edu/~breiman/OOBestimation.pdf; Janitza & Hornung (2018) DOI 10.1371/journal.pone.0201904; Mitchell (2011) DOI 10.4236/ojs.2011.13024.

### 4.3 Variable importance, and two distinct biases

**Mean decrease impurity (MDI).** Accumulate weighted impurity reduction per internal node, attributed to the selected predictor, averaged over trees. MDI favours predictors with many possible cut points, continuous scales, and high cardinality. Because it is computed on data already used to choose splits, noisy predictors receive non-zero importance from deep, nearly-pure trees.

**Permutation importance.** Permute predictor $j$ in OOB observations and re-score. Common variants: MDA (increase in error) and MDM (decrease in margin). Marginal permutation breaks the predictor-response association but, with correlated predictors, dilutes importance across substitutes or produces unstable rankings.

Strobl et al. (2007) identify **two** bias mechanisms, not one: biased split selection in CART (Gini prefers more categories or split points), and the bootstrap-with-replacement effect (even with unbiased trees, bootstrap sampling distorts selection frequency and permutation-importance distributions). Their remedy combines conditional-inference trees, whose selection is permutation-test based rather than impurity based, with subsampling **without replacement** at $0.632n$. Their reported conclusion is reliable variable selection even when predictors differ in scale or number of categories.

API consequence: keep MDI and permutation importance as separate, separately-named outputs. Label neither as causal importance.

Sources: Strobl et al. (2007) DOI 10.1186/1471-2105-8-25; Strobl et al. (2008) DOI 10.1186/1471-2105-9-307; Hooker, Mentch & Zhou (2021) DOI 10.1007/s11222-021-10057-z.

### 4.4 ExtraTrees

Geurts, Ernst & Wehenkel (2006) differ from Breiman RF in two ways: every tree sees the **complete original learning sample** without bootstrap replication, and split thresholds are randomised rather than optimised.

For node subset $S$: stop if $|S| < n_{\min}$, all predictors are constant, or the response is constant. Sample $K$ nonconstant predictors without replacement. For each, find the local min and max and draw $c_i \sim \mathrm{Uniform}[a_{i,\min}, a_{i,\max}]$, giving candidate split $a_i < c_i$. Score candidates by normalised mutual information for classification or relative variance reduction for regression. Choose the best and recurse. Aggregate by plurality vote or mean.

Original defaults: $K = \sqrt{p}$ rounded for classification, $K = p$ for regression; $n_{\min} = 2$ classification, $5$ regression; $M = 100$ trees in the experiments.

Published claims, from the paper: at suitable $K$, variance almost vanishes while bias rises only slightly relative to standard trees; beyond the optimal randomization, variance decreases only slightly while bias often rises sharply; over-randomisation can hide weakly relevant predictors. Across 24 numerical datasets (12 classification, 12 regression), the default $K$ was near-optimal on 22 datasets and only slightly suboptimal on two. ExtraTrees never lost a corrected paired $t$-test against the compared ensemble methods on classification; on regression it lost once against tuned Random Subspace and twice against tuned RF. Training was systematically faster than RF and tree bagging. ExtraTrees trees had roughly 1.5–3 times as many leaves, average depth at most about two levels deeper. Adding bootstrap sampling often reduced ExtraTrees accuracy in their experiments.

**ranger's `splitrule="extratrees"` is not original ExtraTrees.** ranger still bootstraps by default. To approximate original ExtraTrees in ranger you need `splitrule = "extratrees"`, `replace = FALSE`, `sample.fraction = 1`. The ranger documentation explicitly warns that changing `splitrule` does not disable bagging.

Sources: Geurts et al. (2006) DOI 10.1007/s10994-006-6226-1, full text https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf; ranger 0.18.0 manual https://cran.r-project.org/web/packages/ranger/ranger.pdf.

### 4.5 ranger-compatible mechanics and defaults

ranger 0.18.0 split rules:

| Task | Split rules | Default |
|---|---|---|
| Classification | `gini`, `extratrees`, `hellinger` | `gini` |
| Probability | `gini`, `extratrees`, `hellinger` | `gini` |
| Regression | `variance`, `extratrees`, `maxstat`, `beta`, `poisson` | `variance` |
| Survival | `logrank`, `extratrees`, `C`, `maxstat` | `logrank` |

ExtraTrees rule uses `num.random.splits` random thresholds per candidate predictor; default `1`. Other defaults: `alpha = 0.5`, `minprop = 0.1` for `maxstat`; `poisson.tau = 1`; `always.split.variables = NULL`; `split.select.weights = NULL`; `regularization.factor = 1`.

Key defaults relevant to a unified API:

| Parameter | ranger default | Note |
|---|---:|---|
| `num.trees` | 500 | |
| `mtry` | $\lfloor \sqrt{p} \rfloor$ | Same default for classification, probability, regression, and survival |
| `max.depth` | `NULL` or `0` | Unlimited; `1` means stump |
| `replace` | `TRUE` | Bootstrap with replacement |
| `sample.fraction` | `1` if `replace=TRUE`; `0.632` if `FALSE` | May be class-specific |
| `respect.unordered.factors` | `NULL` | Effective: `partition` for ExtraTrees, `ignore` otherwise |
| `min.node.size` | 1 class, 5 regression, 3 survival, 10 probability | Split eligibility, not guaranteed leaf size |
| `min.bucket` | 1 except survival 3 | Terminal-node size floor |
| `num.random.splits` | 1 | ExtraTrees rule |
| `importance` | `none` | Also `impurity`, `impurity_corrected`, `permutation` |
| `keep.inbag` | `FALSE` | Must be enabled to retain per-tree counts |
| `na.action` | `na.learn` | Native missing routing |

**Probability forest.** With `probability=TRUE` and a factor response, trees are grown with classification-style impurity, each tree returns a class-probability estimate, and the forest probability is the mean of tree probability vectors. OOB prediction error is then the **Brier score**, not misclassification rate. This is the ranger lineage for the probability-leaf recommendation in `2026-11-17-ranger-and-grf-forest-mechanisms.md` §5.1, and it should be the reference behaviour for `mltreelib`'s probability leaves.

**In-bag counts.** `inbag.counts` holds, for every tree $b$ and observation $i$, the number of times $i$ was drawn into tree $b$'s bootstrap sample. Count greater than zero means in-bag; zero means OOB; the counts for one tree sum to the bootstrap sample size. `keep.inbag=TRUE` stores them. Manually supplied `inbag` vectors permit stratified sampling. Per ranger maintainer note, class-specific `sample.fraction` values count against the total sample size, so equal class fractions balance minority classes.

**Unordered factors.** Three policies, and the choice matters more than any of the numerical defaults:

- `ignore`: the factor is treated as ordered. Ordinary threshold semantics impose an arbitrary ordering and can produce poor partitions.
- `order`: levels are reordered **once**, not per split — binary classification by proportion in the second class, regression by mean response, multiclass by the first principal component of the weighted class-proportion contingency table, survival by median survival or the largest available quantile. Each node then uses ordinary ordered splits. ranger recommends this because it is fast and handles unlimited category counts.
- `partition`: all two-way partitions of observed levels are considered. Avoids arbitrary order but costs more and becomes expensive at high cardinality. ExtraTrees default.

**Memory and runtime.** ranger's key choices: avoid copies of the original data; store compact node structures and release memory early; pre-sort feature values in large nodes and raw-value sort in small nodes; partial Fisher–Yates/Knuth sampling for candidate predictors. Paper benchmark, 10,000 observations, 150,000 SNPs, 1,000 trees, `mtry=5000`:

| Implementation | Runtime | Peak memory |
|---|---:|---:|
| randomForest | 101.24 h | 39.05 GB |
| ranger standard | 0.56 h | 4.58 GB |
| ranger `save.memory` | 0.93 h | 11.15 GB |
| ranger GWAS mode | 0.23 h | 2.32 GB |

The `save.memory` row is a useful counterexample to "memory-saving mode always uses less memory": in this benchmark standard ranger used less peak memory.

**Missing values.** ranger's default `na.action = "na.learn"`: at each node all missing values go to one side; the direction is chosen by the split criterion's impurity decrease; the direction is saved as the node default; at prediction a missing value follows the saved direction, or goes left if none exists. Other modes: `na.omit` removes incomplete rows, `na.fail` errors. Breiman's original software behaved differently, using median/mode imputation with optional iterative proximity-based imputation over up to six repetitions. These policies must not be conflated.

Sources: Wright & Ziegler (2017) DOI 10.18637/jss.v077.i01, arXiv:1508.04409; ranger 0.18.0 manual https://cran.r-project.org/web/packages/ranger/ranger.pdf.

## 5. Generalized Random Forests

### 5.1 The abstract forest framework

Data are $(X_i, O_i) \in \mathcal{X} \times \mathcal{O}$, $i = 1, \ldots, n$, with $\mathcal{X} = [0,1]^p$. The target is defined by a **local moment equation**

$$E\big[\psi_{\theta(x),\nu(x)}(O_i) \mid X_i = x\big] = 0 \qquad \forall x \in \mathcal{X}. \tag{1}$$

Forest weights:

$$\alpha_{bi}(x) = \frac{\mathbf{1}(\{X_i \in L_b(x)\})}{|L_b(x)|}, \qquad \alpha_i(x) = \frac{1}{B} \sum_{b=1}^{B} \alpha_{bi}(x).$$

Each tree gives equal weight to training points in the same leaf as $x$ and zero to the rest; the forest averages. $\sum_i \alpha_i(x) = 1$, $0 \le \alpha_i(x) \le 1$.

The GRF estimate is **not** an average of tree predictions. It solves a weighted estimating equation:

$$\sum_{i=1}^{n} \alpha_i(x)\, \psi_{\hat\theta,\hat\nu}(O_i) = 0. \tag{2}$$

For least squares, $\psi_\mu(Y) = Y - \mu$, this is algebraically identical to Breiman's ensemble average. For general $\psi$ it is not. A forest is an **adaptive nearest-neighbour kernel**, and the statistic is a local estimating-equation solution.

### 5.2 Splitting to maximise heterogeneity

The ideal criterion is $\mathrm{err}(C_1, C_2) = \sum_{j=1,2} P[X \in C_j \mid X \in P]\, E[(\hat\theta_{C_j} - \theta(X))^2 \mid X \in C_j]$.

The unbiased model-free surrogate:

$$\Delta(C_1, C_2) = \frac{n_{C_1} n_{C_2}}{n_P^2} \big(\hat\theta_{C_1}(\mathcal{J}) - \hat\theta_{C_2}(\mathcal{J})\big)^2, \tag{12}$$

with $\mathrm{err}(C_1, C_2) = K(P) - E[\Delta(C_1, C_2)] + o(r^2)$. Maximising $\Delta$ maximises heterogeneity.

Exact evaluation is too expensive, so GRF uses a **gradient-based approximation**. At parent $P$, solve (4) for $(\hat\theta_P, \hat\nu_P)$ and set

$$A_P = \frac{1}{|\{i : X_i \in P\}|} \sum_{\{i : X_i \in P\}} \nabla \psi_{\hat\theta_P, \hat\nu_P}(O_i), \tag{7}$$

then label each observation with the pseudo-outcome

$$\rho_i = -\xi^\top A_P^{-1} \psi_{\hat\theta_P, \hat\nu_P}(O_i) \in \mathbb{R}, \tag{8}$$

where $\xi$ selects the parameter of interest. Then run a **standard CART regression split** on $\rho_i$ maximising

$$\widetilde\Delta(C_1, C_2) = \sum_{j=1}^{2} \frac{1}{|\{i : X_i \in C_j\}|} \Big(\sum_{\{i : X_i \in C_j\}} \rho_i \Big)^2. \tag{9}$$

If $A_P$ is consistent, $\widetilde\Delta = \Delta + o_P(\max\{r^2, 1/n_{C_1}, 1/n_{C_2}\})$.

**Implementation note that decides whether this is cheap enough to build.** Equation (9) is evaluable for all candidate split points of a feature in **one pass using cumulative sums**, exactly like CART. Optimising the ideal criterion (5) directly does not have that property. That is the stated reason for the entire gradient device, and it means GRF splitting reuses the existing split-scan machinery.

### 5.3 Algorithm 1, honest plus subsampling

```
GeneralizedRandomForest(S, x):
  α ← zeros(|S|)
  for b = 1..B:
      I ← Subsample(S, s)                    # without replacement, size s
      J1, J2 ← SplitSample(I)                 # two even, disjoint halves
      T ← GradientTree(J1, X)                 # Algorithm 2
      N ← Neighbors(x, T, J2)                 # {j ∈ J2 : X_j in same leaf as x}
      for e ∈ N: α[e] += 1/|N|
  return θ̂(x) = solution of (2) with weights α/B
```

Algorithm 2 is a breadth-first walk over nodes: per node compute $(\hat\theta_P, \hat\nu_P, A_P)$, compute $\rho$ via (8), make a CART split via (9), recurse. Note the exact structure: **splits come from $J_1$, leaf membership comes from $J_2$**. That split is re-randomised per tree.

### 5.4 The orthogonalizing moments, stated precisely

**The GRF paper never uses the phrase "three orthogonalizing moments."** What it does is orthogonalize a *single* local moment in three distinct ways. A Rust implementer needs the exact moment conditions, so:

1. **Identification moment (uncentered).** $E[\psi_{\theta(x),\nu(x)}(O) \mid X = x] = 0$ (eq. 1). Examples: least squares $\psi_\mu(Y) = Y - \mu$; quantile regression $\psi_\theta(Y) = q - \mathbf{1}(\{Y \le \theta\})$ with $M_\theta(x) = q - F_x(\theta)$; CAPE $\psi_{\beta(x),c(x)}(Y,W) = (Y - \beta(x) \cdot W - c(x)) \binom{\mathbf{1}\{W^\top\}}{1}$.

2. **Orthogonalized residual-on-residual moment, CAPE.** With centered outcomes $\widetilde Y_i = Y_i - \hat y^{(-i)}(X_i)$ and $\widetilde W_i = W_i - \hat w^{(-i)}(X_i)$:

$$\theta(x) = \xi^\top \operatorname{Var}\big[(W_i - E[W_i \mid X_i]) \mid X_i \in \mathcal{S}\big]^{-1} \operatorname{Cov}\big[(W_i - E[W_i \mid X_i]),\, (Y_i - E[Y_i \mid X_i]) \mid X_i \in \mathcal{S}\big].$$

   This is Robinson (1988)'s orthogonalization. It identifies $\theta(x)$ for any $x$ in a region $\mathcal{S}$ on which $\beta$ is constant. **Local centering of gradients** means computing the pseudo-outcome (8) from the *centered* $Y, W$:

$$\rho_i = \xi^\top A_P^{-1} (W_i - \bar W_P) \Big(Y_i - \bar Y_P - (W_i - \bar W_P)\hat\beta_P\Big), \quad A_P = \frac{1}{|P|} \sum_{i \in P} (W - \bar W_P)^{\otimes 2}.$$

   $\hat\beta_P$ is the least-squares solution of $Y$ on $W$ in the parent; $A_P^{-1}$ is computed once per node. Centering uses **leave-one-out** predictions; the paper notes the theory is unaffected by *any* deterministic centering, but $k$-fold cross-fitting is safer if the theory is to apply verbatim, and LOO is much cheaper in forests.

3. **Instrumental-variables moments (two).** $E[Z_i(Y_i - W_i\tau(x) - \mu(x)) \mid X_i = x] = 0$ and $E[Y_i - W_i\tau(x) - \mu(x) \mid X_i = x] = 0$, with pseudo-outcome $\rho_i = (Z_i - \bar Z_P)\big[(Y_i - \bar Y_P) - (W_i - \bar W_P)\hat\tau_P\big]$ and $V(x) = \begin{pmatrix} E[ZW \mid X=x] & E[Z \mid X=x] \\ E[W \mid X=x] & 1\end{pmatrix}$.

The **variance decomposes into two orthogonal pieces**:

$$\operatorname{Var}\big[\tilde\theta^*(x)\big] = \xi^\top V(x)^{-1} H_n(x;\theta(x),\nu(x)) \big(V(x)^{-1}\big)^\top \xi,$$

with $V(x) = \partial_{(\theta,\nu)} M_{\theta,\nu}(x)\big|_{(\theta(x),\nu(x))}$ and $H_n(x;\theta,\nu) = \operatorname{Var}\big[\sum_i \alpha_i(x) \psi_{\theta,\nu}(O_i)\big]$.

### 5.5 Cluster-local averaging

GRF itself has no cluster parameter. Clustering enters at two levels:

- **Local moments.** Each node's $(\hat\theta_P, \hat\nu_P)$ in (4) is a moment solved on the data falling in that node, and the forest score (2) is a weighted average of *local* moments over the whole training set with weights $\alpha_i(x)$.
- **Software-level cluster-local subsampling and averaging.** Sampling operates on whole clusters: draw cluster IDs, then draw `samples_per_cluster` observations from each selected cluster (default all observations of the cluster; with `equalize.cluster.weights=TRUE`, the size of the smallest cluster). With honesty, the **cluster ID set** is split in half, so one half grows the tree and the other repopulates leaves. OOB determination is cluster level: an observation is OOB only if its cluster was not drawn. For ATE-type summaries each cluster gets equal weight by default, the per-cluster estimate is $\hat\theta_j = \sum_{i:A_i=j} \hat\theta_i / n_j$, and the variance is the cluster-level sandwich $\sum_j (\hat\theta_j - \hat\theta)^2 / (J(J-1))$.

### 5.6 The CLT, its assumptions, and its two admitted gaps

**Assumptions 1–6** on $M_{\theta,\nu}(x) = E[\psi_{\theta,\nu}(O) \mid X = x]$: (1) Lipschitz $x$-signal; (2) smooth identification — $M$ twice continuously differentiable in $(\theta,\nu)$ with uniformly bounded second derivative and $V(x)$ invertible; (3) Lipschitz $(\theta,\nu)$-variogram; (4) regularity — $\psi_{\theta,\nu}(O) = \lambda(\theta,\nu;O_i) + \zeta_{\theta,\nu}(g(O_i))$ with $\lambda$ Lipschitz, $g$ univariate, $\zeta$ monotone and bounded; (5) existence of solutions — $\|\sum_i \alpha_i \psi_{\hat\theta,\hat\nu}(O_i)\|_2 \le C \max_i \alpha_i$; (6) **convexity** — $\psi_{\theta,\nu}(O_i)$ is a negative sub-gradient of a convex function.

**Specification 1** (tree regularity): symmetric; balanced splits with each child receiving at least $\omega$ of the parent's observations; split probability on the $j$-th feature bounded below by $\pi > 0$; forest honest; subsample size $s$ with $s/n \to 0$, $s \to \infty$. To satisfy the $\pi$-condition the implementation draws the number of candidate split variables from $\min\{\max\{\mathrm{Poisson}(m), 1\}, p\}$.

**Subsample-rate window:**

$$\beta_{\min} := 1 - \Big(1 + \pi^{-1} \tfrac{\log(\omega^{-1})}{\log((1-\omega)^{-1})}\Big)^{-1} < \beta < 1, \qquad s = n^\beta.$$

**Theorem 3 (consistency).** Under Assumptions 1–6 and Specification 1, $(\hat\theta(x), \hat\nu(x)) \to (\theta(x), \nu(x))$ in probability.

**Theorem 5 (CLT).** If $\operatorname{Var}[\rho_i^*(x) \mid X_i = x] > 0$, there is $\sigma_n(x)$ with

$$\frac{\hat\theta_n(x) - \theta(x)}{\sigma_n(x)} \Rightarrow N(0,1), \qquad \sigma_n^2(x) = \mathrm{polylog}(n/s)^{-1} \frac{s}{n}.$$

**Confidence intervals** follow from the delta method: with any consistent $\hat\sigma_n(x)/\sigma_n(x) \to_p 1$, the Wald interval has asymptotically nominal coverage. The variance estimator is $\hat\sigma_n^2(x) = \xi^\top \widehat V_n(x)^{-1} \widehat H_n(x) (\widehat V_n(x)^{-1})^\top \xi$.

**Bootstrap of little bags.** The ideal half-sampling estimator $\widehat H_n^{HS}(x)$ is infeasible. The feasible version draws $g = 1, \ldots, B/\ell$ random half-samples $\mathcal{H}_g$ of size $\lfloor n/2 \rfloor$, and forces the subsample $\mathcal{I}_b$ of tree $b$ to satisfy $\mathcal{I}_b \subseteq \mathcal{H}_{\lceil b/\ell \rceil}$ (a "little bag" of $\ell$ trees). Then

$$E_{ss}\Big[\Big(\tfrac{1}{\ell}\sum_{b=1}^{\ell}\Psi_b - \Psi\Big)^2\Big] = \widehat H_n^{HS}(x) + \frac{1}{\ell-1} E_{ss}\Big[\tfrac{1}{\ell}\sum_{b=1}^{\ell}\Big(\Psi_b - \tfrac{1}{\ell}\sum_{b'\le\ell}\Psi_{b'}\Big)^2\Big].$$

$\widehat H_n^{HS}$ is consistent for $H_n$. For small $B$ it can be negative; grf avoids this with a Bayesian analysis of variance using an improper uniform prior on $[0, \infty)$.

**Two admitted gaps, both of which must appear in the API docs.** (i) CI validity relies on **undersmoothing** — without it the intervals account for sampling variability but not bias, and a principled bias correction is open. (ii) **Edge effects**: "forests can exhibit edge effects whereby the slope of our estimates $\hat\theta(x)$ may taper off as we approach the edge of $\mathcal{X}$-space, even when the true function $\theta(x)$ keeps changing."

Source: Athey, Tibshirani & Wager (2019), DOI 10.1214/18-AOS1709, arXiv:1610.01271v4, open access at https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Generalized-random-forests/10.1214/18-AOS1709.full.

## 6. Honesty

### 6.1 Definition

> "a tree is honest if, for each training example $i$, it only uses the response $Y_i$ to estimate the within-leaf treatment effect $\tau$ ... **or** to decide where to place the splits, but not both."

Formally (Definition 2): (a) standard case — the tree does not use the responses $Y_1, \ldots, Y_s$ in choosing where to place its splits; (b) double-sample case — the tree does not use the $\mathcal{I}$-sample responses for placing splits. For causal trees, an honest causal tree may look at $W_i$ but not $Y_i$.

### 6.2 Double-sample tree, the variant grf implements

1. Draw a random subsample of size $s$ **without replacement**; split into $\mathcal{I}$ and $\mathcal{J}$ with $|\mathcal{I}| = \lfloor s/2 \rfloor$, $|\mathcal{J}| = \lceil s/2 \rceil$.
2. Grow the tree. "The splits are chosen using any data from the $\mathcal{J}$ sample and $X$- or $W$-observations from the $\mathcal{I}$ sample, but **without using $Y$-observations from the $\mathcal{I}$-sample**."
3. Estimate leaf-wise responses using **only the $\mathcal{I}$-sample**.

Double-sample causal trees use the Athey–Imbens (2016) splitting rule: **maximise the variance of $\hat\tau(X_i)$ for $i \in \mathcal{J}$**. The justification: since $\sum_{i \in \mathcal{J}}(\hat\mu(X_i) - Y_i)^2 = \sum_{i \in \mathcal{J}} Y_i^2 - \sum_{i \in \mathcal{J}} \hat\mu(X_i)^2$, minimising CART's SSE equals maximising the variance of leaf predictions; the causal analogue maximises the variance of $\hat\tau$.

The second honesty route is the **propensity tree** (Procedure 2): train a classification tree on $(X_i, W_i)$ and estimate $\tau$ on the resulting leaves. It needs no sample splitting but is blind to treatment-effect heterogeneity.

### 6.3 Why honesty removes adaptive bias, and what it costs

The bias mechanism is direct: in a non-honest tree the same $Y_i$ that selected the split also enters the leaf average, so $\hat\tau$ inherits the selection bias of the split search.

Honesty breaks that dependence. No data is wasted: "the forest subsampling mechanism enables us to achieve honesty without wasting any data in this sense, because we re-randomize the $\mathcal{I}/\mathcal{J}$-data splits over each subsample. Thus, although no data point can be used for split selection and leaf estimation in a single tree, each data point will participate in both $\mathcal{I}$ and $\mathcal{J}$ samples of some trees."

The cost is variance. Honesty splits the estimation sample, so each half is smaller and leaf estimates are noisier. It also reduces power, because the splits are chosen without the responses that will be estimated. grf's operational guidance: "honesty may hurt performance when working with very small datasets... the subsample used to determine tree splits is already small, and honesty further cuts this subsample in half, so there may no longer be enough information to choose high-quality splits"; mitigation is to raise `honesty.fraction` (grf's tuner searches **0.5–0.8**), set `honesty.prune.leaves=FALSE`, or disable honesty.

Wager & Athey's own finding: "in practice, double-sample trees can improve upon standard random forests in terms of mean-squared error as well."

The bias bound requires honesty. The scaling window $\beta > \beta_{\min}$ is what makes bias decay faster than variance.

### 6.4 The infinitesimal jackknife

$$\widehat V_{IJ}(x) = \frac{n-1}{n} \Big(\frac{n}{n-s}\Big)^2 \sum_{i=1}^{n} \operatorname{Cov}_*\big[\hat\tau_b^*(x), N_{ib}^*\big]^2$$

where $\hat\tau_b^*(x)$ is the $b$-th tree's estimate and $N_{ib}^* \in \{0,1\}$ indicates use of example $i$ by tree $b$ (for double-sample trees, $N_{ib}^* = 1$ if $i$ is in either the $\mathcal{I}$ or the $\mathcal{J}$ sample). The factor $n(n-1)/(n-s)^2$ is a **finite-sample correction for subsampling without replacement**, absent in Wager–Hastie–Efron (2014), which assumed sampling with replacement. For trivial unsplit trees, $\widehat V_{IJ}$ equals the usual unbiased sample variance and $E[\widehat V_{IJ}] = \operatorname{Var}[\hat\mu]$ exactly.

### 6.5 The theorems, and the important caveat

- **Theorem 1 (regression forests).** $X_i \sim U([0,1]^d)$; $\mu, \mu_2$ Lipschitz; $\operatorname{Var}[Y \mid X=x] > 0$; $E[|Y - \mu(X)|^{2+\delta} \mid X=x] \le M$; $T$ an honest, $\alpha$-regular ($\alpha \le 0.2$), symmetric random-split tree; subsample size $s_n \asymp n^\beta$ with $\beta_{\min} := 1 - \Big(1 + \frac{d}{\pi}\frac{\log(\alpha^{-1})}{\log((1-\alpha)^{-1})}\Big)^{-1} < \beta < 1$. Then $(\hat\mu_n(x) - \mu(x))/\sigma_n(x) \Rightarrow N(0,1)$ and $\widehat V_{IJ}/\sigma_n^2 \to_p 1$.
- **Theorem 3 (bias).** $|E[\hat\mu(x)] - \mu(x)| = O\big(s^{-\frac{1}{2}\frac{\log((1-\alpha)^{-1})}{\log(\alpha^{-1})}\frac{\pi}{d}}\big)$. Requires honesty.
- **Theorem 11 (causal forests).** Under unconfoundedness and overlap ($\varepsilon < P(W=1 \mid X=x) < 1-\varepsilon$), honest $\alpha$-regular ($\alpha \le 0.2$) symmetric random-split causal forest, and $s_n \asymp n^\beta$ with $\beta > \beta_{\min}$: predictions are **consistent and asymptotically both Gaussian and centered**, and the variance is consistently estimated by the IJ. Regularity for causal trees requires at least $k$ observations of *each* treatment group in the leaf.
- **Remark 4 (testing at many points).** It is **not in general possible** to build causal trees that are regular at all $x$ simultaneously. In dimension one with $W_i = \mathbf{1}\{X_i \ge 0\}$ at most one regular leaf exists. The practical fix is to assign different trees to different test points and treat the subset assigned to $x$ as the relevant forest. **This is a real limitation a `ModelForest` API must expose**: inference at $x$ may use a data-dependent tree subset.

### 6.6 Honest versus doubly robust: the comparison that must be explicit

| | Honest (split-sample) | Doubly robust / orthogonalized (AIPW or residual-on-residual) |
|---|---|---|
| What it fixes | **Adaptive bias** from selecting splits on the same outcomes that are averaged in the leaf | **Confounding bias** from $X$ predicting both $W$ and $Y$ |
| Mechanism | Disjoint $\mathcal{I}$ (estimation) / $\mathcal{J}$ (splits), re-randomised per tree | Robinson residualisation ($Y - m(x)$, $W - e(x)$) and/or AIPW scores |
| Cost | Each half-sample is smaller, so higher variance and less power to detect heterogeneity | Needs two nuisance models; requires $n^{1/4}$-consistent nuisances fitted by cross-fitting for the root-$n$ bias to vanish |
| Guarantee | Consistency + centered asymptotic normality | Root-$n$ asymptotics of the *ATE*; for CATE with forests the theory still rests on the forest's own consistency plus orthogonality |
| Substitutes? | **No.** Honesty does nothing about confounding | **No.** DR does nothing about adaptive split-selection bias |

grf uses both: orthogonalisation for the first stage, honesty inside the forest, and AIPW for the ATE summary. **These must be independent switches in the API, and the doubly-robust layer sits above the forest, not inside the tree-growing rule.**

Source: Wager & Athey (2018), JASA 113(523):1228–1242, DOI 10.1080/01621459.2017.1319839, arXiv:1510.04342v4.

## 7. Causal Forests

### 7.1 Difference-in-means leaf estimator

$$\hat\tau(x) = \frac{1}{|\{i : W_i=1, X_i \in L\}|} \sum_{i : W_i=1, X_i \in L} Y_i - \frac{1}{|\{i : W_i=0, X_i \in L\}|} \sum_{i : W_i=0, X_i \in L} Y_i.$$

Forest aggregation is $\hat\tau(x) = B^{-1} \sum_b \hat\tau_b(x)$. Each leaf must contain $k$ or more $\mathcal{I}$-sample observations of **each** treatment class. Unlike Athey & Imbens (2016), who use large leaves plus within-leaf propensity weighting, Wager–Athey use **deep** trees with small leaves and no within-leaf correction.

### 7.2 The doubly-robust score

$$\hat\Gamma_i = \hat\tau^{(-i)}(X_i) + \frac{W_i - \hat e^{(-i)}(X_i)}{\hat e^{(-i)}(X_i)\big[1 - \hat e^{(-i)}(X_i)\big]} \Big(Y_i - \hat\mu^{(-i)}(X_i, W_i)\Big).$$

`average_treatment_effect()` returns the weighted mean of $\hat\Gamma_i$ with an influence-function or cluster-robust standard error, with targets `"all"`, `"treated"`, `"control"`, and `"overlap"` (recommended under poor overlap, following Li–Morgan–Zaslavsky 2018). `best_linear_projection()` regresses the same scores on covariates $A$, with `vcov.type="HC3"` by default. For continuous $W$, the inverse-propensity term is replaced by a forest estimate of $\operatorname{Var}[W \mid X=x]$, with `num.trees.for.weights = 500` auxiliary trees.

### 7.3 The local-centered heterogeneity estimator

$$\hat\theta(x) = \xi^\top \Big(\sum_i \alpha_i(x)(W_i - \bar W_\alpha)^{\otimes 2}\Big)^{-1} \sum_i \alpha_i(x)(W_i - \bar W_\alpha)(Y_i - \bar Y_\alpha),$$

with $\bar W_\alpha = \sum \alpha_i(x) W_i$ and $\bar Y_\alpha = \sum \alpha_i(x) Y_i$. The splitting rule uses the centered pseudo-outcome, so **the split criterion itself is invariant to confounding**.

### 7.4 What the papers do not contain

**There is no quantity called "the local centered heterogeneity score."** Three real objects play that role:

1. The local centered moment (the identifying moment after local centering).
2. The **R-loss** heterogeneity objective used for tuning and calibration (Nie & Wager 2021): $\widehat{\mathrm{Err}}(\hat\tau) = \sum_i \big(Y_i - \hat m^{(-i)}(X_i) - \hat\tau(\mathbf{X}_i)(W_i - \hat e^{(-i)}(X_i))\big)^2$. grf's tuner uses exactly this for causal forests.
3. The **calibration / heterogeneity-detection statistic** (Athey & Wager 2019): build $C_i = W_i - \hat e^{(-i)}(X_i)$ and $D_i = (\hat\tau(X_i) - \bar\tau)C_i$ and regress $Y_i - \hat m^{(-i)}(X_i)$ on $(C_i, D_i)$; the coefficient on $D_i$ is 1 if and only if the CATE estimates are well calibrated. grf ships this as `test_calibration()`. The newer `rank_average_treatment_effect()` (RATE, Yadlowsky et al. 2025) is the recommended modern diagnostic.

**The authors themselves warn**: "asymptotic results justifying this inference are not presently available" for the subgroup and heterogeneity tests built on OOB CATE estimates. Do not claim valid inference for a heterogeneity test on the basis of GRF.

Sources: Wager & Athey (2018); Athey, Tibshirani & Wager (2019); Athey & Wager (2019), *Observational Studies* 5:36–51, arXiv:1902.07409; Nie & Wager (2021) DOI 10.1093/biomet/asaa076; Yadlowsky et al. (2025) arXiv:2111.07966.

## 8. Local Linear Forests

### 8.1 The solve

Forest weights are as in GRF. Let $A = \operatorname{diag}(\alpha_i(x_0))$, $J$ the $(d+1)\times(d+1)$ diagonal with $J_{1,1} = 0$ (intercept unpenalised), and $\Delta_{i,1} = 1$, $\Delta_{i,j+1} = x_{0,i,j} - x_{0,0,j}$. Then

$$\begin{pmatrix} \hat\mu(x_0) \\ \hat\theta(x_0) \end{pmatrix} = \big(\Delta^\top A \Delta + \lambda J\big)^{-1} \Delta^\top A Y.$$

Equivalently $\hat\mu(x_0) = \sum_i \gamma_i \alpha_i(x_0) Y_i$ with $\sum_i \gamma_i \alpha_i = 1$ and $\sum_i \gamma_i \alpha_i \mathbf{X}_i \approx \mathbf{x}_0$ (exact when $\lambda = 0$).

### 8.2 Residual splitting

Because the final regression models the smooth global signal, the forest should only model *local* variation. In each parent $P$, run a ridge regression $\hat Y_{ik} = \mathbf{x}_{ik}^\top \hat\beta_P$ with $\hat\beta_P = (\mathbf{x}_P^\top \mathbf{x}_P + \lambda J)^{-1}\mathbf{x}_P^\top \mathbf{Y}_P$, then run a standard CART split on the **residuals**. CART splits burn splits on the strongest linear signals; residual splits spend them on the hard-to-model structure.

Pragmatic tuning guidance, explicitly *not* covered by the theory: use small $\lambda$ during forest growth; when the child leaf falls below a cutoff, reuse the parent's $\hat\beta_P$ rather than re-estimating on the tiny child; use a larger $\lambda$ at the final prediction step; optionally restrict the prediction step to a selected subset of covariates.

### 8.3 Why it removes boundary bias

Classic local *constant* estimation has $O(h)$ boundary bias; local *linear* has $O(h^2)$ boundary bias, at the cost of $O(h)$ (versus $O(h^2)$) interior bias and higher variance. The forest supplies locality through the adaptive weights; the local-linear step supplies the bias cancellation. A regression forest's bias distorts both the prediction curve and the confidence intervals, which are then not centred on the true function.

### 8.4 Honesty as used by LLF

Each tree uses two non-overlapping subsamples $\mathcal{I}_b, \mathcal{J}_b$. $T_b$ is grown using only $\mathcal{J}_b$'s covariates (outcomes from $\mathcal{J}_b$ are *not* used to select splits), and $L_b(x_0) = \{i \in \mathcal{I}_b : x_0 \leftrightarrow_b \mathbf{X}_i\}$ is the neighbourhood used to form $\alpha_i(x_0)$. The paper states plainly: "there are no available consistency results for random forests with fully grown trees that do not use honesty."

### 8.5 The rate improvement, and why LLF needs its own CLT

Assumptions: regular trees (symmetric, split probability at least $\pi$, balanced with $\omega$, depth-$k$ grown), honest forest, $\omega \le 0.2$; $\mu(x_0)$ twice continuously differentiable; $\operatorname{Var}[Y \mid X=x_0] > 0$; subsample $s = n^\beta$ with

$$\beta_{\min}^{LLF} := 1 - \Big(1 + \frac{d}{1.56\pi}\frac{\log(\omega^{-1})}{\log((1-\omega)^{-1})}\Big)^{-1} < \beta < 1,$$

and $\lambda = \Theta\big(d\sqrt{s/n}\,(s/(2k-1))^{-1.56\frac{\log((1-\omega)^{-1})}{\log(\omega^{-1})}\frac{\pi}{d}}\big)$. Then $(\hat\mu_n(x_0) - \mu(x_0))/\sigma_n(x_0) \Rightarrow N(0,1)$ with $\sigma_n^2(x_0) = O(n^{-(1-\beta)})$.

$\beta_{\min}^{LLF} < \beta_{\min}^{RF}$, because smoothness is assumed and exploited. The constants differ structurally: RF uses $d/\pi$ and $\alpha$; LLF uses $d/(1.56\pi)$ and $\omega$.

**Why LLF needs its own CLT.** In an honest RF, $Y_i \perp \alpha_i(x_0) \mid \mathbf{X}_i$, which is the key step in Wager–Athey. In an LLF, $\hat\mu(x_0) = \sum_i \alpha_i \rho_i$ with $\rho_i = \mathbf{e}_1^\top M_\lambda^{-1}\binom{1}{\mathbf{X}_i - \mathbf{x}_0}Y_i$; conditioning on $\mathbf{X}_i$ does **not** make $\rho_i$ and $\alpha_i(x_0)$ independent, so the Wager–Athey argument breaks.

LLF confidence intervals use $\widehat V(x_0) = M_\lambda$, available in **closed form** from the forest, with $\hat\sigma_n^2 = \zeta^\top \widehat H_n \zeta$, $\zeta = \mathbf{e}_1^\top M_\lambda^{-1}$, $\Gamma_i = (\zeta \cdot \Delta_i)(Y_i - \Delta_i(\mu,\theta)^\top)$. $\widehat H_n$ is estimated by the bootstrap of little bags with Efron (1982) half-sampling variance across all $\binom{n}{\lfloor n/2 \rfloor}$ half-samples, approximated by Monte Carlo with bias corrections.

Source: Friedberg, Tibshirani, Athey & Wager (2021), *JCGS* 30(2), DOI 10.1080/10618600.2020.1831930, arXiv:1807.11408v3.

## 9. Cubist, M5, and the Committee

### 9.1 M5 (1992), the model-tree core

1. **Split criterion (SDR).** At node with cases $T$, compute $\mathrm{SD}(T)$. For a candidate test with outcomes $j$ and subsets $T_{ij}$:

$$\text{SDR} = \mathrm{SD}(T) - \sum_j \frac{n_j}{n}\,\mathrm{SD}(T_j).$$

   CART instead reduces variance or absolute deviation; M5 uses SD.

2. **Error estimate inflation.** The training residual (mean absolute error) is multiplied by $\frac{n+t}{n-t}$, where $n$ is the number of training cases and $t$ the number of model parameters, "to increase the estimated error of models with many parameters constructed from small numbers of cases". The Cubist code implements exactly this: `EstimateErr(Val, NData, NParam) = Val * (NData + NParam) / (NData - NParam)`.

3. **Linear model at every node**, restricted to "the attributes that are referenced by tests or linear models somewhere in the subtree at this node" — a level-playing-field rule, so a linear model and a subtree use the same information.

4. **Simplification of linear models**: greedy elimination of parameters to minimise estimated error. Removing parameters raises mean residual but lowers the multiplicative factor, so estimated error can drop.

5. **Pruning**: each non-leaf is examined bottom-up. Keep whichever of the simplified linear model at the node or the model subtree has lower estimated error; if the linear model wins, prune the subtree to a leaf.

6. **Smoothing (hierarchical backup)**: the leaf prediction is backed up to the root:

$$p' = \frac{n\,p + k\,q}{n+k},$$

where $p$ is the prediction passed up from below, $q$ is the value predicted by the linear model at the node, $n$ is the number of training cases reaching the node below, and $k$ is the smoothing constant, **default 15**. Motivation, from the paper: "Smoothing has most effect on a case when the models along the path predict very different values and when some models were constructed from few training cases."

M5 also differs from CART in that regression trees "will never give a predicted value lying outside the range observed in the training cases whereas model trees can extrapolate".

### 9.2 M5′ and where the smoothing lives

M5′ is Wang & Witten's "rational reconstruction" of M5, with improved missing-value handling and enumerated-attribute support: M5′ converts an $n$-valued nominal attribute into $n-1$ binary attributes using CART's procedure.

M5′ performs the smoothing **inside the leaf model** rather than at prediction time. Frank et al. (1998) state "Quinlan's default value of $k=15$ was used in all experiments" and give the coefficient-level combination: leaf $p = ax + by$, parent $q = cy + dz$ become $\big(\tfrac{na}{n+k}, \tfrac{nb+kc}{n+k}, \tfrac{kd}{n+k}\big)$ — "Continuing in this way up to the root gives us a single, smoothed linear model which we install at the leaf and use for prediction thereafter."

Weka implements the constant literally: `SMOOTHING_CONSTANT = 15.0`, with `smoothingOriginal(n, pred, supportPred) = (n*pred + 15*supportPred)/(n+15)`. Weka also exposes `-U` for unsmoothed predictions.

### 9.3 Cubist's smoothing is a different algorithm, and `m0` is a phantom

**There is no `m0` or `M0` constant anywhere in Cubist.** A grep over the whole GPL tree returns zero hits.

Cubist's smoothing is a Pregibon-like *coefficient shrink* between a node's linear model and its parent's already-smoothed model, in `SmoothModels`:

- skipped when $\sum w_i \le 2$ or residual variance $\le 10^{-10}$;
- for each ancestor, compute `CurrVar` (variance of this node's residuals) and `ParentVar` (variance of the parent's residuals on the same cases), and their covariance `Cov` (weighted, $n-1$ denominator);
- blend weight

$$p_{\text{cur}} = \frac{\text{ParentVar} - \text{Cov}}{\text{CurrVar} + \text{ParentVar} - 2\,\text{Cov}},$$

set to 1 if the denominator is at most $10^{-12}$. This is the variance-minimising convex combination of the two estimators;
- if $0 < p_{\text{cur}} < 1$, the node's model coefficients are blended with the parent's, the node's error variance is recomputed, and recursion continues upward;
- finally coefficients are simplified to a precision grid (`AttUnit`), and coefficients with magnitude below `MinAttCoeff(a) = 0.01 * GlobalSD / AttSD[a]` are zeroed. The intercept is shifted so the **weighted median** residual is zero.

**API consequence.** Implement the two smoothing styles as separate, named modes: `SmoothingKind::QuinlanBackup { k: 15.0 }` and `SmoothingKind::CubistShrink`. Do not ship a parameter called `m0` without stating which of these it belongs to.

### 9.4 Cubist tree growing and split rule

**Split gain (SDR, weighted):**

$$\text{gain} = \text{Node->SD} - \frac{\sum_{v=1}^{3}\,\text{freq}_v\,\text{SD}_v}{\sum_{v=1}^{3}\,\text{freq}_v}, \qquad \mathrm{SD}(N,\Sigma,\Sigma^2) = \sqrt{\tfrac{\Sigma^2 - \Sigma^2/N + 10^{-3}}{N-1}}.$$

**Difference from Quinlan.** Cubist computes the gain on **residuals of the node's own linear model**, not on raw targets: `Val = Root ? Class(Case[i]) : Class(Case[i]) - LinModel(GEnv.LocalModel, Case[i])`. At the root the local model is zero, so root gain equals textbook SDR.

**Continuous attribute cut search.** Cases are sorted by attribute; a candidate cut only exists between *distinct* values. Branch-size guard `Edge = (Cases >= 3*MINITEMS ? MINITEMS : MINSPLIT)`. Each branch's residual sum of squares is derived from the least-squares correlation of the attribute with the residual, so no per-branch regression is fitted.

**Discrete attributes.** Binary attributes use `EvalBinarySplit`; multi-valued use `EvalSubsetSplit`; a split is "reasonable" only if at least two outcomes hold `freq >= MINSPLIT`.

**Attribute preference tie-break.** If a candidate's gain is greater than $0.999 \times$ the best and the attribute has higher `AttPref`, it wins. `AttPref` is derived by counting sign agreement between attribute deviation and target deviation.

**Stopping.** If the maximum absolute residual is below $0.5 \times \text{AttUnit}[0]$, the node stops.

### 9.5 Exact defaults

| Quantity | Value | Source |
|---|---|---|
| `MINSPLIT` | **3**, min branch size for the initial tree | `src/defns.h:85` |
| `MINFRACT` | 0.001, min fraction of cases covered by a rule | `src/defns.h:86` |
| `MINITEMS` | $\max\!\big(3,\ \min(20,\ \mathrm{round}(n/100))\big)$, additionally capped by $n/\text{MAXRULES}$ | `src/construct.c:135-149` |
| `MAXRULES` | **500** (CLI `-r`); CRAN `cubistControl(rules = 100)` | RuleQuest appendix; CRAN manual |
| `NNMAX` / `NNSLACK` | **9** / 11 | `src/defns.h:88-89` |
| `EVALSAMPLE` | 10000, max training cases sampled for evaluation | `src/defns.h:91` |
| Committee size | CLI `-C members`, "recommended value 5"; CRAN `committees = 1` | RuleQuest; CRAN manual |
| Neighbours | CLI `-n` 1..9 or auto; CRAN `predict(..., neighbors = 0)`, default off | CRAN manual |
| Extrapolation limit | CLI `-e percent`, **5%**; CRAN `cubistControl(extrapolation = 100)`, **100%** | RuleQuest; CRAN manual |
| Smoothing constant | `k = 15` | Quinlan 1992; Weka `SMOOTHING_CONSTANT` |

The effective floor on rule coverage is $\max\!\big(3,\ \min(20,\ \mathrm{round}(n/100))\big)$, additionally capped by $n/\text{MAXRULES}$.

### 9.6 Linear models, simplification, unbiased rules

At every node a linear model is fitted by weighted least squares over the **restricted** attribute set (attributes used by ancestor tests plus attributes referenced inside the subtree plus formula-defined attributes). `SimplifyModel` removes attributes greedily; coefficients are rounded to `AttUnit` precision and dropped if $|b_a| < 0.01\,\text{GlobalSD}/\text{AttSD}_a$.

**Default rules are median-centred**: the intercept is shifted so the weighted median residual is zero, which biases rules toward the median to minimise mean absolute error. `-u` (unbiased rules) instead iteratively removes the **mean** bias, re-estimating after each subtraction because range truncation can change the residual mean.

Documented effect from the RuleQuest tutorial: rule 8 constant 3652.1 to 3555.3 and estimated error 116.2 to 169; unbiased rules "have a slightly greater error of 153.1 on the unseen test cases"; and "because cases can be covered by different numbers of rules, the use of unbiased rules does not guarantee that the entire model is unbiased."

### 9.7 Rules, prediction, extrapolation

The tree becomes a rule set. **Rules may overlap.** Prediction for a case is the unweighted average of the linear models of **all** matching rules, each first clipped to that rule's `[LoLim, HiLim]`. If no rule matches, the global mean is used.

**Extrapolation limit** equals the range limit: each rule stores the min and max target of its covered training cases, and a prediction outside $[\min - e \cdot \text{range}, \max + e \cdot \text{range}]$ is clipped to the nearer bound. Worked example from the RuleQuest tutorial: rule range 884 to 2801 (range 1917) gives bounds 788.15 to 2896.85 at the 5% default.

Extra sign constraint: a rule whose training targets are all at least zero never predicts below zero, and a rule whose targets are all at most zero never predicts above zero.

CRAN differs from the CLI on both `rules` (100 versus 500) and `extrapolation` (100% versus 5%). A port must expose and document both defaults; a single "default" is a design decision, not a fact.

### 9.8 Committees, the mechanism that matters

`MEMBERS` is the committee size. "The first member of a committee model is always exactly the same as the model generated without the committee option."

**Member 2 through M training targets are transformed:**

```text
PredVal(Case[i]) = RuleSetPrediction(Cttee[m], Case[i]);
Class(Case[i])  = 2 * CVal(Case[i], ClassAtt) - PredVal(Case[i]];
```

The next member regresses on $y' = 2y - \hat y_m$. **Final prediction is a simple average over members**, with no stage weights, no learning rate, and no row resampling:

```text
PredSum += RuleSetPrediction(Cttee[m], CaseDesc);
return PredSum / MEMBERS;
```

The CRAN vignette states the same and contrasts with boosting: "Unlike traditional boosting, stage weights for each committee are not used to average the predictions from each model tree; the final prediction is a simple average of the predictions from each model tree."

**What this means statistically.** Under equal averaging with target $2y - \hat y$, member $m+1$ receives weight $1/2$ implicitly, so the effective learning rate on the correction is $1/2$ with no decay across members. It is closest to a two-point Newton correction of the ensemble mean. This is why Cubist recommends only about 5 members and why the benefit appears only "when the initial model is reasonably accurate". The error does not shrink monotonically with $M$; gains saturate.

Reported utility from the RuleQuest tutorial (fc2008, 341 test cases): rules-only average absolute error 152.6, committee(5) 141.3, composite 99.8, committee plus instances 94.8.

Cubist also computes a committee quality statistic: `ErrReduction = FinalErr / (SumErr / (MEMBERS - 1))`.

**Naming discipline.** Do not describe Cubist committees as boosting without the qualifier. The API name should be `committee`, not `boosted`, for the Cubist lineage, with `gbdt` reserved for Friedman's gradient boosting.

### 9.9 Composite k-NN instance adjustment

**Neighbour count** is 1 to 9. If unset, Cubist auto-selects by evaluating NN = 1..9 on sampled training instances and picking the **smallest** NN whose total error is below $1.01 \times$ the best error.

**Distance** is Manhattan-style over all attributes, summed:

- continuous: $|x_a - x'_a| / (5\,\text{SD}_a)$, **capped at 1**;
- ordered discrete: $|v - v'| / (\#\text{values}-1)$;
- nominal discrete: $2/(\#\text{values}-1)$ if different;
- one attribute missing in one case and present in the other contributes **1.0**;
- the target attribute and skipped attributes are excluded.

**Neighbour value adjustment:** `AdjustedValue(i, b) = Class(i) + b - RSPredVal[i]`, where $b$ is the model's prediction for the **query** case and `RSPredVal[i]` the model's prediction for the neighbour. The neighbour's target is re-based to the query's prediction level.

**Weighted composite estimate:** weight $w_i = 1/(d_i + 0.5)$; ties are grouped, and if a group would overflow the neighbour budget the group is down-weighted proportionally; the estimate is $\sum w_i v_i / \sum w_i$, clamped to the global floor and ceiling.

**Fallback:** `MinN = (NN+1)/2`, integer division, so $\lceil NN/2 \rceil$. If fewer than `MinN` neighbours were found, the rule-based prediction is returned unchanged.

**Rules versus composite choice:** compare `EstimateErr(RSErr, n, 2*AvRules)` (rules only, each rule counted as 2 parameters) against `EstimateErr(RSIErr, n, AvRules)` (rules plus instances); the composite form is chosen when its inflated error is smaller. CLI flags `-i` force composite, `-a` let Cubist decide, default neither.

Caveat from the tutorial: "Nearest neighbor models are adversely affected by the presence of irrelevant attributes… All attributes are taken into account when evaluating the similarity of two cases". Composite helps most with few, all-relevant attributes.

### 9.10 Case weighting

Optional numeric attribute literally named **`case weight`**; relative weight is attribute value divided by the **average** value. "if the value is undefined ('?'), not applicable ('N/A'), or is less than or equal to zero, the case's relative weight is set to 1". The weight attribute itself is never used as a model attribute, but enters every weighted statistic through `CWeight(Case)`. The CRAN equivalent is the `weights` argument with the same relative-weight definition.

Documented failure mode: weighting helps only when "relatively important cases 'clump together'", that is, when cases with similar predictor values have similar weights; otherwise "case weighting can introduce an unhelpful element of randomness into the model generation process".

### 9.11 M5 versus M5′ versus Cubist, leaf by leaf

| Step | Plain OLS leaf | M5 / M5′ leaf | Cubist leaf |
|---|---|---|---|
| Predictors used | All | Attributes used by tests **or** linear models somewhere in the pruned-away subtree | Continuous attributes used by ancestor tests plus subtree-referenced plus formula-defined attributes |
| Fit | OLS | Weighted least squares, then greedy attribute elimination | Weighted least squares, `SimplifyModel`, coefficient grid rounding, `MinAttCoeff` threshold |
| Centring | None | Intercept by least squares | Intercept shifted so weighted **median** residual is 0 (biased mode) or mean residual is 0 (`-u`) |
| Smoothing | None | Ancestor models mixed in with weight $k/(n+k)$, $k = 15$ | Variance-minimising coefficient shrink against parent |
| Error estimate | — | Inflated by $(n+t)/(n-t)$ | `EstimateErr = Val * (n+t)/(n-t)` |
| Extrapolation | Unbounded | Unbounded | Clipped to rule bounds with the extrapolation limit |

Sources: Quinlan (1992) https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Quinlan-AI.pdf; Wang & Witten (1997); Frank et al. (1998) DOI 10.1023/A:1007421302149; Cubist GPL source https://github.com/topepo/Cubist; RuleQuest https://www.rulequest.com/cubist-unix.html; CRAN Cubist manual https://cran.r-project.org/web/packages/Cubist/refman/Cubist.html; Weka `RuleNode.java`.

## 10. Boosting Foundations

### 10.1 Friedman (2001) greedy function approximation

$$\hat F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma), \qquad \hat F_m(x) = \hat F_{m-1}(x) + \nu\, h_m(x),$$

$$h_m(x) = \arg\min_\gamma \sum_{i=1}^{n} L\big(y_i, \hat F_{m-1}(x_i) + \gamma\big),$$

where $h_m$ is fit by regression on the negative gradient (for squared error, $\gamma_m$ is the mean residual) and $\nu$ is the learning rate or shrinkage. XGBoost's paper restates the lineage: "the second order method is originated from Friedman et al." and describes shrinkage as "scales newly added weights by a factor $\eta$ after each step of tree boosting".

### 10.2 Friedman (2002) stochastic gradient boosting

Each boosting iteration fits the base learner on a **random subsample** of the training cases (fraction $\nu$), with the update remaining $F_m = F_{m-1} + \nu h_m$. The same $\nu$ acts as learning rate and subsample fraction, with $0 < \nu \le 1$; $\nu = 1$ recovers deterministic boosting. Mirrored in every modern implementation's `subsample` parameter.

### 10.3 Bühlmann & Yu (2003)

Correct citation: JASA 98(462):324–339, DOI 10.1198/016214503000125. This is the cleanest statement of boosting's asymptotic character under a smooth loss. **Numerical rate claims are unverified here** because the full text is paywalled; quote only after checking the theorem statements.

### 10.4 Modern GBDT mechanics that distinguish implementations

**XGBoost** (Chen & Guestrin 2016):

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n}\Big[l\big(y_i, \hat y^{(t-1)}\big) + g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)\Big] + \Omega(f_t), \qquad \Omega(f_t) = \gamma T + \tfrac{1}{2}\lambda\|w_k\|^2,$$

with $g_i = \partial_{\hat y^{(t-1)}} l$ and $h_i = \partial^2_{\hat y^{(t-1)}} l$. The **second-order leaf update** is $w = -\sum g / (\sum h + \lambda)$; $\gamma T$ penalises leaf count. Exact-greedy split finding on sorted values, plus an approximate histogram path. Shrinkage $\eta$ plus column subsampling. **Default directions** for missing values are learnt from data, so missingness is a third branch rather than a special value. Docs defaults: `eta = 0.3`, `max_depth = 6`, `min_child_weight = 1`, `gamma = 0`, `lambda = 1`, `subsample = 1`, `colsample_bytree = 1`, `grow_policy = depthwise` (options `depthwise`, `lossguide`), `max_leaves = 0`, `max_bin = 256`, plus `monotone_constraints` and `interaction_constraints`.

**LightGBM** (Ke et al. 2017) names two techniques: **GOSS** (keep large-gradient instances, subsample small-gradient ones) and **EFB** (exclusive feature bundling; optimal bundling is NP-hard, greedy approximation). Growth is **leaf-wise** (best-first): `max_depth` default −1 with the note "Tree still grows leaf-wise"; `num_leaves` default 31; `min_data_in_leaf` default 20; `min_sum_hessian_in_leaf` default 1e-3; `max_bin` default 255; `linear_tree` default false; `monotone_constraints` accepts per-feature `{-1, 0, 1}`; `data_sample_strategy ∈ {bagging, goss}`.

**CatBoost** (Prokhorenkova et al. 2018) uses **ordered boosting**, "a modification of standard gradient boosting algorithm, which avoids target leakage", and **ordered target statistics** with random permutations $\sigma_{cat}$ and $\sigma_{boost}$ so a case's own target never enters its own statistic, removing the classic prediction shift of plain GBDT with categoricals. Base predictors are **oblivious** decision trees. Two modes, Plain and Ordered; the paper reports ordered better at the same asymptotic complexity. Monotone support is via leaf-value shrinkage under symmetric trees.

**Explainable Boosting Machine** (Nori et al. 2019, arXiv:1909.09223) is **not a tree ensemble**. It is a glassbox generalised additive model with pairwise interactions: "learns each feature function $f_j$ using modern machine learning techniques such as bagging and gradient boosting. The boosting procedure is carefully restricted to train on **one feature at a time in round-robin fashion** using a **very low learning rate** so that feature order does not matter", with automatic pairwise terms. interpretML fitted-parameter defaults: `max_bins = 1024`, `max_interaction_bins = 64`, `learning_rate = 0.015`, `greedy_ratio = 10.0`, `smoothing_rounds = 75`, `interaction_smoothing_rounds = 75`, `max_rounds = 50000`, `early_stopping_rounds = 100`, `min_samples_leaf = 4`, `min_hessian = 0.0001`, `reg_alpha = 0.0`, `reg_lambda = 0.0`, `max_leaves = 2` (classifier default), `interactions = '3x'`, `monotone_constraints = None`, `outer_bags = 14`. Its role for `ModelForest` is as an interpretability baseline, not a member.

**NGBoost** (Duan, Avati, Ding, Thai, Basu & Ng, arXiv:1910.03225) is multiparameter boosting with one base learner **per distribution parameter** per stage, a single scalar stage scaling $\rho^{(m)}$, and **natural gradients** (inverse Fisher information) instead of ordinary gradients. Not a forest, and out of scope for a forest regression API.

### 10.5 Model-tree plus boosting hybrids

- **GBDT-PL** (Shi, Li & Li, IJCAI 2019, arXiv:1802.05640). Base learner is a PL Tree whose leaf $s$ carries $f_s(x_i) = b_s + \sum_j \alpha_{s,j} x_{i,k_{s,j}}$ over a **subset** of features chosen per leaf. The split criterion itself is linear-aware: "the reduction of $\mathcal{L}$ when splitting leaf $s$ with $x_{i,j} \le c$ **and fitting data in both child nodes**". The objective is second-order with $\Omega(t_k) = \lambda \sum_{s \in t_{k+1}} \omega(f_s)$, $\omega(f_s)$ the L2 norm of the leaf's coefficient vector, and the leaf fit is analytic in $H = \operatorname{diag}(h_i)$.
- **LinXGBoost** (de Vito 2017, arXiv:1710.03634) extends XGBoost so "each leaf stores a linear model… equivalent to piecewise regularized least-squares".
- **JOPLEn** (Raymond, Violi & Scott, IWMLSP 2024, arXiv:2405.00303) does post-hoc **joint** refit of piecewise-linear models at all leaf nodes of an already-trained ensemble.
- **LightGBM `linear_tree`** (v4.0.0+, default false) is the cheap hybrid: ordinary splits, then a linear leaf over all numerical features of that branch, regularised by `linear_lambda`.

**Why a linear-leaf booster is not a constant-leaf booster with a bigger leaf.** A constant-leaf booster approximates the target by an axis-aligned step function inside each leaf, with approximation error $O(\text{within-leaf gradient})$ and a discontinuity at every leaf boundary. A linear-leaf booster fits a local plane $b_s + \alpha_s^\top x$ per leaf, capturing first-order variation inside the leaf, reducing the leaves needed for a given accuracy and reducing boundary discontinuities — the same effect M5 got from $k = 15$. The costs are real:

1. Leaf coefficient variance grows with small leaf samples, requiring explicit L2 on $\alpha$ ($\lambda\|\alpha\|^2$ in GBDT-PL; `linear_lambda` in LightGBM). Unregularised linear leaves overfit more than constant leaves.
2. Leaf fits must be redone whenever the split set changes, and in GBDT-PL the split evaluation must refit linear models in both children, costing roughly 14–20× without the paper's own speedups.
3. Extrapolation outside the leaf's training range is unbounded unless clipped. Cubist's `LoLim`/`HiLim` and Random Model Trees' leaf-range capping are the published guards.
4. Features must be scaled to comparable mean and variance before linear fitting.
5. Monotone constraints must now constrain **leaf coefficients**, not just splits.

**Consequence for the unified API.** A linear-leaf booster is not a constant-leaf booster with a bigger leaf. Its second-order leaf solution changes from $w = -\sum g / (\sum h + \lambda)$ to a coefficient vector solving $(X^\top H X + \Lambda) b = X^\top H g$ for the leaf's regressors, and its split gain must be defined over that fit, not over a constant.

## 11. Bagging versus Boosting versus Forest

**Bagging** (Breiman 1996): bootstrap replicates of the learning set, one learner each, aggregation by averaging. Variance reduction that pays off *because* trees are unstable. No sequential dependence, no residual targeting, members highly correlated.

**Random forest** (Breiman 2001): bagging **plus** a random subset of candidate split variables per node, decorrelating the trees further. Weak trees. OOB error estimation.

**Boosting** (Friedman 2001): sequential; each new learner is fit to the negative gradient of the loss given the current ensemble; learning rate $\nu$; the ensemble is a **sum** of shrunk learners. Bias-reducing rather than variance-reducing.

**Cubist committee**: sequential (member $m$ trains on $2y - \hat y_{m-1}$), but aggregation is an **unweighted average**, not a shrunk sum, and no row or feature subsampling occurs.

| | Members | Member trained on | Aggregation | Correlation between members | Typical failure |
|---|---|---|---|---|---|
| RF of model trees | $B$ bootstrap trees | Bootstrap sample | Mean of tree outputs | Lowered by feature subsampling | Linear leaves overfit small leaves; extrapolation blow-up unless clipped |
| Cubist committee | $M$ sequential members | $2y - \hat y_{m-1}$ | Equal mean | High; target relabelling keeps members close | Error does not shrink monotonically with $M$; gains saturate |
| GBDT, constant leaves | $M$ sequential trees | Negative gradient | Sum of $\eta$-shrunk trees | Decorrelated by residual fitting | Overfits without shrinkage and early stopping |
| GBDT-PL / LightGBM linear_tree | $M$ sequential trees | Negative gradient plus linear leaf | Sum of $\eta$-shrunk trees | As GBDT | Leaf-coefficient variance; needs $\lambda$ and scaling |

## 12. Modern Forest Variants

**Mondrian forests.** Random axis-aligned partitions instead of response-optimised splits. For hyperrectangle $B$, the linear dimension is $L(B) = \sum_j \ell_j$. Under the standard Mondrian process, waiting time to split is exponential with rate proportional to $L(B)$, the split coordinate is selected with probability proportional to $\ell_j$, the threshold is uniform on that interval, and a lifetime parameter $\lambda$ limits recursion. The online implementation uses a hierarchical Bayesian nonparametric smoothing process for leaf labels and supports incremental updates whose distribution matches batch training. The 2016 regression extension places hierarchical Gaussian priors on leaf parameters, computes posterior parameters by Gaussian belief propagation, and aggregates tree predictions, reporting better uncertainty calibration than approximate GP and decision-forest alternatives on chosen large-scale tasks. **Failure modes:** splits are response-independent, so many irrelevant predictors hurt; explicit rectangles make memory linear in dimension; the online update and label-model machinery is substantial. Sources: arXiv:1406.2673; PMLR v51.

**Oblivious and equi-depth trees.** An oblivious decision tree of depth $d$ uses the same feature and threshold at every node on a given depth, can be represented as a $2^d$ leaf table, and permits level-wise parallel evaluation. For RF, feature candidates can be sampled at node or level. Liu & Loh (2019) analyse candidate-set size in an oblivious equi-depth setting and show the optimum is dataset-dependent, motivating OOB-guided search rather than a universal default. **Naming caution:** "equi-depth forest" can mean either fixed-depth oblivious trees or forests constrained to equal depth. The API should name `split_structure=standard|oblivious` and `max_depth` separately.

**Distributed and parallel RF.** Classical RF is embarrassingly parallel across trees; tree $b$ depends on independent bootstrap and feature-selection randomness, and split-feature evaluations can also be parallelised within a node. Distribution is an execution concern, not a new statistical estimator. **Reproducibility requirement:** derive each tree's seed from the forest seed plus the tree index, so thread scheduling never determines randomness.

**Streaming and online forests.** Oza & Russell's Online Bagging does not itself define a fixed-tree RF. It defines online bagging: for each incoming example and each base model $m$, draw $K_m \sim \mathrm{Poisson}(1)$, update model $m$ exactly $K_m$ times with that example, and classify by unweighted vote. Poisson(1) approximates the large-$N$ bootstrap multiplicity limit. Online RF variants combine such resampling with incremental tree learners such as Hoeffding trees; those are separate algorithms, not batch RF with a different sampler. Sources: PMLR R3; DOI 10.1145/502512.502565.

**Quantile regression forests.** Meinshausen (2006) keeps all response values at each leaf instead of only the mean. For query $x$:

$$w_i(x) = \frac{1}{B}\sum_{b=1}^{B} \frac{\mathbf{1}\{x_i \text{ and } x \text{ share leaf } b\}}{|L_b(x)|}, \qquad \widehat F(y \mid x) = \sum_i w_i(x) \mathbf{1}\{y_i \le y\},$$

with the quantile obtained by inverting the weighted empirical CDF. **OOB quantiles require per-tree in-bag counts** to compute correct forest weights. ranger supports this through `quantreg=TRUE` and `keep.inbag=TRUE`.

GRF's version differs in an important way: because $\psi$ is discontinuous, GRF uses a **heuristic pseudo-outcome** $\rho_i = \mathbf{1}(\{Y_i > \hat\theta_{q,P(X_i)}\})$ and, for several quantiles at once, labels each observation by the interval $[\hat\theta_{q_{j-1},P}, \hat\theta_{q_j,P})$ it falls into and uses a **multiclass** split rule, which avoids crossing quantile estimates in finite samples. Honest forests produce visibly smoother sample paths.

**Random survival forests.** Ishwaran, Kogalur, Blackstone & Lauer (2008): draw a bootstrap sample; at a node randomly select $m_{\text{try}}$ candidate covariates; evaluate survival split rules (log-rank, conservation of events, standardised log-rank score, random log-rank split point); grow the tree deeply; at a terminal node estimate the Nelson–Aalen cumulative hazard; average terminal-node cumulative hazards over trees. DOI 10.1214/08-AOAS169.

**Hothorn survival ensembles.** A distinct method, often conflated with Cox forests: estimate a censoring survival function $G$, optionally with a Cox model; censored observations receive zero loss weight; draw weighted bootstrap case counts; grow weighted regression trees; predict survival time through weighted leaf combination. It is not primarily a Cox-partial-likelihood-split forest. DOI 10.1093/biostatistics/kxj011. Separately, Hothorn, Hornik & Zeileis (2006) define conditional-inference forests supporting Cox responses with permutation-test split selection, avoiding CART's scale and category selection bias. DOI 10.1198/106186006X133933.

**Forest-based feature selection** is a meta-algorithm, not a new estimator: fit RF, rank features by MDI or permutation importance, keep the top $k$, refit and evaluate, repeat or search $k$, optionally report stability across resamples. VSURF separates relevant-plus-possible-redundants from strong-relevant-without-weak-redundant-companions. A 2025 review of 12 RF-based selection methods found rankings often disagree and downstream performance varies, warning against blind selection.

**RF with linear model leaves.** Two constructions must be distinguished. **True linear-leaf RF** replaces the constant leaf prediction with a fitted linear model; the split criterion should measure loss improvement under the same leaf model, not CART impurity used for constant leaves. Safeguards: ridge shrinkage at small leaves, minimum observations per fitted model, restricted leaf covariates, constrained extrapolation, aggregation of tree-level linear predictions. RaFFLE 2025 uses PILOT piecewise-linear organic trees, considers constant, univariate linear, broken-line, and piecewise-linear models, selects by BIC, uses bootstrap, samples predictors separately at each node, and weakens regularisation and removes the broken-line model to increase tree diversity; it reports superiority over CART, classical RF, Lasso, Ridge, and XGBoost across 136 regression datasets. arXiv:2502.10185.

**Local linear forest** is not literally a linear model in every leaf: it fits a local linear smoother and then uses RF machinery to estimate residuals. Ao et al.'s "Linear Random Forest" is another implementation aimed at logging regression. DOI 10.1016/j.petrol.2018.11.067.

## 13. Interpretability Limits of Forests

**Proximity measures.** Breiman proximity for observations $i, j$:

$$\operatorname{prox}(i,j) = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\{i, j \text{ reach same terminal leaf in tree } b\}.$$

Diagonal set to 1. Used for clustering and multidimensional scaling, outlier detection, missing-value imputation, and local similarity. Proximity depends on forest depth, leaf sizes, training membership, and forest randomness. Original in-bag proximity can favour observations appearing together in bootstrap samples; OOB-only proximity variants reduce this effect.

**Partial dependence and individual effects.** Partial dependence averages prediction over selected features while marginalising others; it can extrapolate over unsupported feature combinations and may hide interactions. Individual conditional expectation curves retain local variation.

**SHAP and TreeSHAP.** TreeSHAP computes exact Shapley values for tree ensembles in $O(TLD^2)$, where $T$ is tree count, $L$ maximum leaves, $D$ maximum depth. Compared with standard path-based tree attributions, SHAP satisfies local accuracy and consistency. arXiv:1802.03888.

The limits that matter for a forest API:

1. **Correlated predictors.** Marginal SHAP can assign credit to several redundant predictors, or make counterfactual and unconditional substitutions implausible.
2. **Path-dependent versus marginal.** Path-dependent TreeSHAP need not equal marginal Shapley values. Filom et al. exhibit equivalent tree ensembles with different path-dependent TreeSHAP rankings but equal marginal values. arXiv:2302.08434.
3. **Background dependence.** Interventional marginal SHAP depends on the selected background data.
4. **Model explanation, not causal truth.** SHAP explains the fitted prediction function; it does not establish a real-world causal effect.
5. **RF-specific bias remains.** Loecher (2023) shows standard in-sample RF SHAP scores can inherit model overfitting and recommends OOB-only or shrunk variants. DOI 10.1007/s10182-023-00479-7.

**The SHAP interoperability constraint.** A forest API must expose a stable public per-tree structure, because SHAP's tree algorithm requires the tree itself, not a black-box predictor. `mltreelib`'s `ModelTree` already exposes `shapley` and `break_down`; a `ModelForest` must aggregate those across trees and expose per-tree export.

## 14. Benchmark Evidence

### 14.1 Grinsztajn et al. (2022), and a correction

The paper studied 45 curated tabular datasets, mostly around 10,000 samples, with tree models beating neural networks across random-search budgets. It identified three inductive biases: robustness to uninformative features, preservation of feature orientation rather than rotation invariance, and ability to learn irregular target functions.

**It did not establish that RF handles missing values because trees split on missing values.** The benchmark preprocessing removes columns with many missing values and then removes every row containing at least one missing entry, so all reported results use **complete-case data**. Missing-data handling is listed as a separate methodological question, not benchmarked.

**Correct claim:** the paper explains tree-model strength on typical medium-sized tabular data through irregular functions, uninformative features, and orientation, not through missing-value splitting.

**Incorrect claim:** "Grinsztajn et al. showed RF handles missing values because trees split on missing values." The paper removed missing data.

Benchmark scope: training sets capped at 10,000 samples, $d/n < 1/10$, fewer than 500 predictors, minimum 3,000 samples, IID only, classification targets reduced to the two most frequent classes and balanced.

### 14.2 McElfresh et al. (2023)

19 algorithms over 176 datasets, 538,650 trained models, 10 OpenML folds per dataset. The "NN versus GBDT" framing overstates practical differences on many datasets; for many datasets simple baselines or lightly tuned GBDT rival family-level selection. GBDTs outperform NNs on average, tend to do better on larger datasets and irregular feature distributions including skewness and heavy tails, and small datasets favour NNs in their analysis. TabPFN was exceptional, outperforming other methods on average within its 3,000-sample restriction.

### 14.3 TabArena (2025)

Manually filtered 1,053 candidate datasets down to 51 IID datasets; 16 models; approximately 25 million model instances; eight-fold inner validation and outer repeated folds; 200 tuned configurations per model plus post-hoc ensembles. Under conventional tuning, CatBoost ranks first among individual methods; after post-hoc ensembling, neural models become the strongest single-model families; RF remains a strong, inexpensive baseline but does not lead; cross-model ensembles such as AutoGluon give the strongest overall results.

### 14.4 Distribution shift

**TableShift**: 15 binary tabular tasks with natural shifts; baselines MLP, XGBoost, LightGBM, CatBoost; 100 HyperOpt trials per model selected by ID validation accuracy. ID and OOD accuracy were strongly correlated. No specialised approach consistently beat standard tree baselines. Methods reducing shift gaps often did so by sacrificing ID accuracy. No method reliably eliminated shift gaps.

A 45-method study across 86,000 configurations on five tabular datasets found inconsistent method performance across shift types and limited gains from generic robust-learning methods.

### 14.5 Default guidance

**Prefer RF when** zero or low tuning is operationally important; CPU-only or latency-sensitive training matters; labels may contain moderate noise; parallelism across trees is desired; a robust, competitive baseline suffices; or the downstream API needs probability estimates and local and global diagnostics.

**Prefer GBDT when** peak predictive accuracy is primary; data are medium-to-large with irregular, skewed, or heavy-tailed feature distributions; categorical features need native handling; a tuning budget is available; sequential boosting can be amortised.

**Under suspected shift:** do not assume RF is safer. Create validation splits resembling the target population and shift mechanism. Compare tuned RF and GBDT under the same preprocessing and metric. Stress-test missingness, category levels, and feature-range extrapolation. Prefer target-population data or shift-aware resampling over algorithmic robustness wrappers.

**There is no timeless "RF beats GBDT" ordering.** RF is low-tuning, fast, parallel, and robust. GBDT often reaches higher peak accuracy on irregular, medium-to-large tabular data. Validation protocol, tuning budget, and ensembling can reorder families.

## 15. Competing and Adjacent API Surfaces

### 15.1 scikit-learn

Full method matrix across the ensemble family:

| Estimator | fit | predict | predict_proba | predict_log_proba | apply | decision_path | feature_importances_ | oob_score param | oob_decision_function_ | estimators_ | n_estimators_ | random_state param |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForestClassifier | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | — | Y |
| RandomForestRegressor | Y | Y | — | — | Y | Y | Y | Y | — | Y | — | Y |
| ExtraTreesClassifier | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | — | Y |
| ExtraTreesRegressor | Y | Y | — | — | Y | Y | Y | Y | — | Y | — | Y |
| RandomTreesEmbedding | Y | — | — | — | Y | Y | Y | Y | — | — | Y | — | Y |
| IsolationForest | Y | Y | — | — | — | — | — | Y | — | — | Y | — | Y |
| GradientBoostingClassifier | Y | Y | Y | Cond. | Y | — | Y | Y | — | — | Y | Y | Y |
| GradientBoostingRegressor | Y | Y | — | — | Y | — | Y | Y | — | — | Y | Y | Y |
| HistGradientBoostingClassifier | Y | Y | Y | — | — | — | — | Y | — | — | — | — | Y |
| HistGradientBoostingRegressor | Y | Y | — | — | — | — | — | Y | — | — | — | — | Y |
| VotingClassifier | Y | Y | Cond. | — | — | — | — | Y | — | — | Y | — | — |
| VotingRegressor | Y | Y | — | — | — | — | — | Y | — | — | Y | — | — |
| StackingClassifier | Y | Y | Y | — | — | — | — | Y | — | — | Y | — | — |
| StackingRegressor | Y | Y | — | — | — | — | — | Y | — | — | Y | — | — |
| BaggingClassifier | Y | Y | Y | Y | — | — | — | Y | Y | Y | Y | Y | Y |
| BaggingRegressor | Y | Y | — | — | — | — | — | Y | Y | — | Y | Y | Y |

Qualifications that a reimplementation must get right:

1. `RandomForestClassifier.predict_log_proba` computes the log of **mean** tree probabilities, not mean log probabilities.
2. `GradientBoostingClassifier.predict_proba` and `predict_log_proba` exist only when the selected loss supports probabilities.
3. `VotingClassifier.predict_proba` requires `voting="soft"`; hard voting returns labels.
4. `BaggingClassifier.predict_proba` falls back to vote proportions if base estimators lack `predict_proba`.
5. `HistGradientBoostingClassifier` exposes `n_iter_`, not `n_estimators_`; no public `estimators_`, `apply`, or `decision_path`.
6. Random-forest classes document the `n_estimators` constructor parameter and `estimators_` fitted list, but **not** `n_estimators_`. Bagging and classic Gradient Boosting document fitted `n_estimators_`.
7. Forest OOB decisions can contain `NaN` when a small ensemble leaves a row never out-of-bag.
8. `IsolationForest` exposes `decision_function(X)` and `score_samples(X)`, not `predict_proba`.

Key defaults worth matching or consciously diverging from:

```python
RandomForestClassifier(
    n_estimators=100, *, criterion="gini", max_depth=None,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.0,
    bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
    verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
    max_samples=None, monotonic_cst=None,
)
```

Regressor differences: `criterion="squared_error"`, `max_features=1.0`, no `class_weight`. Classifier `max_features="sqrt"` means `sqrt(n_features)`; regressor `1.0` means all features, equivalent to bagged trees.

ExtraTrees differs from RandomForest only in `bootstrap=False` by default, plus `max_features="sqrt"`.

```python
GradientBoostingClassifier(
    *, loss="log_loss", learning_rate=0.1, n_estimators=100, subsample=1.0,
    criterion="deprecated", min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
    init=None, random_state=None, max_features=None, verbose=0,
    max_leaf_nodes=None, warm_start=False, validation_fraction=0.1,
    n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0,
)
```

OOB behaviour differs from RF: `oob_scores_`, `oob_score_`, and `oob_improvement_` exist only when `subsample < 1.0`; there is no `oob_decision_function_`.

```python
HistGradientBoostingClassifier(
    loss="log_loss", *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
    max_depth=None, min_samples_leaf=20, l2_regularization=0.0,
    max_features=1.0, max_bins=255, categorical_features="from_dtype",
    monotonic_cst=None, interaction_cst=None, warm_start=False,
    early_stopping="auto", scoring="loss", validation_fraction=0.1,
    n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None,
    class_weight=None,
)
```

**Model-kind switching precedent inside scikit-learn: none.** It offers separate estimator classes plus meta-estimators. Two narrower precedents exist: LightGBM's `boosting_type="gbdt" | "dart" | "rf"` and ranger's `splitrule` variants. Both switch related training regimes inside one family; neither unifies unrelated ensemble APIs.

### 15.2 R ecosystem

**ranger**: `ranger(dependent.variable, formula=, data=, num.trees=500, num.random.splits=1, mtry, max.depth=0, min.node.size=1, sample.fraction=NULL, replace=TRUE, case.weight, na.action=na.learn, splitrule=NULL, alpha=0.5, minprop=0.1, x.replace=FALSE, importance="none", write.forest=TRUE, ..., seed=NULL, num.threads=NULL, verbose=TRUE, inbag=NULL, keep.inbag=FALSE, respect.unordered.factors=NULL, na.omit=FALSE, save.memory=FALSE, ...)`.

**randomForest**: `randomForest(x, y=NULL, xtod=NULL, ytod=NULL, data=NULL, subset, na.action=na.omit, mtry=NULL, ntnode=NULL, forest=TRUE, ntree=500, replace=TRUE, classwt=NULL, sampsize=NULL, nodesize=NULL, maxnodes=NULL, x=FALSE, y=FALSE, importance=FALSE, ...)`. The `replace=FALSE` case draws `sampsize` samples without replacement.

**party / partykit**: `ctree(formula, data, controls = ctree_control(), ...)`, with `ctree_control(testtype, mincriterion, minsplit, minbucket, maxdepth, maxsurrogate, mtry, maxvars)`. The conditional-inference framework supports permutation-test split selection, avoiding CART's scale and category selection bias.

**CRAN `Cubist`**, the reference for Cubist parity:

```r
cubist(x, y, committees = 1, control = cubistControl(), weights = NULL, ...)
cubistControl(unbiased = FALSE, rules = 100, extrapolation = 100,
              sample = 0, seed = sample.int(4096, size = 1) - 1L,
              label = "outcome", strip_time_stamps = TRUE)
predict(object, newdata = NULL, neighbors = 0, ...)
```

`committees = 1` gives one M5-style model; `committees > 1` adds committee boosting; `neighbors = 0` gives rule-model prediction while `neighbors = 1..9` applies nearest-training-instance correction at prediction time; `rules = 100` is the maximum rules per committee; `sample = 0` means no random subset for model building; `extrapolation = 100` controls correction of rule-model predictions outside the training range; `unbiased = FALSE` controls the unbiased error estimate. **There is no `smoothing` argument** in the current public interface.

**rpart**: `rpart(formula, data, weights, subset, na.action=na.rpart, method, model=FALSE, x=FALSE, y=TRUE, parms, control, cost, ...)` with `method` in `"anova"`, `"poisson"`, `"class"`, `"exp"`, or custom split functions, and `cost` assigning nonnegative split costs per variable.

**gbm**: `gbm(formula, data, distribution = "bernoulli", n.trees = 100, interaction.depth = 1, n.minobsinnode = 10, shrinkage = 0.001, n.folds = 10, max.cat.codes = 32, ...)` with `bag.fraction` and `train.fraction` controlling stochasticity.

**xgboost**, **lightgbm**, **grf**, **econml**: see the cited documentation.

### 15.3 Python ecosystem

**EconML** (`py-why/EconML`): classes are `econml.dml.{DML, LinearDML, SparseLinearDML, KernelDML, NonParamDML, CausalForestDML}`, `econml.dr.{DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner}`, `econml.iv.dml.{OrthoIV, DMLIV, NonParamDMLIV}`, `econml.iv.dr.{DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV, IntentToTreatDRIV, LinearIntentToTreatDRIV}`, `econml.orf.{DMLOrthoForest, DROrthoForest}`, `econml.grf.CausalIVForest`. **There is no `PartiallyLinear`, `ForestDML`, `CausalForestDMLIV`, `GRS`, or `DML/Poisson` class.**

`CausalForestDML` documented signature includes `model_y='auto'`, `model_t='auto'`, `featurizer=None`, `treatment_featurizer=None`, `discrete_outcome=False`, `discrete_treatment=False`, `categories='auto'`, `cv=2`, `mc_iters=None`, `mc_agg='mean'`, `drate=True`, `n_estimators=100`, `criterion='mse'`, `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=5`, `min_weight_fraction_leaf=0.0`, `min_var_fraction_leaf=None`, `min_var_leaf_on_val=False`, `max_features='auto'`, `min_impurity_decrease=0.0`, `max_samples=0.45`, `min_balancedness_tol=0.45`, `honest=True`, `inference=True`, `fit_intercept=True`, `subforest_size=4`, `n_jobs=-1`, `random_state=None`, `verbose=0`. **No `num_trees`, `sub_sample`, `ortho_norm`, `control_overrides`.**

Mechanism: it fits $q(X,W)$ and $f(X,W)$ **cross-fitted**, computes $\tilde Y = Y - q$ and $\tilde T = T - f$, then fits a subsampled honest forest solving the local moment $E[(Y - E[Y|X,W] - \langle\theta(x), T - E[T|X,W]\rangle - \beta(x))(T;1) \mid X=x] = 0$. Split criteria `'mse'` and `'het'` use the GRF-style heterogeneity criterion with the local Jacobian $\rho(\text{child}) = E[(T;1)(T;1)^\top \mid X \in \text{parent}]^{-1} E[(Y - \langle\theta,T\rangle - \beta)(T;1) \mid X \in \text{child}]$.

Honesty implementation: shuffle indices, `samples_train, samples_val = inds[:n//2], inds[n//2:]`. With `inference=True`, `max_samples` must be an integer smaller than `n_samples//2` or a float at most 0.5. `min_var_leaf_on_val=True` "partially violates the honesty structure."

Inference is bootstrap-of-little-bags covariance plus an objective-Bayes positive-variance correction. **`groups=` exists in `fit()` only to keep grouped rows together in cross-fitting folds; it does not give cluster-robust covariance.** `StatsModelsInference` accepts only `cov_type='nonrobust'|'HC0'|'HC1'`.

**CausalML** (`uber/causalml`): classes are `BaseSLearner`, `BaseTLearner`, `BaseXLearner`, `BaseRLearner`, `BaseDRLearner`, `BaseDRIVLearner`, `IVRegressor`, `TMLELearner`, plus `CausalTreeRegressor`, `CausalRandomForestRegressor`, `UpliftTreeClassifier`, `UpliftRandomForestClassifier`. **No `RLearner`, `DRLearner`, `DML`, `ForestDML`, `CausalForestDML` classes.**

`CausalRandomForestRegressor` defaults from source: `n_estimators=100`, `criterion='causal_mse'`, `alpha=0.05`, `min_samples_split=60`, `min_samples_leaf=100`, `min_group_samples=50`, `max_features=1.0`, `groups_penalty=0.5`, `bootstrap=True`, `honesty=True`, `estimation_sample_size=0.5`, `cv_folds=5`.

Honesty trade-off stated in the official docs: "Honest estimation removes this bias by splitting the sample in two ... **at the cost of fitting each half with less data**. ... The trade is variance: each half sees only part of the data, so an honest tree is shallower and noisier per leaf. Pass `honesty=False` when halving the sample leaves too little to split on." CausalML's honest split is **treatment-stratified**, unlike GRF's. **There is no `clusters=` argument anywhere in the CausalML API.**

**interpretML**: `ExplainableBoostingClassifier` with the round-robin, very-low-learning-rate training described above.

**grf Python**: no official Python package. `grf-labs/grf` ships R plus a C++ core. Upstream issue #257 "Any python implementations?" is open. The closest Python wrapper is third-party `skgrf` on PyPI.

### 15.4 The three unification patterns, and the recommendation

**scikit-learn estimator contract.** Constructor stores parameters only; `fit(X, y)` learns and returns `self`; `predict(X)` infers; `get_params(deep=True)` and `set_params(**params)` provide generic cloning and grid-search surface. Public estimator tags, exposed since sklearn 1.6, describe capabilities such as sparse support, output types, and supported methods. Meta-estimators accept child estimator lists and expose `named_estimators_`. Strongest precedent for generic method names and nested estimator parameters. Not `kind`-switch precedent.

**tidymodels / parsnip.** Specification decomposes into three axes: **model type** (`rand_forest`, `boost_tree`), **mode** (regression, classification, censored regression, quantile regression), and **engine** (implementation package: `ranger`, `randomForest`, `spark`). Main argument names are harmonised across engines (`trees`, not `ntree`, `num.trees`, or `n.trees`); engine-specific arguments remain unchanged. This is **three-axis specialisation**, not a `kind` switch. It avoids one parameter name meaning different concepts across model types.

**Julia MLJ.** Each model implements generic `fit(model, verbosity, X, y, w)` and `predict(model, fitresult, Xnew)`. Model traits include `is_supervised`, `prediction_type` (`:probabilistic`, `:deterministic`, `:unknown`), `input_scitype`, `target_scitype`, `predict_scitype`, `output_scitype`, `transform_scitype`, `supports_weights`, `supports_class_weights`, `reports_feature_importances`, and `implemented_methods`. A registry filters models by traits and data compatibility. Measures are independent abstractions; `evaluate(model, X, y, resampling=CV(), measure=rms)` dispatches the compatible prediction operation based on model prediction type. `machine(model, X, y)` stores model plus data; `fit!(mach)`; `predict(mach, Xnew)`. `EnsembleModel(model; weights, bagging_fraction, rng, n, parallel, out_of_bag_measure)` wraps any atomic model as an ensemble.

**Verdict on a `kind`-switched facade.** Exact precedent exists for LightGBM's `boosting_type` and ranger's `splitrule`. No mature precedent exists for one facade unifying RF, ExtraTrees, classic GBDT, HistGBDT, Cubist committee, random-tree embedding, isolation forest, voting, and stacking through one `kind` enum.

The recommended reference pattern combines all three:

```text
ModelForest
  kind: ForestKind              # rf, extra_trees, committee, grf, gbdt
  mode: ModelMode               # regression, classification
  parameters: common + kind-specific validated config
  capabilities/traits:
    supports_probability
    supports_oob
    supports_apply
    supports_decision_path
    supports_feature_importance
    supports_honesty
    supports_inference
  fit(X, y, ...)
  predict(X)
  predict_proba(X)
  predict_log_proba(X)
  apply(X)
  decision_path(X)
  feature_importances_
  estimators_
  n_estimators_
  oob_score_
  oob_decision_function_
```

Key rule: keep `kind` algorithm-specific and `mode` task-specific. Use trait queries to guard unsupported methods. Avoid one flat parameter union where every kind accepts irrelevant fields.

**Note on `get_params`.** `mltreelib` has already decided, in Spec 25, to drop sklearn's `BaseEstimator` parameter protocol: no `get_params`, no `set_params`, no cloning, no `deep`, no generic `score`, with `config()` replacing `get_params` for inspection. The pattern above still applies, minus the sklearn-specific parts. **The `sklearn` import and dependency prohibition is unchanged.**

### 15.5 Modern challengers, 2023–2026

TabPFN v1 (Hollmann et al., Nature 2025, DOI 10.1038/s41586-024-08328-6) and v2 (arXiv:2412.17174) are in-context-learning tabular foundation models with a restricted data regime. TabICL (arXiv:2502.05564) is a retrieval-augmented tabular in-context classifier. TabDPT (arXiv:2601.09778) is a retrieval-based tabular foundation model. ModernNCA (arXiv:2504.08973) is a deep metric-learning model. **All are OUT_OF_SCOPE for a native forest API**: they have external checkpoints, an inference stack, and a different fit lifecycle. Their relevance is as benchmark challengers. The honest summary of their claims is that they compete within a data-size regime and pay an inference cost that forests do not; a curated living benchmark (TabArena) still found RF a strong, inexpensive baseline that does not lead.

### 15.6 The `grf` implementation reference

`https://github.com/grf-labs/grf/blob/master/REFERENCE.md` is the authoritative algorithm specification. Facts a Rust port must match:

- **`mtry` is Poisson.** "the number of variables considered is actually drawn from a poisson distribution with mean equal to `mtry`. A new number is sampled from the distribution before every tree split." This is exactly the GRF $\pi$-condition, and it is why `mtry` need not be an integer.
- **Honest leaf repopulation.** The tree is grown on the split half; the estimation half is **pushed down** the finished tree and each observation is added to the leaf it lands in. Empty leaves are pruned by default (`honesty.prune.leaves=TRUE`); with `FALSE`, trees with an empty leaf are skipped for affected test points.
- **Split balance in causal forests.** `min.node.size` requires at least `min.node.size` samples **on each side of the parent's mean treatment**, not simply per arm. `alpha` and `imbalance_penalty` use the information-content node size $\sum_{i \in \text{node}} (W_i - \bar W)^2$. `imbalance_penalty` multiplies the split score by $1/\text{size(left)} + 1/\text{size(right)}$. `alpha \in (0, 0.25)`, default 0.05.
- **Variance estimation.** Trees are grown in **groups** sharing a half-sample; `ci.group.size` (default 2) trees per group; prediction variance compares within-group and across-group variation. Because training starts from a half-sample, **`sample.fraction` cannot exceed 0.5 when variance estimation is on.**
- **Cluster-robust training.** Sampling operates on cluster IDs; `samples_per_cluster` drawn per selected cluster; honesty splits the **cluster ID set** in half; OOB is cluster-level.
- **Sample weights.** Honoured in splitting and prediction for most forest types; **not** for local linear forests and quantile forests; for `survival_forest` only at prediction time.
- **Missing values.** Handled by the MIA criterion (Twala, Jones & Hand 2008) — three candidate splits per threshold (missing to left, missing to right, missing separated), default missing to left.
- **Categorical inputs.** Numerical only; grf recommends `sufrep`, one-hot, or hashing.
- **Scale.** "a `causal_forest` with 1 000 000 observations and 30 continuous covariates takes around 1 hour to train (using default settings on a machine with 24 cores and 150 GB RAM), and the final forest takes up ca. 25 GB of memory (this is expected to scale linearly in the number of trees)."
- **Reproducibility.** The same seed is not enough across compilers and architectures; pre-2.4.0 the thread count changed results; rounding inputs to at most 8 significant digits helps.
- **Tuning.** `tune.parameters="all"` tunes `sample.fraction`, `mtry`, `min.node.size`, `honesty.fraction`, `honesty.prune.leaves`, `alpha`, `imbalance.penalty`. For causal forests the OOB error is the **R-loss**. `honesty.fraction` is searched in 0.5–0.8.
- **Verified `causal_forest()` signature and defaults.** `causal_forest(X, Y, W, Y.hat=NULL, W.hat=NULL, num.trees=2000, sample.weights=NULL, clusters=NULL, equalize.cluster.weights=FALSE, sample.fraction=0.5, mtry=min(ceiling(sqrt(ncol(X))+20), ncol(X)), min.node.size=5, honesty=TRUE, honesty.fraction=0.5, honesty.prune.leaves=TRUE, alpha=0.05, imbalance.penalty=0, stabilize.splits=TRUE, ci.group.size=2, tune.parameters="none", tune.num.trees=200, tune.num.reps=50, tune.num.draws=1000, compute.oob.predictions=TRUE, num.threads=NULL, seed=runif(1,0,.Machine$integer.max))`.
- **Forest weights.** For each training sample and each tree, record whether the test point fell in the same leaf; the similarity weight is $\alpha_i(x) = \sum_b \mathbf{1}\{\text{same leaf in tree } b\}/|L_b(x)|$, normalised to sum to one. `get_forest_weights()` returns a **sparse** matrix. Cost per test point is $O(\sum_b |L_b(x)|)$ in the worst case.

### 15.7 What is genuinely hard in a pure-Rust port

**Mechanical, build now.** Subsampling without replacement; per-tree group structure; Poisson `mtry` draws; split candidate enumeration with cumulative sums; honest split-and-repopulate with leaf-to-training-index lists and empty-leaf pruning; forest weights by leaf co-occurrence with sparse storage and OOB weights; gradient relabelling given a $\psi$ interface; local estimating-equation solve given weights; closed-form ridge local-linear solve; parallel tree construction with deterministic RNG plumbing.

**Genuinely hard, design carefully.**

1. **The doubly robust nuisance layer.** Cross-fitting with leakage control, group-aware folds, MC repeats and aggregation, leave-one-out nuisance predictions used inside tree growing, and AIPW score construction with propensity truncation (`min_propensity=1e-6` in EconML).
2. **Orthogonalisation.** Residualising $Y$ and $W$ with the correct forest-weighted OOB predictions *before* the causal forest is grown. Getting this wrong silently reintroduces confounding.
3. **Cluster-local averaging.** Cluster-level subsampling, cluster-level OOB, cluster-level equal weighting, and cluster-robust sandwich variances. Neither EconML nor CausalML ships this; grf does, and it is the least portable piece.
4. **Bootstrap-of-little-bags variance** with the within/between-bag ANOVA decomposition and the objective-Bayes non-negativity correction.
5. **Numerical stability of the local Jacobian.** $A_P^{-1}$ per node, moment aggregation for multi-treatment ($K^2$) and multi-outcome ($M$) cases, and the $\rho(\text{child})$ criterion. This is where the Cython implementations live.
6. **Inference-valid test-point-specific forests.** Wager–Athey Remark 4 may require a per-$x$ subset of trees, which changes the API shape: prediction must know which trees are "regular at $x$".
7. **Excluding the forest's own Monte-Carlo error** from the reported variance, and the platform-reproducibility traps.

## 16. Portfolio Verdicts

### 16.1 Forest kinds for the unified API

| Kind | Verdict | Reason |
|---|---|---|
| Random forest (`rf`) | **BUILD_NOW** | Core, mature, high-value default; reuses the native CART tree core |
| ExtraTrees split rule and variant (`extra_trees`) | **BUILD_NOW** | Small change to the split search; material speed and diversity option |
| Cubist-style committee (`committee`) | **BUILD_NOW** | Roughly 30 lines over the existing tree builder; no new data structures; large accuracy gain on smooth targets |
| Quinlan/M5 backup smoothing `k = 15` | **BUILD_NOW** | Pure linear algebra on stored node and leaf models |
| Cubist variance-shrink smoothing | **BUILD_LATER** | Needs per-node residual variance and covariance bookkeeping; only worth it for bit-compat with Cubist |
| Cubist composite k-NN adjustment, exact formulas | **BUILD_NOW** | `predict_adjusted` exists; needs the exact distance, weight, adjusted-value, fallback, and auto-NN selection rules |
| SD-based SDR split rule | **BUILD_NOW** | `GainFn` abstraction exists; add `sdr` on residuals |
| Cubist constants `MINSPLIT=3`, `MINITEMS`, `MAXRULES` | **BUILD_NOW** | Constants plus documentation; expose both CLI and R defaults |
| Extrapolation and range limit plus non-negative sign rule | **BUILD_NOW** | Cheap post-hoc clip in the predict path |
| Case-weighting semantics | **BUILD_NOW** | `fit_weighted` exists; only the Cubist normalisation rule must be implemented |
| Probability forest | **BUILD_LATER** | ranger shows the value; requires leaf probability aggregation semantics |
| Honesty with per-tree re-randomised halves | **BUILD_NOW** | Prerequisite for any inference-valid claim; already partly present in `ModelTreeParams` |
| GRF forest weights and local estimating equation | **BUILD_NOW** | Core abstraction that subsumes regression, quantile, and causal forests; the gradient split reuses CART cumulative sums |
| GRF gradient-based splitting | **BUILD_NOW** | One-pass cumulative sums, same as CART |
| Local linear forest | **BUILD_LATER** | Needs ridge at every node plus a per-test-point linear solve; payoff is a better rate, not new capability |
| GRF variance decomposition and BLB | **BUILD_LATER** | Needs grouped-tree training and the Bayesian ANOVA positivity fix |
| Infinitesimal jackknife | **BUILD_LATER** | Consistency is asymptotic; needs per-tree estimates and the $n(n-1)/(n-s)^2$ correction |
| Causal forest difference-in-means leaf estimator | **BUILD_NOW** | Trivial given the forest weights; the baseline any DR layer must beat |
| AIPW scores and ATE/BLP summaries | **BUILD_LATER** | Needs nuisance cross-fitting plus cluster-robust sandwich |
| Orthogonalised splitting and R-learner residualisation | **BUILD_NOW** | Small and high value: two nuisance forests plus residual training data |
| Quantile forests | **BUILD_NOW** | Simple specialisation of the relabeling interface; multi-quantile via multiclass relabelling |
| IV forests | **BUILD_LATER** | Two moments plus instrument centring plus compliance score |
| Multi-treatment and multi-outcome forests | **BUILD_LATER** | Vector moments; memory $O(\#\text{nodes}\cdot M\cdot K^2)$ per tree; cap the arm count |
| Friedman GBDT, constant leaves, Newton leaf update | **BUILD_NOW** | Core of the third pillar; reuses the tree builder with a constant leaf plus `l2` |
| Friedman stochastic boosting | **BUILD_NOW** | One RNG stream plus row subsampling per stage |
| Leaf-wise growth | **BUILD_NOW** | Trivial over an existing grower |
| Monotone constraints, split finding plus leaf-value projection | **BUILD_NOW** | `tree_monotone` and `leaf_monotone` already exist |
| Interaction constraints | **BUILD_NOW** | `interaction_groups` already present; enforce in split search and leaf regressors |
| Histogram binning | **BUILD_LATER** | Large engineering; big speed win, no accuracy gain |
| Ordered target statistics and ordered boosting | **BUILD_LATER** | High complexity, high accuracy on categorical-heavy data |
| GBDT-PL, linear leaves with linear-aware split evaluation | **BUILD_LATER** | Leaf Newton step must solve a coefficient system and split evaluation must refit both children |
| LightGBM-style `linear_tree` | **BUILD_LATER** | Cheaper than GBDT-PL and a good first cut |
| LinXGBoost-style piecewise regularised least squares | **BUILD_LATER** | Same machinery as the two above with a different regulariser; fold into one `LeafKind::Linear` path |
| JOPLEn-style joint post-hoc refit | **BUILD_LATER** | Needs a global least squares over all leaves; a nice upgrade path for trained boosters |
| RF with linear model leaves | **BUILD_NOW** | Exploits the existing M5/Cubist core; closest fit to library identity |
| Forest recursive feature selection | **BUILD_NOW** | Useful companion API, but expose ranking stability and selection bias |
| Exact TreeSHAP engine | **BUILD_LATER** | Separate explainability subsystem; do not bury it in the fitting API |
| Proximity matrix | **BUILD_LATER** | Valuable but $O(n^2)$ memory |
| Mondrian forest | **BUILD_LATER** | Attractive streaming and uncertainty option, but new partition and label-model machinery |
| Oblivious / equi-depth trees | **BUILD_LATER** | Good parallelism option, but specialised API and training semantics |
| Random survival forest | **ORACLE_ONLY** | Survival response, censoring, and cumulative-hazard output exceed ordinary classification and regression scope |
| Hothorn survival ensemble, conditional Cox forest | **ORACLE_ONLY** | Censoring weights and separate survival-model semantics need dedicated design |
| Online / Oza-Russell forest | **ORACLE_ONLY** | Streaming update lifecycle differs from a batch `ModelForest` fit and predict API |
| Survival forest and causal survival forest | **ORACLE_ONLY** for now | Needs censoring-robust orthogonal estimating equations and a survival sub-model |
| Policy learning from forests | **OUT_OF_SCOPE** | Separate artifact, not a forest estimator |
| Isolation forest | **OUT_OF_SCOPE** | Different task: anomaly scoring, labels, contamination threshold |
| EBM / GA2M | **ORACLE_ONLY** | Additive glassbox model, not a forest family; interpretability comparator |
| NGBoost | **OUT_OF_SCOPE** | Probabilistic output with per-parameter base learners |
| MOTR-BART and other Bayesian model trees | **OUT_OF_SCOPE** | MCMC posterior inference; wrong complexity class |
| TabPFN, TabICL, TabDPT, ModernNCA | **OUT_OF_SCOPE** | External checkpoints, inference stack, different lifecycle |
| Distributed RF | **OUT_OF_SCOPE** | Statistical method unchanged; a runtime parallelisation concern, not a separate estimator |
| Random-projection forest, Forest-RC | **OUT_OF_SCOPE** | Breimin's linear random-feature extension; optional research variant |
| "Brezinski et al. 2004 adaptive gradient boosting" | **ORACLE_ONLY** | No resolvable record; cannot be specified or cited until located |
| Bühlmann & Yu (2003) asymptotics | **OUT_OF_SCOPE** | Theory only; quote qualitatively, never rates without the text |

### 16.2 What the API must not claim

1. OOB error is **not** unbiased. It is a diagnostic with parameter-dependent optimism, especially with balanced classes, small $n$, many predictors, weak signal, and low predictor correlation.
2. GRF and causal forest intervals are **asymptotic**, pointwise, and require **undersmoothing**. There is no finite-sample coverage guarantee.
3. Undersmoothing is required, and edge effects remain. The GRF paper says so directly.
4. Regularity cannot hold simultaneously at all test points. A per-test-point tree subset may be needed, and the API must expose which trees were used.
5. Overlap and unconfoundedness are **assumptions on the data**, not checkable properties of the estimator. A propensity histogram is a diagnostic, not a guarantee.
6. MDI is biased toward high-cardinality, continuous, many-cut-point predictors. Permutation importance has its own correlated-predictor problems. Neither is causal importance.
7. SHAP explains the fitted prediction function. It is not a causal statement, and path-dependent TreeSHAP need not equal marginal Shapley values.
8. Cubist committees are **not** gradient boosting. The aggregation is an unweighted average, not a shrunk sum.
9. GRF forests are **not** random forests with a better split rule. The tree predictions are never averaged; a weighted estimating equation is solved.
10. Honesty and double robustness are **not** substitutes. They fix different biases.

## 17. Design Implications for `ModelForest`

1. **Separate forest algorithm from split rule.** `kind` selects the algorithm family; `split_rule` selects gini, variance, hellinger, logrank, maxstat, beta, poisson, sdr, or gradient.
2. **Make randomness explicit and reproducible.** Forest seed, per-tree derived seed, bootstrap versus subsample, sample fraction, `mtry` including the Poisson variant, random threshold count. Thread scheduling must never determine randomness.
3. **Preserve in-bag count vectors**, not only OOB masks. Multiplicities matter for quantile forests and for stratified sampling.
4. **Keep honesty and orthogonalisation as independent switches.** The doubly-robust layer sits above the forest.
5. **Keep `kind` algorithm-specific and `mode` task-specific.** Follow the parsnip three-axis pattern.
6. **Use trait queries to guard unsupported methods**, following the MLJ pattern.
7. **Treat missing values and unordered factors as explicit, named policies** with per-node state persisted in the model.
8. **Expose a stable public per-tree structure** so SHAP, inspection, pruning, serialisation, and audit all work.
9. **Report what the method actually guarantees.** The inference section appears only for methods with a defensible path, and it states the asymptotic conditions.
10. **Name the Cubist lineage honestly.** `committee` for Cubist, `gbdt` for Friedman, `rf` for Breiman, `grf` for Athey-Tibshirani-Wager. Never `boosted` for Cubist.
11. **Keep the sklearn import prohibition.** No sklearn import, no sklearn dependency, no `get_params`/`set_params`/`clone`/`deep`/generic `score`. `config()` for inspection.
12. **One multi-core job at a time.** Forest training parallelises across trees, so it must not run concurrently with another multi-core fit.

## 18. Citation List

1. Breiman, L. (2001). "Random Forests." *Machine Learning* 45:5–32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
2. Breiman, L. (2001). "Out-of-Bag Estimation." [Author PDF](https://www.stat.berkeley.edu/~breiman/OOBestimation.pdf)
3. Breiman, L.; Cutler, A. (2003). *Using Random Forests v4.0*. [Manual PDF](https://www.stat.berkeley.edu/~breiman/Using_random_forests_v4.0.pdf)
4. Breiman, L. (1996). "Bagging Predictors." *Machine Learning* 24:123–140. [https://www.stat.berkeley.edu/~breiman/bagging.pdf](https://www.stat.berkeley.edu/~breiman/bagging.pdf)
5. Geurts, P.; Ernst, D.; Wehenkel, L. (2006). "Extremely Randomized Trees." *Machine Learning* 63:3–42. [https://doi.org/10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1)
6. Wright, M. N.; Ziegler, A. (2017). "ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R." *JSS* 77(1):1–17. [https://doi.org/10.18637/jss.v077.i01](https://doi.org/10.18637/jss.v077.i01)
7. Wright, M. N.; Ziegler, A. "ranger" manuscript. [arXiv:1508.04409](https://arxiv.org/abs/1508.04409)
8. Wright, M. N. *ranger* 0.18.0 manual. [CRAN PDF](https://cran.r-project.org/web/packages/ranger/ranger.pdf)
9. Athey, S.; Tibshirani, J.; Wager, S. (2019). "Generalized Random Forests." *Ann. Statist.* 47(2):1148–1178. [https://doi.org/10.1214/18-AOS1709](https://doi.org/10.1214/18-AOS1709) · [arXiv:1610.01271v4](https://arxiv.org/html/1610.01271v4) · [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Generalized-random-forests/10.1214/18-AOS1709.full)
10. Wager, S.; Athey, S. (2018). "Estimation and inference of heterogeneous treatment effects using random forests." *JASA* 113(523):1228–1242. [https://doi.org/10.1080/01621459.2017.1319839](https://doi.org/10.1080/01621459.2017.1319839) · [arXiv:1510.04342v4](https://arxiv.org/html/1510.04342v4)
11. Friedberg, R.; Tibshirani, J.; Athey, S.; Wager, S. (2021). "Local Linear Forests." *JCGS* 30(2). [https://doi.org/10.1080/10618600.2020.1831930](https://doi.org/10.1080/10618600.2020.1831930) · [arXiv:1807.11408v3](https://arxiv.org/html/1807.11408v3)
12. Athey, S.; Wager, S. (2019). "Estimating Treatment Effects with Causal Forests: An Application." *Observational Studies* 5:36–51. [arXiv:1902.07409](https://arxiv.org/pdf/1902.07409)
13. Athey, S.; Wager, S. (2021). "Policy Learning with Observational Data." *Econometrica* 89(1):133–161. [https://doi.org/10.3982/ECTA15732](https://doi.org/10.3982/ECTA15732) · [arXiv:1702.02896](https://arxiv.org/abs/1702.02896)
14. Athey, S.; Imbens, G. (2016). "Recursive partitioning for heterogeneous causal effects." *PNAS* 113(27):7353–7360. [https://doi.org/10.1073/pnas.1510489113](https://doi.org/10.1073/pnas.1510489113)
15. Athey, S.; Imbens, G.; Wager, S. (2018). "Approximate residual balancing." *JRSS-B* 80(4):597–623. [https://doi.org/10.1111/rssb.12268](https://doi.org/10.1111/rssb.12268) · [arXiv:1604.07125](https://arxiv.org/abs/1604.07125)
16. Nie, X.; Wager, S. (2021). "Quasi-oracle estimation of heterogeneous treatment effects." *Biometrika* 108(2):299–319. [https://doi.org/10.1093/biomet/asaa076](https://doi.org/10.1093/biomet/asaa076)
17. Cui, Y.; Kosorok, M. R.; Sverdrup, E.; Wager, S.; Zhu, R. (2023). "Estimating Heterogeneous Treatment Effects with Right-Censored Data via Causal Survival Forests." *JRSS-B* 85(2):179–211. [https://doi.org/10.1093/jrsssb/qkac001](https://doi.org/10.1093/jrsssb/qkac001) · [arXiv:2001.09887](https://arxiv.org/abs/2001.09887)
18. Xu, Y.; Ignatiadis, N.; Sverdrup, E.; Fleming, S.; Wager, S.; Shah, N. (2023). "Treatment Heterogeneity with Survival Outcomes." In *Handbook of Matching and Weighting Adjustments for Causal Inference*, Ch. 21, Chapman & Hall/CRC. [arXiv:2207.07758](https://arxiv.org/abs/2207.07758)
19. Wager, S.; Hastie, T.; Efron, B. (2014). "Confidence intervals for random forests: the jackknife and the infinitesimal jackknife." *JMLR* 15:1625–1651. [https://www.jmlr.org/papers/v15/wager14a.html](https://www.jmlr.org/papers/v15/wager14a.html)
20. Wager, S.; Walther, G. (2015). "Adaptive concentration of regression trees, with application to random forests." [arXiv:1503.06388](https://arxiv.org/abs/1503.06388)
21. Biau, G. (2012). "Analysis of a random forests model." *JMLR* 13(38):1063–1095. [https://www.jmlr.org/papers/v13/biau12a.html](https://www.jmlr.org/papers/v13/biau12a.html)
22. Meinshausen, N. (2006). "Quantile Regression Forests." *JMLR* 7:983–999. [https://www.jmlr.org/papers/v7/meinshausen06a.html](https://www.jmlr.org/papers/v7/meinshausen06a.html)
23. Ishwaran, H.; Kogalur, U.; Blackstone, E.; Lauer, M. (2008). "Random Survival Forests." *Ann. Appl. Statist.* 2(3):860–873. [https://doi.org/10.1214/08-AOAS169](https://doi.org/10.1214/08-AOAS169)
24. Hothorn, T.; Bühlmann, P.; Yale, M.; van de Wiel, S. (2006). "Survival Ensembles." *Biostatistics* 7(3):355–373. [https://doi.org/10.1093/biostatistics/kxj011](https://doi.org/10.1093/biostatistics/kxj011)
25. Hothorn, T.; Hornik, K.; Zeileis, A. (2006). "Unbiased Recursive Partitioning." *JASA* 101(476):1517–1528. [https://doi.org/10.1198/106186006X133933](https://doi.org/10.1198/106186006X133933)
26. Huang, C.; Khowaja, K.; Härdle, W. (2021). "Uniform Inference for Generalized Random Forests." [https://doi.org/10.24818/ida-cl/2021.37](https://doi.org/10.24818/ida-cl/2021.37)
27. Khowaja, K.; Huang, C.; Härdle, W. (2022). "Uniform Confidence Bands for Generalized Random Forests." [https://doi.org/10.2139/ssn.4079006](https://doi.org/10.2139/ssn.4079006)
28. Hou, Y.; Fernández-Loría, C. (2026). "Honesty in Causal Forests: When It Helps and When It Hurts." *Information Systems Research*. [https://doi.org/10.1287/isre.2025.2343](https://doi.org/10.1287/isre.2025.2343) · [arXiv:2506.13107](https://arxiv.org/abs/2506.13107)
29. "The Honest Truth About Causal Trees: Accuracy Limits for Heterogeneous Treatment Effect Estimation." [arXiv:2509.11381](https://arxiv.org/abs/2509.11381)
30. Bladt, M.; Lemvig, R. F. (2026). "Consistency of Honest Decision Trees and Random Forests." [arXiv:2601.14991](https://arxiv.org/abs/2601.14991)
31. Yadlowsky, S.; Fleming, S.; Shah, N.; Brunskill, E.; Wager, S. (2025). "Evaluating treatment prioritization rules via rank-weighted average treatment effects." *JASA* 120(549). [arXiv:2111.07966](https://arxiv.org/abs/2111.07966)
32. Zhou, Z.; Athey, S.; Wager, S. (2023). "Offline Multi-Action Policy Learning." *Operations Research* 71(1). [https://doi.org/10.1287/opre.2022.2271](https://doi.org/10.1287/opre.2022.2271) · [arXiv:1810.04778](https://arxiv.org/abs/1810.04778)
33. Zhou, Z.; Athey, S.; Wager, S. (2022). "The Infinitesimal Jackknife and Combinations of Models." [arXiv:2209.00147](https://arxiv.org/abs/2209.00147)
34. Robinson, P. M. (1988). "Root-n-consistent semiparametric regression." *Econometrica*.
35. Chernozhukov, V. et al. (2018). "Double/debiased machine learning." *Econometrics Journal* 21(1). [https://doi.org/10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)
36. Quinlan, J. R. (1992). "Learning with continuous classes." AI'92, 343–348. [https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Quinlan-AI.pdf](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Quinlan-AI.pdf)
37. Wang, Y.; Witten, I. H. (1997). "Induction of model trees for predicting continuous classes." ECML-97 poster papers, 128–137. [PDF](https://researchcommons.waikato.ac.nz/server/api/core/bitstreams/f2b82fb0-3c87-462a-bc1d-764d314e14f7/content)
38. Frank, E.; Wang, Y.; Inglis, S.; Holmes, G.; Witten, I. H. (1998). "Using Model Trees for Classification." *Machine Learning* 32:63–76. [https://doi.org/10.1023/A:1007421302149](https://doi.org/10.1023/A:1007421302149)
39. Cubist GPL Edition C source, RuleQuest (2010), release 2.07. [https://github.com/topepo/Cubist](https://github.com/topepo/Cubist)
40. RuleQuest. "An Overview of Cubist." [https://www.rulequest.com/cubist-unix.html](https://www.rulequest.com/cubist-unix.html) · [https://www.rulequest.com/cubist-win.html](https://www.rulequest.com/cubist-win.html)
41. CRAN `Cubist` vignette. [https://cran.r-project.org/web/packages/Cubist/vignettes/cubist.html](https://cran.r-project.org/web/packages/Cubist/vignettes/cubist.html) · [reference manual](https://cran.r-project.org/web/packages/Cubist/refman/Cubist.html) · [package page](https://cran.r-project.org/package=Cubist)
42. Weka `M5P` sources. [RuleNode.java](https://raw.githubusercontent.com/Waikato/weka-3.8/master/weka/src/main/java/weka/classifiers/trees/m5/RuleNode.java) · [Rule.java](https://raw.githubusercontent.com/Waikato/weka-3.8/master/weka/src/main/java/weka/classifiers/trees/m5/Rule.java) · [M5Base.java](https://raw.githubusercontent.com/Waikato/weka-3.8/master/weka/src/main/java/weka/classifiers/trees/m5/M5Base.java)
43. Kotsiantis, S. B.; Tsekouras, G. E.; Pintelas, P. E. (2005). "Bagging Model Trees for Classification Problems." PCI 2005, LNCS 3746:328–337. [https://doi.org/10.1007/11573036_31](https://doi.org/10.1007/11573036_31)
44. Frank, E. et al. (2015). "Alternating Model Trees." ACM SAC 2015:871–878. [https://doi.org/10.1145/2695664.2695848](https://doi.org/10.1145/2695664.2695848)
45. Pfahringer, B. (2010). "Random model trees: an effective and scalable regression method." Waikato CS WP 03/2010. [https://hdl.handle.net/10289/4056](https://hdl.handle.net/10289/4056)
46. Jekabsons, G. et al. *M5PrimeLab* toolbox. [http://www.cs.rtu.lv/jekabsons/Files/M5PrimeLab.pdf](http://www.cs.rtu.lv/jekabsons/Files/M5PrimeLab.pdf)
47. Friedman, J. H. (2001). "Greedy function approximation: a gradient boosting machine." *Ann. Statist.* 29(5):1189–1232. [https://doi.org/10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
48. Friedman, J. H. (2002). "Stochastic gradient boosting." *Ann. Statist.* 30(1):115–123. [https://doi.org/10.1214/aos/1016345136](https://doi.org/10.1214/aos/1016345136)
49. Bühlmann, P.; Yu, B. (2003). "Boosting with the L2 loss: regression and classification." *JASA* 98(462):324–339. [https://doi.org/10.1198/016214503000125](https://doi.org/10.1198/016214503000125)
50. Quinlan, J. R. (1996). "Bagging, boosting, and C4.5." AAAI-96, 725–730.
51. Chen, T.; Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." [arXiv:1603.02754](https://arxiv.org/pdf/1603.02754)
52. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017. [https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
53. Prokhorenkova, L. et al. (2018). "CatBoost: unbiased boosting with categorical features." [arXiv:1706.09516](https://arxiv.org/pdf/1706.09516)
54. Shi, Y.; Li, J.; Li, Z. (2019). "Gradient Boosting With Piece-Wise Linear Regression Trees." IJCAI 2019. [arXiv:1802.05640](https://arxiv.org/pdf/1802.05640)
55. de Vito, L. (2017). "LinXGBoost: Extension of XGBoost to Generalized Local Linear Models." [arXiv:1710.03634](https://arxiv.org/abs/1710.03634)
56. Raymond, M.; Violi, A.; Scott, C. (2024). "Joint Optimization of Piecewise Linear Ensembles." IWMLSP 2024. [arXiv:2405.00303](https://arxiv.org/pdf/2405.00303)
57. Nori, H. et al. (2019). "InterpretML: A Unified Framework for Machine Learning Interpretability." [arXiv:1909.09223](https://arxiv.org/pdf/1909.09223) · [docs](https://interpret.ml/docs/)
58. Duan, T.; Avati, A.; Ding, D. Y.; Thai, K. K.; Basu, S.; Ng, A. "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." [arXiv:1910.03225](https://arxiv.org/pdf/1910.03225)
59. Lundberg, S. M.; Erion, G.; Lee, S.-I. (2018/2020). "Consistent Individualized Feature Attribution for Tree Ensembles." [arXiv:1802.03888](https://arxiv.org/abs/1802.03888)
60. Lundberg, S. M.; Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017. [https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
61. Koklev, P. "What's the Price of Monotonicity? A Multi-Dataset Benchmark of Monotone-Constrained Gradient Boosting." [arXiv:2512.17945](https://arxiv.org/pdf/2512.17945)
62. Grinsztajn, L.; Oyallon, E.; Varoquaux, G. (2022). "Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?" [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
63. McElfresh, D. et al. (2023). "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" [arXiv:2305.02997](https://arxiv.org/abs/2305.02997)
64. Erickson, N. et al. (2025). "TabArena: A Living Benchmark for Machine Learning on Tabular Data." [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf) · [arXiv:2506.16791](https://arxiv.org/abs/2506.16791)
65. Strobl, C. et al. (2007). "Bias in Random Forest Variable Importance Measures." [https://doi.org/10.1186/1471-2105-8-25](https://doi.org/10.1186/1471-2105-8-25)
66. Strobl, C. et al. (2008). "Conditional Variable Importance for Random Forests." [https://doi.org/10.1186/1471-2105-9-307](https://doi.org/10.1186/1471-2105-9-307)
67. Hooker, G.; Mentch, L.; Zhou, S. (2021). "Unrestricted Permutation Forces Extrapolation." [https://doi.org/10.1007/s11222-021-10057-z](https://doi.org/10.1007/s11222-021-10057-z)
68. Janitza, S.; Hornung, R. (2018). "On the Overestimation of Random Forest's Out-of-Bag Error." [https://doi.org/10.1371/journal.pone.0201904](https://doi.org/10.1371/journal.pone.0201904)
69. Mitchell, M. W. (2011). "Bias of the Random Forest Out-of-Bag Error for Certain Input Parameters." [https://doi.org/10.4236/ojs.2011.13024](https://doi.org/10.4236/ojs.2011.13024)
70. Lakshminarayanan, B.; Roy, D. M.; Teh, Y. W. (2014). "Mondrian Forests: Efficient Online Random Forests." [arXiv:1406.2673](https://arxiv.org/abs/1406.2673)
71. Lakshminarayanan, B.; Roy, D. M.; Teh, Y. W. (2016). "Mondrian Forests for Large-Scale Regression when Uncertainty Matters." [PMLR v51](https://proceedings.mlr.press/v51/lakshminarayanan16a.html)
72. Oza, N. C.; Russell, S. J. (2001). "Online Bagging and Boosting." [PMLR R3](https://proceedings.mlr.press/r3/oza01a.html) · [KDD DOI](https://doi.org/10.1145/502512.502565)
73. Liu, Y.; Loh, W.-Y. (2019). "On the Optimal Size of Candidate Feature Set in Random Forest." [https://doi.org/10.3390/app9050898](https://doi.org/10.3390/app9050898)
74. Popov, S. et al. (2019). "Neural Oblivious Decision Ensembles." [arXiv:1909.06312](https://arxiv.org/abs/1909.06312)
75. Raymaekers, J. et al. (2025). "A Powerful Random Forest Featuring Linear Extensions." [arXiv:2502.10185](https://arxiv.org/abs/2502.10185)
76. Ao, Y. et al. (2019). "The Linear Random Forest Algorithm." [https://doi.org/10.1016/j.petrol.2018.11.067](https://doi.org/10.1016/j.petrol.2018.11.067)
77. Genuer, R.; Poggi, J.-M.; Tuleau-Malot, C. (2015). "VSURF." *R Journal*. [https://journal.r-project.org/articles/RJ-2015-018/](https://journal.r-project.org/articles/RJ-2015-018/)
78. Iranzad, R.; Liu, X. (2025). "A Review of Random Forest-Based Feature Selection Methods." [https://doi.org/10.1007/s41060-024-00509-w](https://doi.org/10.1007/s41060-024-00509-w)
79. Filom, K. et al. (2023/2024). "On Marginal Feature Attributions of Tree-Based Models." [arXiv:2302.08434](https://arxiv.org/abs/2302.08434)
80. Loecher, M. (2023). "Debiasing SHAP Scores in Random Forests." [https://doi.org/10.1007/s10182-023-00479-7](https://doi.org/10.1007/s10182-023-00479-7)
81. Greenwell, B. M. (2017). "pdp: An R Package for Constructing Partial Dependence Plots." [https://doi.org/10.32614/RJ-2017-016](https://doi.org/10.32614/RJ-2017-016)
82. Greenwell, B. M.; Boehmke, B. C.; McCarthy, A. J. (2018). "A Simple and Effective Model-Based Variable Importance Measure." [arXiv:1805.04755](https://arxiv.org/abs/1805.04755)
83. Rhodes, J. S.; Cutler, A.; Moon, K. R. (2023). "Geometry- and Accuracy-Preserving Random Forest Proximities." [arXiv:2201.12682](https://arxiv.org/abs/2201.12682)
84. TableShift, NeurIPS 2023. [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a76a757ed479a1e6a5f8134bea492f83-Paper-Datasets_and_Benchmarks.pdf) · [arXiv:2312.07577](https://arxiv.org/abs/2312.07577)
85. Yang, K. et al. (2023). "On the Need for a Language Describing Distribution Shifts." [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a134eaebd55b7406ff29cd75d5f1a622-Paper-Datasets_and_Benchmarks.pdf)
86. Teney, A. et al. (2023/2026). "Rethinking Distribution Shifts." [arXiv:2307.05284](https://arxiv.org/abs/2307.05284)
87. Scornet, E.; Hooker, G. (2026). "Theory of Random Forests." [https://doi.org/10.1146/annurev-statistics-112723-034707](https://doi.org/10.1146/annurev-statistics-112723-034707)
88. Twala, B.; Jones, J.; Hand, D. J. (2008). "On the Missingness Mechanism in Decision Trees." [DOI 10.1007/s10994-007-9073-9](https://doi.org/10.1007/s10994-007-9073-9)
89. Ghosal, I.; Hooker, G. (2020). "Boosting random forests to reduce bias: one-step boosted forest and its variance estimate." *JCGS*.
90. Hou et al. honesty diagnostics. See entry 28.
91. Software: grf algorithm reference [https://github.com/grf-labs/grf/blob/master/REFERENCE.md](https://github.com/grf-labs/grf/blob/master/REFERENCE.md) · grf docs [https://grf-labs.github.io/grf/](https://grf-labs.github.io/grf/) · [causal_forest](https://grf-labs.github.io/grf/reference/causal_forest.html) · [average_treatment_effect](https://grf-labs.github.io/grf/reference/average_treatment_effect.html) · [best_linear_projection](https://grf-labs.github.io/grf/reference/best_linear_projection.html) · [get_forest_weights](https://grf-labs.github.io/grf/reference/get_forest_weights.html) · [instrumental_forest](https://grf-labs.github.io/grf/reference/instrumental_forest.html) · [ll_regression_forest](https://grf-labs.github.io/grf/reference/ll_regression_forest.html) · [multi_arm_causal_forest](https://grf-labs.github.io/grf/reference/multi_arm_causal_forest.html) · [survival_forest](https://grf-labs.github.io/grf/reference/survival_forest.html)
92. Software: EconML [https://github.com/py-why/EconML](https://github.com/py-why/EconML) · [docs](https://www.pywhy.org/EconML/) · CausalML [https://github.com/uber/causalml](https://github.com/uber/causalml) · [docs](https://causalml.readthedocs.io/)
93. Software: scikit-learn ensemble docs, [https://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html) · [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) · [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) · [ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) · [RandomTreesEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html) · [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) · [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) · [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) · [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) · [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) · [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) · [developer guide](https://scikit-learn.org/stable/developers/develop.html)
94. Software: XGBoost docs [https://xgboost.readthedocs.io/en/stable/parameter.html](https://xgboost.readthedocs.io/en/stable/parameter.html) · [monotone tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html) · LightGBM docs [https://lightgbm.readthedocs.io/en/latest/Parameters.html](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
95. Software: tidymodels parsnip [https://parsnip.tidymodels.org/](https://parsnip.tidymodels.org/) · [set_engine](https://parsnip.tidymodels.org/reference/set_engine.html) · Julia MLJ [model search](https://juliaai.github.io/MLJ.jl/stable/model_search/) · [cheatsheet](https://juliaai.github.io/MLJ.jl/stable/mlj_cheatsheet/) · [MLJModelInterface quick start](https://juliaai.github.io/MLJModelInterface.jl/dev/quick_start_guide)
96. Software: CRAN [ranger](https://cran.r-project.org/package=ranger) · [randomForest](https://cran.r-project.org/package=randomForest) · [partykit](https://cran.r-project.org/package=partykit) · [rpart](https://cran.r-project.org/package=rpart) · [gbm](https://cran.r-project.org/package=gbm) · [xgboost](https://cran.r-project.org/package=xgboost) · [lightgbm](https://cran.r-project.org/package=lightgbm) · [grf](https://cran.r-project.org/package=grf) · [Cubist](https://cran.r-project.org/package=Cubist)
97. Hollmann, D. et al. (2025). "Accurate predictions on small data with a tabular foundation model." *Nature*. [https://doi.org/10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6) · TabPFN v2 [arXiv:2412.17174](https://arxiv.org/abs/2412.17174)
98. Qu, J. et al. "TabICL." [arXiv:2502.05564](https://arxiv.org/abs/2502.05564)
99. "TabDPT." [arXiv:2601.09778](https://arxiv.org/abs/2601.09778)
100. "ModernNCA." [arXiv:2504.08973](https://arxiv.org/abs/2504.08973)
