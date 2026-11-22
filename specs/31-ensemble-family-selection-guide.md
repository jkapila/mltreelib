# Spec 31: Choosing an Ensemble Family — Forest, Boost, or Meta

Status: proposed

Cross-cutting specification. Governs Spec 28 (`ModelForest`), Spec 29 (`ModelBoost`), Spec 30 (`ModelMeta`), and Spec 32 (`ModelBayes`).

## Problem Statement

`mltreelib` will ship four ensemble families:

- `ModelForest` — bagging. Members trained on resampled data, combined by averaging or voting.
- `ModelBoost` — boosting. Stages trained sequentially on loss gradients, combined by a shrunk sum.
- `ModelMeta` — meta-estimators. Base learners trained independently, combined by a fixed or learned rule over their predictions.
- `ModelBayes` — Bayesian. A posterior over the function, not a point prediction.

A user with a tabular dataset and a prediction problem has to pick one. Today, the honest answer is "it depends", and the honest explanation is four paragraphs long. That is not an acceptable answer for a library that prides itself on explainability.

The problem is not that the methods are hard. It is that the **decision** has never been written down anywhere in this repository, and every user has to reconstruct it from first principles while the library gives them no vocabulary for it.

Four specific failures follow from that gap.

### Failure 1: Users default to boosting because boosting is famous

Gradient boosting wins most tabular benchmarks. TabArena's conventional-tuning results put CatBoost first among individual methods. That creates a default that ignores the actual trade-off.

But the benchmark evidence has a specific scope: medium-to-large datasets, IID, irregular feature distributions, and a tuning budget of 200 configurations per model. Outside that regime the ordering does not hold, and for a library whose selling points are first-class missing-value handling, categorical partition search, monotone constraints, rule extraction, and interpretable leaf models, those constraints are frequently the binding requirement rather than the accuracy metric.

### Failure 2: Users reach for stacking when they mean bagging

Stacking is the most discussed meta-estimator and the most misused. It requires out-of-fold predictions, it needs enough data to fit a second-level model, it fails silently when members are correlated, and it adds a whole extra fitting stage to every retrain.

A user who wants "several trees instead of one tree" has reached for the wrong tool. They want `ModelForest`.

### Failure 3: Nobody knows that the bagging family here is not just random forest

`mltreelib`'s forest family includes a Cubist-style committee whose members carry linear model leaves, and a linear-leaf random forest. These are not "random forest with different hyperparameters". They are a different bias-variance profile, and on smooth scientific and engineering targets they behave differently from constant-leaf forests in a way the library can state precisely.

The decision framework must therefore not be "forest versus boost". It must be "which member model, which training procedure, which combination rule".

### Failure 4: Users reach for a point prediction when they wanted uncertainty

The library's three original families all produce point predictions with attached diagnostics. None of them produces a distribution. A user who wants to know how confident the model is, which predictors the data actually support, or where the function is uncertain has no family that answers that, and the nearest substitute, a conformal band, answers a different question with different assumptions.

The Bayesian family exists for that question, and the framework has to make it reachable without implying that a posterior band and a conformal band are interchangeable.

### What this specification does

It writes the decision down, once, in a form a user can act on, and makes it part of the library's documented surface. The framework is:

1. **Hard constraints first.** Some requirements eliminate a family outright. Interpretability, a hard prediction budget, a mandated algorithm, a distributional output, a missing-data guarantee.
2. **Then the data regime.** Size, dimension, feature type mix, noise level, and whether the target is smooth.
3. **Then the cost model.** Build time, inference latency, retrain frequency, and the available tuning budget.
4. **Then the default.** If nothing eliminates a family, a documented default applies.

The framework produces a decision, not a survey. Every branch ends in a specific recommendation.

## Solution

A decision procedure with named stages, a decision table per stage, and a documented default path.

### 1. The four families in one table

| | `ModelForest` | `ModelBoost` | `ModelMeta` | `ModelBayes` |
|---|---|---|---|---|
| What it varies | Training data per member | Target per stage | Combination rule | Model structure and leaf values jointly |
| Reduces | Variance | Bias | Neither directly | Uncertainty, by producing a posterior |
| Output | Point prediction | Point prediction | Point prediction | **Posterior over functions** |
| Member count | 50 to 5000 | 100 to 5000 | 2 to 10 | 1 to 200 trees per draw, times draws |
| Parallelism | Across members | Within a stage | Across members and folds | Across chains only; a chain is sequential |
| Tuning burden | Low | High | Medium to high | High |
| Build cost | Linear in members | Linear in stages | Members times folds, plus the fold phase | Draws times trees times proposals per tree |
| Inference cost | Linear in members | Linear in stages | Linear in members, plus the meta-model | Linear in trees, times draws for intervals |
| Needs held-out data | No | For early stopping | Yes, for stacking | No, the posterior substitutes |
| Retrain cost | Full | Full | Full, plus a fold phase | Full, plus the chain |
| Out-of-bag diagnostic | Yes, with known optimism | Only with row subsample | No, out-of-fold instead | No; credible intervals instead |
| Overfitting mode | Too few members | Too many stages, too high a learning rate | Meta-model too flexible, too few rows | Prior too weak, or chain not mixed |
| Uncertainty guarantee | None | None | None | **Exact under the model**, up to Monte Carlo error |

### 2. The local Elo board is the primary evidence

The framework below is grounded in this repository's own 56-dataset Elo arena, not only in cited papers. The local board is the strongest evidence available, because it measures the library's actual model tree against the actual competitors on the actual data.

**Overall leaderboard, 56 datasets, 47 entrants.** Test RMSE on 18 regression tasks, test accuracy on 38 classification tasks.

| Rank | Model | Overall | Regression Elo | Classification Elo | Family | Mean fit time |
|---:|---|---:|---:|---:|---|---:|
| 1 | grf_llf | 1600.2 | 1600.2 | · | forest, local linear | 20.97 s |
| 2 | CatBoost(200) | 1457.8 | 1498.5 | 1372.4 | boosting | 1.01 s |
| 3 | perpetual (Rust) | 1400.1 | 1505.9 | 1314.7 | boosting, Rust | 7.19 s |
| 4 | EBM | 1353.6 | 1383.4 | 1268.2 | additive | 66.59 s |
| 5 | sklearn RF | 1349.7 | 1686.2 | 1264.3 | forest | 18.56 s |
| 6 | ChimeraBoost | 1340.0 | 1763.7 | 1254.6 | boosting | 13.71 s |
| 7 | bart | 1313.9 | 1243.3 | 1228.6 | Bayesian | 25.49 s |
| 8 | ranger | 1298.2 | 1700.6 | 1212.8 | forest | 17.78 s |
| **9** | **ModelTree(all)** | **1290.1** | **715.9** | **1204.7** | **single tree** | **4.01 s** |
| 10 | sklearn GBM | 1274.0 | 1125.0 | 1188.6 | boosting | 6.79 s |
| 11 | c50_rules | 1269.5 | · | 1184.1 | rules | — |
| 12 | HistGradientBoosting | 1258.6 | 1425.8 | 1173.2 | boosting | 0.94 s |
| 13 | c50_tree | 1242.6 | · | 1157.2 | tree | — |
| 14 | LightGBM(200) | 1240.9 | 1542.4 | 1155.5 | boosting | 0.34 s |
| 15 | CTBoost | 1208.7 | 1292.9 | 1123.3 | boosting | 0.75 s |
| 16 | ExtraTrees | 1169.5 | 1640.4 | 1084.1 | forest | 12.64 s |
| 17 | forust (Rust) | 1141.8 | 1376.2 | 1056.4 | forest, Rust | 1.19 s |
| 18 | MARS (mars-earth) | 1105.0 | 882.8 | 1019.7 | additive | — |
| 19 | modeltrees lmtree | 1088.9 | 1088.9 | · | linear tree | 15.29 s |
| 20 | lmtree | 1088.8 | 1088.8 | · | linear tree | 19.36 s |
| 21 | ctree | 1039.7 | 669.1 | 954.3 | tree | — |
| 22 | ModelTree(opt) | 1031.4 | 850.4 | 946.0 | single tree | — |

**Four findings the framework must encode.**

**Finding 1: `ModelTree(all)` is strong on classification and weak on regression.** Its classification Elo of 1204.7 beats sklearn GBM (1188.6), HistGradientBoosting (1173.2), LightGBM(200) (1155.5), and ExtraTrees (1084.1). Its regression Elo of 715.9 is near the bottom, behind ctree (669.1 is worse, but everything else is far ahead): ChimeraBoost 1763.7, sklearn RF 1686.2, ranger 1700.6, ExtraTrees 1640.4.

**A single tree has high variance on regression and comparatively low variance on classification.** That is the classic bias-variance asymmetry: squared-error loss punishes variance directly, while accuracy on a well-separated problem tolerates a wobbly boundary. So the library's single tree is not "bad at regression". It is **under-ensembled** on regression.

**Finding 2: this is the strongest argument in the whole project for building `ModelForest` first.** The gap is not a modelling deficiency. It is a missing variance-reduction step. Bagging over the existing `ModelTree` is precisely the fix, and the local board says the fix is worth roughly 900 Elo points on regression if the forest performs like `ranger` or `sklearn RF` rather than like a single tree.

The prediction is testable: a `ModelForest(kind="rf")` over today's `ModelTree` should land near ranger's 1700.6 regression Elo, not near ModelTree's 715.9.

**Finding 3: a Rust boosting library already ranks third.** `perpetual (Rust)` scores 1400.1 overall, 1505.9 regression, 1314.7 classification. It beats CatBoost(200) on regression Elo (1505.9 against 1498.5), beats LightGBM(200), sklearn GBM, and sklearn RF on overall Elo, and beats every forest except `grf_llf` on regression.

**This kills the "boosting is a multi-year project" claim.** A Rust GBDT is competitive today, and it is competitive without the library's leaf families, monotone constraints, categorical partition search, or missing-value routing. It competes on speed and simplicity. That raises the bar for `ModelBoost` rather than lowering it, and it means the library's boosting entry cannot be "a plain GBDT"; it has to carry something the plain Rust booster does not.

**Finding 4: the speed gap is the real engineering problem.** Fit times on the regression board:

| Model | Mean fit time | Regression Elo |
|---|---:|---:|
| LightGBM(200) | 0.34 s | 1542.4 |
| CTBoost | 0.75 s | 1292.9 |
| HistGradientBoosting | 0.94 s | 1425.8 |
| CatBoost(200) | 1.01 s | 1498.5 |
| forust (Rust) | 1.19 s | 1376.2 |
| perpetual (Rust) | 7.19 s | 1505.9 |
| sklearn GBM | 6.79 s | 1125.0 |
| ModelTree(winnow) | 4.01 s | 997.2 (different config) |
| ChimeraBoost | 13.71 s | 1763.7 |
| ExtraTrees | 12.64 s | 1640.4 |
| ranger | 17.78 s | 1700.6 |
| sklearn RF | 18.56 s | 1686.2 |
| grf_llf | 20.97 s | 1600.2 |
| bart | 25.49 s | 1243.3 |
| EBM | 66.59 s | 1383.4 |

The fast boosters are the ones with histogram binning. LightGBM fits in 0.34 s because it bins features once and scans binned histograms, not raw values. ranger and sklearn RF take 18 s because they rescan raw values per node.

`mltreelib` already has a histogram path: `SplitEval::Histogram` with at most 256 bins per numeric feature. It is currently restricted to classification, because `uses_class_histogram` gates it to gini, entropy, gain ratio, and twoing. **Extending histogram evaluation to the regression gains is the single highest-leverage performance change for the whole ensemble layer**, and it is a smaller project than any boosting algorithm.

That change lives in **Spec 27, Decision 15 and Phase 1**, because the criteria that need the binned path are the regression criteria and the regression families are that specification's subject. Specs 28, 29, and 30 all consume it.

**What the board does not say.** These are 56 IID datasets with a fixed protocol. The board has no distribution-shift tasks, no latency constraint, and no cost model beyond fit time. It says nothing about missing-value handling, monotone constraints, categorical cardinality, or interpretability, all of which the library treats as first-class. A model that ranks 9th overall while owning native missing values, monotone constraints, and rule extraction is in a different competitive position than the Elo number alone suggests.

### 3. Stage 1: Hard constraints

These eliminate families. Ask in order and stop at the first hit.

| Constraint | Eliminates | Why |
|---|---|---|
| A mandated algorithm, a compliance requirement, or a contractual restriction | Everything except the mandated family | Not a modelling decision |
| The output must be a probability distribution, a survival curve, a quantile set, or a treatment effect | `ModelBoost` in its first wave | Boosting with constant leaves produces point predictions. The distributional machinery lives in the leaf model, and the boosting leaf update is a constant-value Newton step |
| Every prediction must be traceable to readable rules | `ModelBoost` and `ModelBayes` | A boosted ensemble has no per-member rule set that survives aggregation, and a posterior has no rules at all. A forest member does |
| The output must be a **credible interval with a stated coverage guarantee**, not a bootstrap or conformal band | `ModelForest` and `ModelBoost` | Those two families produce no interval of their own. `ModelMeta` produces conformal intervals, which are distribution-free and finite-sample. `ModelBayes` produces posterior intervals, which are exact under the model |
| Monotone constraints must hold exactly | None outright | All four can enforce them. `ModelForest` enforces them per member and the aggregate inherits it exactly. `ModelBoost` must enforce them at every stage and can lose them under shrinkage unless the leaf projection is applied after every update. `ModelBayes` needs the monotone-constrained posterior variant, and `ModelMeta` inherits its members' constraints |
| A hard inference latency budget below roughly one millisecond per row | `ModelForest` above a few hundred members, `ModelMeta`, and `ModelBayes` | Boosting scales with stages, forests with members, a stack with both, and a posterior with trees times draws |
| A hard build-time budget | `ModelMeta` and `ModelBayes` | The fold phase multiplies base fits by the fold count, and the posterior sampler multiplies tree fits by the draw count |
| Only one model type may be used for regulatory reasons | `ModelMeta` | A stack's members may be different families |
| Point predictions only, with no uncertainty of any kind wanted | `ModelBayes` | A posterior is strictly more expensive than a point fit and produces a point prediction only as one of its summaries |

**The rule for the second row.** A distributional output is the strongest single argument for `ModelForest` in this library, and it is an argument about the library, not about the literature. `ModelTree` already carries survival leaves, quantile leaves, and uplift leaves. A forest of those inherits them for free. A booster would need a distributional leaf update, which is a different research problem.

**The rule for the credible-interval row.** This is the sharpest boundary between the families, because the four families offer three genuinely different interval provenances:

| Family | Interval provenance | Guarantee |
|---|---|---|
| `ModelForest` | Split-conformal from out-of-bag nonconformity scores | Distribution-free, finite-sample |
| `ModelBoost` | None | No interval |
| `ModelMeta` | Split-conformal from out-of-fold residuals | Distribution-free, finite-sample |
| `ModelBayes` | Posterior quantiles of the draw vector | Exact under the model, up to Monte Carlo error |

**These are not interchangeable.** A conformal band is valid without any model assumption and is therefore wider when the model is wrong. A posterior band is narrower and sharper when the model is right, and wrong in a specific, model-dependent way when it is wrong. The report labels the provenance of every interval.

### 4. Stage 2: Data regime

| Property | Favours `ModelForest` | Favours `ModelBoost` | Favours `ModelBayes` |
|---|---|---|---|
| Rows below roughly 10,000 | Yes, strongly | No. Boosting needs enough rows per stage to fit a tree and to place a validation split | No. The chain needs rows to mix |
| Rows from 10,000 to 100,000 | Either | `ModelBoost` usually wins on accuracy | Possible, but the sampler cost bites |
| Rows above 100,000 | Either | `ModelBoost` usually wins, and the gap grows | Possible. `bart` holds 1243.3 regression Elo on the local board |
| Features below roughly 20 | Either | `ModelBoost` usually wins | The BART priors handle many features well |
| Features above roughly 100 | `ModelForest` is competitive; `ModelBoost` needs feature subsampling | `ModelBoost` with column subsampling | The leaf-shrinkage prior helps, and posterior inclusion probabilities are informative |
| Categorical features with many levels | `ModelForest` with the partition policy | `ModelBoost` only with native ordered categoricals, which is a large project | Poor. The R BART packages use dummy expansion; the library's partition search is better but is not part of the sampler |
| Missing values in the split column | `ModelForest` handles natively, per member | `ModelBoost` needs a default-direction rule at every stage | Not natively. The sampler needs a missing-value-aware marginal likelihood |
| Heavy label noise | `ModelForest` | Neither. Both overfit; use a robust leaf and early stopping | The error-variance prior absorbs noise explicitly, which is its strength |
| A smooth target with local linearity | `ModelForest` with the committee or linear-leaf kinds | `ModelBoost` with constant leaves loses to a linear-leaf booster, which is deferred | BART's priors are well matched to smooth targets |
| An irregular target with interactions | Either | `ModelBoost` usually wins | The sum-of-trees structure captures interactions |
| Very few rows and many features | `ModelForest` with strong feature subsampling | No | No |
| The real question is uncertainty, not accuracy | No | No | **Yes. This is the family's reason to exist** |

**The caveat that must appear.** These are benchmark-derived tendencies with a specific scope, not laws. Grinsztajn et al. (2022) used 45 curated IID datasets, mostly around 10,000 samples, with $d/n < 1/10$ and fewer than 500 predictors, and removed incomplete rows. McElfresh et al. (2023) used 176 datasets and 538,650 trained models. TabArena (2025) used 51 curated IID datasets with 200 tuned configurations per model. None of these establish an ordering for a specific dataset with a specific constraint set.

### 5. Stage 3: Cost model

| Question | `ModelForest` | `ModelBoost` | `ModelMeta` | `ModelBayes` |
|---|---|---|---|---|
| Build time | Linear in member count | Linear in stage count | Members times fold count | Draws times trees times proposals per tree |
| Retrain frequency | Full retrain | Full retrain | Full retrain plus a fold phase | Full retrain plus the chain |
| Inference latency | Members times one tree | Stages times one tree | Members times one tree plus the meta-model | Trees times one tree, times draws for intervals |
| Tuning burden | Low. A handful of knobs | High. The learning rate and stage count interact, and early stopping needs a validation split | Medium to high. The final estimator and the fold count interact | High. Prior parameters, draw count, burn-in, thinning, and mixing all matter |
| Tuning budget available | Anything | A real budget, ideally with a validation split | A real budget, and enough rows for the meta-model | A real budget, and enough rows for the chain to mix |
| Parallelism | Across members, embarrassingly | Within a stage | Across members and folds | Across chains only. A chain is sequential |

**The decision rule.** If the tuning budget is one or two manual attempts, use `ModelForest`. A forest with default settings is competitive on most datasets and needs no search. A booster with default settings is often mediocre, and a booster that is not tuned is worse than a forest.

**The decision rule for latency.** At roughly 100 members and depth 8, a forest costs about 100 tree traversals per row. A booster with 500 shallow stages costs about 500 traversals. A forest with 500 members and a booster with 500 stages are comparable, and the forest trains far faster because its members fit in parallel.

**The decision rule for the Bayesian family.** The posterior sampler's cost is the reason it is not the default. On the local board, `bart` takes 25.49 s mean fit time against `ranger`'s 17.78 s and LightGBM's 0.34 s, for a model that is competitive but not dominant. A posterior is worth that cost when the *uncertainty* is the deliverable, and rarely otherwise.

### 6. Stage 4: The default path

When no hard constraint applies and the data regime is unremarkable, the documented default is:

**`ModelForest(kind="rf", mode=...)`, with `n_estimators=200` and `max_features="sqrt"`.**

**The reason is the failure mode, not the peak accuracy.** A forest has no catastrophic failure mode. It cannot overfit catastrophically with default settings, it trains in parallel, it needs no validation split, and its diagnostics are honest once the out-of-bag caveat is stated. A booster that is mistuned can be much worse than a forest, a stack that is under-resourced can silently learn nothing, and a posterior that has not mixed produces intervals that are confidently wrong.

**The local board backs this default twice over.** First, `ModelTree(all)` already scores 1204.7 classification Elo, ahead of sklearn GBM, HistGradientBoosting, and LightGBM. The tree is good. Second, its regression Elo of 715.9 is the single largest fixable gap in the library, and bagging is the fix. A `ModelForest` over today's tree should land near ranger's 1700.6 regression Elo rather than ModelTree's 715.9. That is the default path with a measured upside rather than a hoped-for one.

**When to override the default.**

- **Reach for `ModelBoost`** when the dataset is medium-to-large, the target is irregular, the tuning budget exists, and accuracy is the primary objective.
- **Reach for `ModelMeta`** when there are already two or more genuinely different models with uncorrelated errors and enough rows to fit a second-level model.
- **Reach for `ModelBayes`** when the deliverable is uncertainty rather than accuracy: a credible band on a prediction, a posterior over which predictors matter, or a distribution over the function for a downstream decision. Note that `ModelBayes` also fits when the accuracy question is secondary, since the posterior mean is usually competitive. On the local board `bart` scores 1313.9 overall, ahead of ranger, `ModelTree(all)`, and every boosting implementation except CatBoost(200).

**The bar `ModelBoost` must clear.** `perpetual (Rust)` scores 1400.1 overall with 7.19 s mean fit time and no leaf families, no monotone constraints, no categorical partition search, and no native missing-value routing. A `ModelBoost` that is merely a plain GBDT would be a worse version of an existing Rust library. It has to earn its place with something the plain Rust booster does not have. The candidates are the library's distributional leaves, its monotone constraints, its categorical handling, and its missing-value routing. Spec 29 has to name which one leads, or the family should not be built.

**The bar `ModelBayes` must clear.** `bart` already scores 1313.9 overall on the local board. The library's version has to add the things the R packages lack: native missing-value routing, monotone constraints that hold on the *summed* posterior, and the library's categorical partition search inside the sampler. A posterior without those is `bart` reimplemented, which is not worth a family.

### 7. The choice within the bagging family

`ModelForest` has five kinds. Choosing among them is a second decision.

| Kind | Choose it when |
|---|---|
| `bagging` | You want the simplest variance reduction and nothing else. Every member sees all features |
| `rf` | The default. You want decorrelation from feature subsampling |
| `extra_trees` | Training time matters more than last-decimal accuracy, or the target is very high-dimensional |
| `committee` | The target is smooth and locally linear, and you want better extrapolation than a constant-leaf forest. Members carry linear leaves and are trained sequentially on relabelled targets |
| `linear_forest` | The target is smooth and locally linear and you want parallel training rather than the committee's sequential dependence |
| `rotation_forest` | You want per-member PCA rotation for decorrelation that feature subsetting does not give |
| `double_bagging` | You want an out-of-bag auxiliary model whose predictions become extra tree features |
| `iterated_bagging` | You want out-of-bag pseudo-targets to update the responses between stages |

Options on the family: `bootstrap=False` for subagging, `bootstrap="bayesian"` for the Bayesian bootstrap, `max_features < 1.0` for feature-subset bagging, and a member feature mask for random subspace partitioning. Spec 28 §1.1 gives the full inventory.

**The committee versus the linear-leaf forest.** Both use linear leaves. The difference is training: the committee is sequential with relabelled targets and an unweighted member average, and the linear-leaf forest is parallel with bootstrap resampling and a uniform average. The committee fits a smooth target better in some regimes. The linear-leaf forest trains faster and parallelises. Neither dominates, and the honest statement is that this needs a benchmark inside the library.

### 8. The choice within the meta family

| Kind | Choose it when |
|---|---|
| `voting` | You have two or more classification models with partly independent errors and want the cheapest possible combination |
| `averaging` | The regression equivalent |
| `stacking` | You want the combination to depend on the input region, and you have enough rows and uncorrelated members |

**The precondition for stacking.** Before recommending a stack, check that the members' out-of-fold errors are not strongly correlated. Two models trained on the same data with the same features will have correlated errors, and a stack over them will not beat the better member. The package computes this correlation from the out-of-fold matrix and warns when it is high.

### 9. The choice within the Bayesian family

| Kind | Choose it when |
|---|---|
| `prior_cart` | You want BART's regularisation story without the sampler. Deterministic, fast, and the priors are the documented ones, but there is no posterior |
| `posterior_sum` | You need credible intervals, a posterior over predictors, or a distribution over the function. Slower, and the intervals are only as good as the chain's mixing |

**The distinction is not cosmetic.** `prior_cart` gives you a tree grown under BART's size and shrinkage priors. `posterior_sum` gives you the BART posterior, with intervals that are exact under the model up to Monte Carlo error. The report for `prior_cart` states plainly that it is not the posterior.

**Choose `prior_cart` when** you want the regularisation and not the uncertainty, or when the sampler cost is unaffordable. **Choose `posterior_sum` when** the uncertainty is the deliverable.

### 10. The documentation obligation

The framework above must appear in three places, or it does not exist.

1. **A decision section in each family spec**, pointing here.
2. **A `mltreelib` documentation page** under the ensemble section, with the stage 1 to stage 4 procedure and the decision tables.
3. **An agent skill entry** that answers "which ensemble should I use for this dataset" with the framework, so an agent chooses the family the same way the documentation says a human should.

The package's own agent skill already exists under `.agents/skills/`. The ensemble entry extends it.

## User Stories

### Decision Framework

1. As a user, I want a documented procedure for choosing between forest, boost, and meta, so that I do not have to reconstruct it from first principles.
2. As a user, I want hard constraints checked first, so that a requirement like readable rules or a distributional output eliminates the wrong families immediately.
3. As a user, I want the data regime considered second, so that I know what the benchmark evidence actually applies to.
4. As a user, I want the cost model considered third, so that I do not pick a method I cannot afford to run.
5. As a user, I want a documented default when nothing else decides, so that I have a starting point instead of a survey.
6. As a user, I want the default to be the method with the least catastrophic failure mode, so that a reasonable first attempt is a reasonable model.
7. As a user, I want the benchmark caveats stated with the tables, so that I do not read a tendency as a law.
8. As a user, I want to know that Grinsztajn et al. removed incomplete rows, so that I do not apply its conclusion to a missing-heavy dataset.
9. As a user, I want to know the TabArena result includes post-hoc ensembling, so that I understand why the leaderboard ordering shifts.
10. As a user, I want a second decision step within the chosen family, so that choosing "forest" does not leave me at random forest by default.

### Bagging Family Internals

11. As a user, I want to know when to pick `bagging` over `rf`, so that I am not adding feature subsampling I do not need.
12. As a user, I want to know when to pick `extra_trees`, so that I reach for it deliberately.
13. As a user, I want to know when to pick the committee over the linear-leaf forest, so that I understand the sequential versus parallel difference.
14. As a user, I want to know the committee and the linear-leaf forest are not the same method, so that I do not treat them as interchangeable.

### Meta Family Internals

15. As a user, I want a precondition check before stacking, so that I do not build a stack over correlated members.
16. As a user, I want the member correlation reported, so that I can check the precondition myself.
17. As a user, I want to know that stacking needs out-of-fold predictions, so that I understand the fold phase's cost.

### Bayesian Family Internals

18. As a user, I want to know when the posterior is the deliverable rather than the accuracy, so that I choose `ModelBayes` for the right reason.
19. As a user, I want the difference between a prior-regularised tree and the posterior stated, so that I do not mistake one for the other.
20. As a user, I want the interval provenance named, so that I know whether a band is conformal, bootstrap, or posterior.
21. As a user, I want to know the sampler's cost before choosing it, so that I do not pick an unaffordable estimator.

### Documentation and Agent Use

22. As a documentation reader, I want the decision procedure in the docs, so that I can find it without reading source.
23. As an agent user, I want the agent skill to apply the same procedure, so that the agent's choice matches the documented guidance.
24. As a library maintainer, I want the four family specs to reference one framework, so that the guidance cannot drift between them.

## Implementation Decisions

### Decision 1: The Framework Is a Specification, Not Only Prose

The framework lives in this specification with explicit tables, and the family specs reference it. It is not a blog post.

Reason: the user's stated priority is that this concept gets ingrained. Guidance that lives only in a chat answer or a blog post does not get ingrained. A specification with tables, referenced from three other specifications, gets updated when the research changes.

### Decision 2: Hard Constraints Come First

The procedure checks requirements before data properties.

Reason: a requirement like "every prediction must be traceable to readable rules" eliminates the boosting family regardless of how the data looks. Checking the data regime first would bury that.

### Decision 3: The Default Is the Forest

`ModelForest(kind="rf")` with 200 members and `max_features="sqrt"`.

Reason: the forest has the smallest catastrophic failure mode. It cannot overfit badly at default settings, it trains in parallel, it needs no validation split, and its one diagnostic caveat is documented. The alternative default, boosting, requires a tuning budget and a validation split to be safe, and a mistuned booster is worse than a forest.

This is a deliberate choice to make the low-effort path good rather than the peak-accuracy path the default.

### Decision 4: The Benchmark Evidence Is Cited With Its Scope Every Time

Every table that leans on a benchmark result names the dataset count, the tuning budget, the missing-data handling, and the protocol.

Reason: the most common way benchmark evidence gets misused is by dropping the scope. Grinsztajn et al. is IID, medium-sized, complete-case, and heavily tuned. TabArena's post-hoc-ensembling result changes the ordering. A user who reads "boosting wins" without that scope will pick the wrong method for their data.

### Decision 5: The Decision Procedure Is Not Automated

There is no `recommend_ensemble()` function that inspects the data and returns a family.

Reason: the hard constraints and the cost model are not derivable from the data. A function could only answer the data-regime part, and a function that returns one answer would hide the human decision that the framework is meant to make explicit. The framework is documentation plus an agent skill entry.

### Decision 6: The Distribution Argument Is About This Library

The claim that a distributional output favours `ModelForest` is a claim about `mltreelib`, not about the literature.

Reason: `ModelTree` already carries survival, quantile, and uplift leaves. A forest of those members inherits them. A generic GBDT with constant leaves does not produce them, and a GBDT with distributional leaves is a different research project. The recommendation follows from the library's actual contents, and it would not follow for a library without those leaves.

### Decision 7: The Framework Names the Cases It Does Not Decide

The stage 2 table says "usually wins", not "wins". Where the evidence does not settle a case, the framework says so and points to the benchmark protocol.

Reason: a decision framework that pretends every case is settled is worse than no framework, because it teaches false confidence. The honest output of stage 2 is a leaning, and stage 1 and stage 3 are where the real decisions happen.

### Decision 8: The Agent Skill Applies the Framework Verbatim

The ensemble entry in the `mltreelib` agent skill asks the hard-constraint questions first, then applies the stage 2 and stage 3 tables, then states the default.

Reason: an agent that reasons from source code will reach different conclusions on different days. The skill entry pins the reasoning so the agent's answer matches the documentation.

### Decision 9: The Framework Is Versioned with the Library

The framework carries a version marker, and the family specs record which version they were written against.

Reason: the underlying benchmark evidence will change as TabArena and its successors update. A framework that silently drifts from the evidence is the same failure the framework is meant to prevent.

## Exact Build Sequence

This specification produces documentation and an agent skill entry, not runtime code. It lands in phases alongside the family specs.

### Phase 0 — Framework Document

1. Write the decision procedure page under `docs/`, with the stage 1 to stage 4 tables and the caveats.
2. Add the decision section to each family spec, pointing at this specification.
3. Add a framework version marker.

Gate: the page exists, the family specs link it, and the tables match this specification.

### Phase 1 — Agent Skill Entry

1. Add the ensemble entry to the `mltreelib` agent skill.
2. The entry asks the hard-constraint questions first, applies the stage 2 and stage 3 tables, and states the default.
3. Add a worked example for a missing-value-heavy categorical dataset and for a latency-constrained dataset.

Gate: the skill entry reproduces the framework's decision for both worked examples.

### Phase 2 — Benchmark Validation

1. Add a benchmark that runs the default forest, a tuned forest, and a tuned booster across the TabArena-style datasets.
2. Record the actual ordering in this repository's data, not in a cited paper.
3. Update the stage 2 table if the local evidence disagrees with the cited tendencies.

Gate: the local benchmark exists and the stage 2 table reflects it.

## Testing Decisions

- **Framework consistency seam.** The tables in this specification, in the family specs, in the documentation page, and in the agent skill entry must agree. A test or a check compares the decision tables across those four locations.
- **Worked example seam.** The two worked examples in the agent skill entry must reach the same conclusion through the framework as the framework's own worked examples.
- **Benchmark seam.** The phase 2 benchmark asserts that the default forest and a tuned forest both beat a mistuned booster on at least one dataset, which is the claim the default rests on.
- **Documentation seam.** Every family spec carries a link to this specification, and this specification links back to all three.
- **Good tests assert behaviour.** The framework's tests assert that the documented decision paths produce the documented recommendation. They do not test that prose reads well.

## Out of Scope

- An automated `recommend_ensemble()` function.
- Auto-tuning infrastructure. The framework states the tuning budget requirement; it does not build a tuner.
- A comprehensive benchmark harness. Phase 2 adds a small one; the full harness is Spec 21's job.
- Selection guidance for the non-ensemble families: clustering, anomaly detection, component analysis, and the regression leaf families. Each has its own guidance in its own specification.
- Neural and foundation-model alternatives. Their comparison is a separate research question and the framework deliberately stays inside tree ensembles.
- Distribution-shift-specific guidance beyond the note that a validation split resembling the target population matters more than the algorithm choice.

## Further Notes

- The research basis for the benchmark evidence, the family distinctions, and the API precedent: `references/research/2026-11-22-ensemble-families-and-api-architecture.md` and `references/research/2026-11-22-forest-methods-landscape.md`.
- Spec 28 (`ModelForest`), Spec 29 (`ModelBoost`), and Spec 30 (`ModelMeta`) are the three families this specification governs.
- The framework's default path is a claim about `mltreelib`'s strengths, not a general claim about forests. If the library's leaf and constraint machinery changes, the default's justification changes with it.
- The benchmark evidence cited here is scoped to IID tabular data. The distribution-shift evidence, from TableShift and the shift-taxonomy work, says that validation split design matters more than the algorithm choice when the target population differs from the training population. That belongs in stage 3, as a note under the cost model.
