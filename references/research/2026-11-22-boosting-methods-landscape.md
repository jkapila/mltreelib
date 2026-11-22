# Boosting Methods Landscape for `mltreelib`

**Date:** 2026-11-22
**Scope:** Gradient boosting mechanics, DART, LightGBM's boosting-type modes, CatBoost's ordered boosting, XGBoost's linear booster, monotone constraints inside tree boosting, EBM and GA²M as an interpretability comparator, and the implementation feasibility of each variant on this library's existing tree engine.
**Decision informed:** Whether the library should build a boosting family, which variants are buildable on the current tree engine, and what bar a new implementation must clear.
**Confidence:** High for the mechanics verified from primary sources and source code. Explicitly marked gaps where a claim could not be verified.
**Code changes:** None. Research and specification only.

## 1. Executive Summary

The boosting family is worth building, and the local benchmark says so. On the library's own 56-dataset Elo arena, the boosting family occupies the top of the table: CatBoost(200) 1457.8, `perpetual (Rust)` 1400.1, ChimeraBoost 1340.0, sklearn GBM 1274.0, HistGradientBoosting 1258.6, LightGBM(200) 1240.9. `ModelTree(all)` sits at 1290.1.

But the bar is set by an existing Rust library, and the honest finding is that a plain GBDT is not the deliverable.

**Finding 1: `perpetual (Rust)` already ranks third overall**, at 1400.1 overall, 1505.9 regression Elo, 1314.7 classification Elo, with a 7.19 s mean fit time. It beats CatBoost(200) on regression Elo, and beats LightGBM(200), sklearn GBM, and sklearn RF on overall Elo. It does this with no leaf families, no monotone constraints, no categorical partition search, and no native missing-value routing.

**Finding 2: this library's tree engine is better suited to boosting than XGBoost's or LightGBM's.** Both mainstream implementations must special-case leaf fitting to a Newton constant $w = -\sum g / (\sum h + \lambda)$, because a constant cannot express a model. That special case is why neither can offer linear leaves, monotone leaf coefficients, or distributional leaves in a stage. `mltreelib` can fit a stage with its existing builder on pseudo-residuals, and gets all three for free from `LeafKind`, `LeafFamily`, and `leaf_monotone`.

For squared error, $u_i = y_i - F_{m-1}(x_i)$ and the least-squares leaf fit gives $\sum u_i / n$, which equals $-\sum g / (\sum h + \lambda)$ when $h_i = 1$. The two formulations coincide there, and the library's formulation additionally supports the richer leaf. For logistic loss the pseudo-residual fit approximates the Newton step unless the leaf family is the matching GLM, which `LeafFamily` makes selectable rather than hidden.

**Finding 3: the performance lever is histogram binning, and it is a prerequisite.** On the local regression board, LightGBM(200) fits in 0.34 s mean while ranger takes 17.78 s and sklearn RF 18.56 s. The difference is not the algorithm: LightGBM bins each numeric feature once into at most 255 bins and scans the binned histogram at every node. The library already has `SplitEval::Histogram` with 256 bins, currently gated to gini, entropy, gain ratio, and twoing. Lifting that gate to the regression gains is a smaller project than any boosting algorithm, and it is the shared performance foundation for all three ensemble families.

**Finding 4: three citations in common circulation are wrong.** DART is Rashmi and Gilad-Bachrach, not a Müller, Mentrup, and Guestrin NeurIPS 2015 paper. ChimeraBoost is a PyPI package and GitHub repository, not a paper. GA²M is Lou, Caruana, Gehrke, and Hooker at KDD 2013, DOI `10.1145/2487575.2487579`.

## 2. The gradient boosting core

### 2.1 Friedman's functional gradient descent

For empirical risk

$$\widehat C(F) = \frac{1}{n}\sum_i \rho(y_i, F(x_i)),$$

boosting computes the negative gradient

$$u_i = -\left.\frac{\partial \rho(y,f)}{\partial f}\right|_{f = F_{m-1}(x_i)},$$

fits a base learner $\hat g_m$ to the pseudo-responses $u_i$, and performs a line search

$$F_m = F_{m-1} + \hat s_m \hat g_m.$$

Friedman's greedy function approximation writes this as

$$\hat F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma), \qquad \hat F_m(x) = \hat F_{m-1}(x) + \nu\, h_m(x),$$

$$h_m(x) = \arg\min_\gamma \sum_{i=1}^{n} L\big(y_i, \hat F_{m-1}(x_i) + \gamma\big),$$

where $h_m$ is fitted by regression on the negative gradient, so for squared error $\gamma_m$ is the mean residual, and $\nu$ is the learning rate or shrinkage.

Friedman's stochastic gradient boosting fits each iteration's base learner on a **random subsample** of the training cases with fraction $\nu$, with the update remaining $F_m = F_{m-1} + \nu h_m$. The same $\nu$ acts as learning rate and subsample fraction, with $0 < \nu \le 1$; $\nu = 1$ recovers deterministic boosting.

Sources: Friedman (2001), DOI [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451); Friedman (2002), DOI [10.1214/aos/1016345136](https://doi.org/10.1214/aos/1016345136); Bühlmann & Hothorn (2007), DOI [10.1214/07-STS242](https://doi.org/10.1214/07-STS242), [manuscript](https://stat.ethz.ch/Manuscripts/buhlmann/BuehlmannHothorn_Boosting-rev.pdf).

### 2.2 The second-order update

XGBoost's regularised objective at step $t$:

$$\mathcal L^{(t)} = \sum_{i=1}^{n}\Big[l\big(y_i, \hat y^{(t-1)}\big) + g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)\Big] + \Omega(f_t), \qquad \Omega(f_t) = \gamma T + \tfrac{1}{2}\lambda\|w_k\|^2,$$

with $g_i = \partial_{\hat y^{(t-1)}} l$ and $h_i = \partial^2_{\hat y^{(t-1)}} l$. The second-order leaf update is $w = -\sum g / (\sum h + \lambda)$, and $\gamma T$ penalises the leaf count.

XGBoost's own documentation defaults: `eta` 0.3, `max_depth` 6, `min_child_weight` 1, `gamma` 0, `lambda` 1, `subsample` 1, `colsample_bytree` 1, `grow_policy` `depthwise` (options `depthwise`, `lossguide`), `max_leaves` 0, `max_bin` 256, plus `monotone_constraints` and `interaction_constraints`.

Sources: Chen & Guestrin (2016), [arXiv:1603.02754](https://arxiv.org/pdf/1603.02754); [XGBoost parameter docs](https://xgboost.readthedocs.io/en/stable/parameter.html).

### 2.3 Histogram binning, and why it dominates fit time

The single largest engineering difference between a fast and a slow booster is binning.

LightGBM's two named techniques are **GOSS** (Gradient-based One-Side Sampling: keep instances with large gradients, subsample the small-gradient ones, keeping all of the large) and **EFB** (Exclusive Feature Bundling: bundle mutually exclusive features; optimal bundling is NP-hard, greedy approximation).

LightGBM's growth is **leaf-wise** (best-first): `max_depth` default −1 with the note "Tree still grows leaf-wise"; `num_leaves` default 31; `min_data_in_leaf` default 20; `min_sum_hessian_in_leaf` default 1e-3; `max_bin` default 255; `linear_tree` default false; `monotone_constraints` accepts per-feature `{-1, 0, 1}`; `data_sample_strategy ∈ {bagging, goss}`.

**A correction to a common claim.** Scikit-learn's `HistGradientBoosting` is often described as depth-wise with a leaf cap, while LightGBM is called leaf-wise. The actual sklearn implementation uses a **gain-priority heap** (`heappop` on the highest gain), which is leaf-wise best-first growth, the same as LightGBM. It is not depth-wise.

Sources: Ke et al. (2017), [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html); [LightGBM parameter docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html); [scikit-learn ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html).

### 2.4 GOSS mechanics

From the LightGBM paper's Algorithm 2, GOSS operates as follows. Let the training instances be ranked by their absolute gradient magnitude, $I = \{i_1, \ldots, i_n\}$ with $|g_{i_1}| \ge \ldots \ge |g_{i_n}|$. Let $a \cdot n$ be the number of large-gradient instances kept, and $b \cdot n$ the number of small-gradient instances sampled. The kept set is

$$\mathcal{I} = \{i_1, \ldots, i_a\} \cup \{i_{a+1}, \ldots, i_{a+b}\},$$

where the first part keeps all large-gradient instances and the second samples from the remainder. The sampled instances receive a weight amplification factor of $\frac{1-a}{b}$ so that the gradient sum is preserved.

The intuition is that large gradients carry most of the information about the loss surface, so discarding them degrades the model, while discarding small-gradient instances with appropriate reweighting is approximately harmless.

## 3. DART

### 3.1 The correct citation

DART is Rashmi and Gilad-Bachrach, *Dropout Adaptive boosting for Machine Learning*, arXiv:1505.01866, PMLR v38 (2015).

**The premise that a Müller, Mentrup, and Guestrin NeurIPS 2015 paper called "Fast Boosting with Outlier Detection" exists is false.** The search covered the NeurIPS 2015 accepted-paper list, the NeurIPS 2015 conference book with author lists, the NeurIPS 2015 workshop book, the arXiv API under title and author queries, Crossref, OpenAlex, Semantic Scholar, CORE, DBLP, and web search. Zero records. The NeurIPS 2015 conference book contains no "Mentrup" or "Guestrin" string at all.

Note also that the DART paper itself contains **zero** occurrences of "outlier". The outlier framing belongs to a different line of work.

Sources: [arXiv:1505.01866](https://arxiv.org/abs/1505.01866); [PMLR v38](http://proceedings.mlr.press/v38/korlakaivinayak15.html).

### 3.2 Algorithm 1, verbatim from the paper

```text
Let N be the total number of trees to be added to the ensemble
S1 ← {x, −L0(x)}
T1 be a tree trained on the dataset S1
M ← {T1}
for t = 2, ..., N do
    D ← the subset of M such that T ∈ M is in D with probability p_drop
    if D = ∅ then D ← a random element from M
    M̂ ← M \ D
    St ← {x, −L0(x) − M̂(x)}
    Tt be a tree trained on the dataset St
    M ← M ∪ {Tt}
    for T ∈ D do
        Multiply T in M by a factor of |D| / (|D| + 1)
    end for
end for
Output M
```

The dropped trees are scaled by $|D|/(|D|+1)$, and the new tree absorbs the correction. This is the dropout that regularises the ensemble by forcing later stages not to duplicate earlier ones.

### 3.3 The paper's results

From the paper's Table 4, L2 error on the CT slices regression task, with the best parameter combination for each ensemble size:

| Ensemble size | 25 | 50 | 100 | 250 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| MART | 35.13 | 31.79 | 30.92 | 30.07 | 29.76 | 29.28 |
| DART | 32.50 | 30.50 | 29.66 | 28.14 | 28.11 | 27.98 |
| Random Forest | 32.76 | 33.21 | 32.88 | 32.36 | 32.66 | 32.33 |

DART beat MART at every ensemble size and beat random forest at every size. The paper's ranking-task result was approximately 0.4 NDCG points over MART, with 0.2 points at position 1 and 0.38 at position 2. The paper notes that in the Yahoo! learning-to-rank challenge the gap between the winner and the fifth-place entry was about 0.35 points.

The paper's parameter scan for regression (Table 3): shrinkage 0.05, 0.1, 0.2, 0.3, 0.5 for MART and none for DART; dropout rate $\epsilon$ in 0, 0.01, 0.025, 0.05, 0.1, 0.2 for DART; fraction of instances per tree 1.0 for MART and DART, 0.25/0.5/0.75/1.0 for random forest; number of trees 25 to 1000; leaves per tree 50 to 1000; fraction of features scanned per leaf 0.05 to 1.0.

### 3.4 XGBoost's implementation differs from the paper in two ways

Verified from `src/gbm/gbtree.cc` in the XGBoost repository.

**Weighted dropping.** The paper drops uniformly with probability $p_{drop}$. XGBoost drops with probability proportional to each tree's weight. The condition in source is:

```cpp
if (runif(rnd) < dparam_.rate_drop * model_.weight_drop.size() * model_.weight_drop[i] / sum_weight)
```

**Rescaling.** A `normalize_type` option selects the post-drop rescaling. One variant computes `factor = num_drop / (num_drop + lr)`; another pushes `1.0 / (num_drop + lr)` onto the weight vector. The `num_drop` path returns early when `num_drop == 0`.

**Skip and count.** `skip_drop` is the probability of skipping the dropout procedure entirely in an iteration: `skip = (runif(rnd) < dparam_.skip_drop)`. `num_drop` caps the number of dropped trees.

Source: [xgboost `src/gbm/gbtree.cc`](https://github.com/dmlc/xgboost/blob/master/src/gbm/gbtree.cc).

### 3.5 LightGBM's DART parameters

Verified from `docs/Parameters.rst`:

| Parameter | Default | Meaning |
|---|---:|---|
| `drop_rate` | 0.1 | Fraction of previous trees to drop; used only in `dart`. Alias `rate_drop` |
| `max_drop` | 50 | Maximum trees dropped in one iteration; `<= 0` means no limit |
| `skip_drop` | 0.5 | Probability of skipping the dropout procedure in an iteration |
| `xgboost_dart_mode` | false | Switch to XGBoost's DART mode |
| `uniform_drop` | false | Uniform dropping instead of weighted |
| `drop_seed` | 4 | Random seed for the drop selection |

The `xgboost_dart_mode` flag exists precisely because LightGBM's default DART differs from XGBoost's. A library implementing DART must pick one semantics and document it.

Source: [LightGBM parameter docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

## 4. LightGBM's three boosting types

LightGBM exposes `boosting_type` with three values.

**`gbdt`.** Standard gradient boosting. Each stage fits a tree to the negative gradient and adds it with the learning rate.

**`dart`.** The dropout variant above.

**`rf`.** Random forest *as a booster*. Each boosting iteration fits a random forest to the negative gradient and adds the forest's prediction as the stage contribution. The source contains a dedicated driver path for this mode.

The distinction matters for the library's own design. `rf` mode is bagging inside a boosting loop, so it needs the forest machinery from Spec 28 *and* the stage loop from Spec 29. It is deferred until both exist.

Source: [LightGBM `src/boosting/`](https://github.com/LightGBM/LightGBM/tree/master/src/boosting).

## 5. CatBoost's ordered boosting

### 5.1 Ordered target statistics

Prokhorenkova et al. (2018) identify a specific failure of plain GBDT with categorical features, called **prediction shift**.

The mechanism: a categorical feature's target statistic is computed from the training data, and a case's own target contributes to that statistic. When the model later predicts on that case, the feature value carries information the model should not have about that case's target. The statistics are computed on the training distribution but applied at prediction time, where the case does not contribute to its own encoding, so the encodings are systematically shifted.

CatBoost's fix is to compute each statistic from a **random permutation** of the data, so a case's own target never enters its own statistic. The paper uses two permutations: $\sigma_{cat}$ for the statistics and $\sigma_{boost}$ for the boosting rounds.

Sources: Prokhorenkova et al. (2018), [arXiv:1706.09516](https://arxiv.org/pdf/1706.09516); [CatBoost parameter docs](https://catboost.ai/docs/en/features/parameters-main).

### 5.2 Ordered boosting

Ordered boosting extends the permutation idea to the stage structure. Rather than fitting each stage on the full data with full-data statistics, each stage uses statistics computed from a strictly earlier permutation, through **supporting models** fit on subsets of the data.

The paper reports that ordered boosting reduces prediction shift and that the resulting model overfits less, and that the ordered variant has the same asymptotic complexity as the plain one.

### 5.3 Oblivious trees and parameters

CatBoost's base predictors are **oblivious** decision trees: every node at a given depth splits on the same feature and threshold. The tree can be represented as a $2^d$ leaf table indexed by a bit vector.

Parameters: `boosting_type` is `Plain` or `Ordered`. `bootstrap_type` is `Bayesian`, `Bernoulli`, `MVS`, or `No`.

**Verified default behaviour.** The current CPU default is `Plain`. On GPU, the documented rule is `Plain` for `MultiClass` and `MultiClassOneVsAll`, and `Plain` for more than 50,000 objects, with `Ordered` used otherwise. The `v1.0.0` tag already carried this GPU threshold rule and `v1.2.7` retains it. The exact release that introduced the 50,000-object threshold remains **unverified**; earlier tags used different documentation layouts and the fetched query did not expose comparable text. No verified global `boosting_type` default change was found.

Source: [CatBoost repository](https://github.com/catboost/catboost), tags `v1.0.0` and `v1.2.7`.

### 5.4 What this costs the library

Ordered boosting is **not** a leaf-level change. It changes what the split engine sees, because the statistics available to a split depend on a permutation and on supporting models.

`mltreelib` already has categorical partition search, which is a different approach: it searches over level partitions directly rather than encoding levels into numeric targets. Ordered target statistics and partition search are not obviously composable.

This is the single largest blocker in the boosting family. It is deferred for that reason.

## 6. XGBoost's linear booster

### 6.1 Parameters

Verified from source. `eta` **0.5**, `lambda` **0**, `alpha` **0**, `updater` `shotgun`, `feature_selector` `cyclic`, `top_k` 0. `num_feature` is not a user-facing gblinear parameter.

The commonly cited tutorial URL `xgboost.readthedocs.io/.../tutorials/gblinear.html` does not exist at any released version; the wiki page was deleted. Any formula attributed to it is reconstructed from source, not from the tutorial.

### 6.2 The update

The update is a **soft-thresholded proximal step** with a zero-crossing clamp, `eta` shrinkage, an early-out when the Hessian sum falls below $10^{-5}$, and L1 applied as a *threshold*. Penalties are denormalised by the instance-weight sum.

This differs from the frequently quoted form $w_j \leftarrow (\Sigma(y - (\text{pred} - w_j x))x - \alpha)/(\Sigma x^2 + \lambda)$.

### 6.3 `shotgun` versus `coord_descent`

`updater` selects between `shotgun`, which cycles through features in a round-robin accumulating per-feature statistics and updating in batches, and `coord_descent`, which updates one feature at a time. `shotgun` is the default.

### 6.4 `num_parallel_tree`

Verified from source in both the `hist` and `exact` paths: each parallel tree gets **its own row subsample and its own column subsample**. The layer's contribution is $\eta/K \cdot \sum_k T_k$, the mean of the $K$ trees at the learning rate $\eta$.

Source: [xgboost `src/linear/coordinate_common.h`](https://github.com/dmlc/xgboost/blob/master/src/linear/coordinate_common.h).

## 7. scikit-learn's boosting surfaces

### 7.1 `GradientBoostingRegressor` and `GradientBoostingClassifier`

```python
GradientBoostingRegressor(
    loss="squared_error",
    *,
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion="deprecated",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.1,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0,
)
```

Loss options: `squared_error`, `absolute_error`, `huber`, `quantile`.

**Corrections to commonly stated facts.**

- `criterion`'s live default is the sentinel `'deprecated'`. The parameter is a **no-op**, deprecated since 1.9, with removal planned in 1.11.
- `init` accepts `None`, the literal `'zero'`, or an estimator with `fit` plus `predict` for a regressor, or `fit` plus `predict_proba` for a classifier. There is **no** `'estimator'` string form and **no** array-of-predictions form.
- Early stopping does **not** require `subsample < 1`. Early stopping uses its own `train_test_split`; `subsample < 1` only enables the out-of-bag attributes.
- `subsample` performs per-tree subsampling of the training cases before fitting each base learner, and the out-of-bag attributes are only computed when `subsample < 1`.

### 7.2 `HistGradientBoostingRegressor`

```python
HistGradientBoostingRegressor(
    loss="squared_error",
    *,
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_features=1.0,
    max_bins=255,
    categorical_features="from_dtype",
    monotonic_cst=None,
    interaction_cst=None,
    warm_start=False,
    early_stopping="auto",
    scoring="loss",
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-07,
    verbose=0,
    random_state=None,
)
```

**The `early_stopping="auto"` rule is `n_samples > 10_000 or validation_data_provided`**, not `validation_fraction < 1 and n_iter_no_change is not None`.

It replaces `n_estimators` with `max_iter`, supports native missing values and categorical features, and, as noted above, uses **leaf-wise best-first growth** via a gain-priority heap.

Sources: [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html), [ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html).

## 8. Monotone constraints inside tree boosting

### 8.1 XGBoost

`monotone_constraints` takes `+1` for nondecreasing, `-1` for nonincreasing, `0` for unconstrained. An example `(1, -1)` means the first feature increasing and the second decreasing.

XGBoost enforces the constraint during tree construction, but **not** by a "weighted-count bound". The current implementation, in `src/tree/split_evaluator.h`, uses leaf-weight ordering with propagated bounds. For a candidate split on feature $f$, the unconstrained leaf weights are

$$w_L = -\frac{G_L}{H_L + \lambda}, \qquad w_R = -\frac{G_R}{H_R + \lambda}.$$

- An increasing constraint accepts only $w_L \le w_R$.
- A decreasing constraint accepts only $w_L \ge w_R$.
- A violating candidate gets $-\infty$ gain, or the constrained child weights are replaced by a common pooled weight.

Bound propagation sets $m = (w_L + w_R)/2$. For an increasing feature the left child inherits upper bound $m$ and the right child inherits lower bound $m$; for a decreasing feature the bounds swap. Descendant leaf weights are clipped to the inherited $[L, U]$ interval.

**The official documentation warns** that `hist` and `approx` can yield shallow trees under constraints, because constrained candidates may eliminate all splits at a node, and recommends increasing `max_bin` to expose more candidates.

Source: [XGBoost monotone tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html), [split_evaluator.h](https://github.com/dmlc/xgboost/blob/master/src/tree/split_evaluator.h).

### 8.2 LightGBM

`monotone_constraints` takes `1` increasing, `-1` decreasing, `0` unconstrained, and **all features must be specified in order**. An example is `[-1, 0, 1]`. Defaults: `monotone_constraints` unset, `monotone_constraints_method="basic"`.

Source: [LightGBM parameter docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

### 8.3 The ensemble-level problem

Both implementations constrain **within** a tree. Neither guarantees the property on the **summed ensemble**, which is what a monotone-constraint claim must mean.

A constraint that says "feature $f$ must not decrease the prediction" has to hold for $\sum_m \nu h_m(x)$ over all stages. If stage 1 raises the prediction on $f$ and stage 2 lowers it, the ensemble can violate the constraint even when each stage respects it.

`mltreelib` already has `tree_monotone` for split direction and `leaf_monotone` for leaf coefficients, and both act within one tree. The booster needs a third mechanism: a per-feature bound vector carried across stages, with each stage's leaf values clipped to the bounds implied by the accumulated stages, and the clipping frequency reported.

## 9. EBM and GA²M as an interpretability comparator

### 9.1 The correct citation

**GA²M** means Generalised Additive Models plus Interactions. The definitive paper is Yin Lou, Rich Caruana, Johannes Gehrke, and Giles Hooker, "Accurate intelligible models with pairwise interactions", KDD 2013, pp. 623–631, DOI [10.1145/2487575.2487579](https://doi.org/10.1145/2487575.2487579), [PDF](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf).

**arXiv:1303.3217 is not GA²M.** It resolves to Roberto Mossa, "A note on diastatic entropy and balanced metrics", math.DG.

Two related papers: Lou, Caruana, Gehrke, Hooker, "Intelligible models for classification and regression", KDD 2012, DOI [10.1145/2339530.2339556](https://doi.org/10.1145/2339530.2339556); Caruana, Lou, Gehrke, Koch, Sturm, Elhadad, "Intelligible Models for HealthCare", KDD 2015, DOI [10.1145/2783258.2788613](https://doi.org/10.1145/2783258.2788613).

### 9.2 The GA²M model

$$g\big(\mathbb{E}[y \mid x]\big) = \sum_{i=1}^{n} f_i(x_i) + \sum_{1 \le i < j \le n} f_{ij}(x_i, x_j).$$

$f_i(x_i)$ is a univariate shape function, plotted as a curve. $f_{ij}(x_i, x_j)$ is a pairwise shape function, plotted as a 2D heatmap. Centred components satisfy $\mathbb{E}[f_u] = 0$, with the paper using $\mathbb{E}[f_u^2] < 1$ as Hilbert-space normalisation.

The objective is $F^\star = \arg\min_{F \in H} \mathbb{E}[L(y, F(x))]$, with $L$ a nonnegative convex loss; squared loss gives regression, logistic loss classification.

The paper's conceptual algorithm uses greedy forward stagewise selection: fit an additive model, compute the residual $R = y - F(x)$, for each candidate pair fit an interaction predictor on the residual, choose the pair giving the largest squared-error reduction, add it, and repeat until no gain.

### 9.3 FAST interaction detection

FAST, the Frequent Approximate Subset of Top-dimension Interactions test, searches for influential feature pairs without an exhaustive search. The score for a pair combines a statistic on the pair's joint distribution with a marginal statistic on each feature.

Source: Lou et al. (2013), Section 4 and the FAST material.

### 9.4 EBM's training shape

EBM is a glassbox generalised additive model with pairwise interactions. It "learns each feature function $f_j$ using modern machine learning techniques such as bagging and gradient boosting. The boosting procedure is carefully restricted to train on **one feature at a time in round-robin fashion** using a **very low learning rate** so that feature order does not matter". It round-robins to mitigate the effects of collinearity, and it can automatically detect and include pairwise terms.

interpretML fitted-parameter defaults: `max_bins` 1024, `max_interaction_bins` 64, `learning_rate` 0.015, `greedy_ratio` 10.0, `smoothing_rounds` 75, `interaction_smoothing_rounds` 75, `max_rounds` 50000, `early_stopping_rounds` 100, `min_samples_leaf` 4, `min_hessian` 0.0001, `reg_alpha` 0.0, `reg_lambda` 0.0, `max_leaves` 2 (classifier default), `interactions` `'3x'`, `monotone_constraints` None, `outer_bags` 14.

Source: [interpretML docs](https://interpret.ml/docs/), Nori et al. (2019), [arXiv:1909.09223](https://arxiv.org/pdf/1909.09223).

### 9.5 GA²M versus EBM

| Property | GA²M, Lou 2013 | Current EBM |
|---|---|---|
| Function class | Main effects plus selected pairs | Main effects plus selected pairs; higher-order terms possible |
| Pair detection | FAST on residual; conceptual greedy selection | FAST or Hessian-based interaction detection |
| Pair fitting | Stage-2 residual pair shaping with a three-cut shallow model | Per-pair cyclic gradient boosting on a binned 2D tensor |
| Main fitting | Gradient boosting with shallow tree-like shape functions | Cyclic or greedy gradient boosting, one term per step |
| Bagging | The paper's model is not defined by EBM's outer and inner bag machinery | Outer and optional inner bagging |
| Typical output | 1D curves and 2D heatmaps | Same, plus higher-order tensors |
| Runtime | Research algorithm | Parallel C++ implementation |

Avoid the shorthand "EBM is GA²M plus stronger regularisation". The more accurate statement is that EBM is a fast modern GA²M-style implementation using cyclic Newton boosting, FAST-style detection, bagging, smoothing, and direct tensor representation. The official interpretML README describes EBMs as a "fast derivative of GA2M".

### 9.6 Monotone EBM

**There is no verified "AISTATS 2024 monotone EBM" or "ExMonotone" paper.** The verification searched the AISTATS proceedings listings, arXiv, Crossref, and OpenAlex for such a title and found no matching record.

Training-time monotone constraints in EBM do exist, and post-fit monotonisation of the learned shape functions is also available in the package. The closest verified substitutes for a published monotone-EBM method are the post-hoc monotonisation literature and the monotone-constraint work inside tree boosting.

### 9.7 Why EBM is a comparator and not a family member

EBM is **not a tree ensemble**. It is additive in features with pairwise interactions, where each feature function is learned by boosting over that feature alone. The library's `mars.rs` leaf is the closer analogue, since MARS is also an additive model with basis expansions.

On the local board EBM scores 1353.6 overall, 1383.4 regression, 1268.2 classification, with a 66.59 s mean fit time, the slowest entrant measured. It is the interpretability benchmark that a linear-leaf or additive leaf family should be compared against, not a candidate for `ModelBoost`.

## 10. ChimeraBoost is a repository, not a paper

**Verified negative result.** arXiv's API returns `totalResults` 0 for `all:ChimeraBoost`, `all:"Chimera Boost"`, and `ti:ChimeraBoost`. Crossref returns `total-results: 0` for `query.bibliographic`, `query.title`, and `query` queries. OpenAlex returns three works, all unrelated, matched on abstract text rather than title. DBLP was bot-challenged and Semantic Scholar was rate-limited, so those two are "not checked" rather than negative.

**What ChimeraBoost actually is.** PyPI package `chimeraboost`, version 0.34.0, described as "CatBoost-inspired gradient boosting in pure Python with a numba backend", repository `bbstats/chimeraboost`, Apache-2.0, 13 stars, created 2026-05-26, last push 2026-11-21. Dependencies: `numpy>=1.22`, `numba>=0.57`, `scikit-learn>=1.0`, `scipy>=1.7`. Thirty-five released versions from 0.5.2 to 0.34.0.

Its own README claims "#2 On TabArena GBDT Elo (Defaults)". Its documented internals: oblivious symmetric trees, histogram binning with `max_bins` default **128**, missing values get their own bin, CatBoost-style ordered target statistics with `cat_n_permutations` default 4 and `cat_smoothing` shrink to the global mean, `cat_count_features` on by default for columns with at least 256 categories, `cat_combinations` for pairwise categorical features, OpenFE-style `cross_features`, `linear_leaves` for per-leaf ridge models, temperature scaling, bagging as a first-class feature with `n_ensembles`, automatic early stopping, and a `quality` parameter from 1 to 5 defaulting to 3.

Its own attribution document states plainly: "Almost everything ChimeraBoost does was invented by someone else." Its claimed original contributions are three: **structure-transfer refit** (`refit_full="replay"`, default on, replaying the winner's splits against gradients computed on all rows and refitting only the leaf values, reported at −34.8% fit time on Grinsztajn and faster on 58 of 59 datasets with accuracy a wash), **gradient-matrix-free quantile split search**, and equivalence-preserving engineering. It explicitly says its speed comes from oblivious trees plus numba plus warm defaults, and that the "replay" refit is what avoids a second full fit.

**The TabArena figures.** The local `references/data/tabarena-leaderboard.csv` rows give "ChimeraBoost (tuned + ensembled)" at 1365.0 Elo, "ChimeraBoost (tuned)" at 1338.0, and "ChimeraBoost (default)" at 1302.0.

**What this means for the library.** ChimeraBoost is a strong competitor and a useful reference implementation, especially for the "replay" refit idea, which is a cheap and general optimisation. It is not a citable paper.

**The closest verified real papers with similar titles**, all confirmed by arXiv id resolution:

| Paper | arXiv | Note |
|---|---|---|
| Chimera: Improving Generalist Model with Domain-Specific Experts | 2412.05983 | Real, unrelated. LLM domain experts. Easy name collision |
| Accelerating Gradient Boosting Machine | 1903.08708 | Lu, Karimireddy, Ponomareva, Mirrokni (Google). Nesterov acceleration for GBM |
| Accelerated Gradient Boosting | 1803.02042 | Biau, Cadre, Rouvière. Independent Nesterov variant |
| Quantized Training of Gradient Boosting Decision Trees | 2207.09682 | LightGBM team |
| SketchBoost: Fast Gradient Boosted Decision Tree for Multioutput Problems | 2211.12858 | Iosipoi, Vakhrushev |
| Minimal Variance Sampling in Stochastic Gradient Boosting | 1910.13204 | Ibragimov, Gusev. The MVS row sampling |
| Gradient Boosting With Piece-Wise Linear Regression Trees | 1802.05640 | Shi, Li, Li. The linear-leaf booster |
| OpenFE: Automated Feature Generation with Expert-level Performance | 2211.12507 | Zhang, Zhang, Fan, Luo |

## 11. Multi-output and multivariate boosting

**Genuine multi-output trees exist and are distinct from fitting one tree per target.** A multi-output tree splits once and carries several leaf models, so the targets share the partition structure. This is what makes multi-output regression tractable on structured targets where the outputs are correlated.

**Citation status.** A paper titled "Multi-output trees for structured data" is widely attributed to Cerda, Varoquaux, and Kégl, and is cited in the multi-output learning literature. It could **not** be verified by title or author search in the arXiv API (title and all-field queries return zero entries), the NeurIPS 2018 proceedings index (1009 papers, no match), PMLR volume 80 (621 papers, no match), or Crossref. Treat the attribution as **UNVERIFIED** and do not cite an arXiv id, DOI, or page range for it.

**Verified substitutes**, each confirmed by resolving the arXiv id and reading the title and authors:

| Paper | arXiv | What it contributes |
|---|---|---|
| Joly, A.; Wehenkel, L.; Geurts, P. "Gradient tree boosting with random output projections for multi-label classification and multi-output regression" | [1905.07558](https://arxiv.org/abs/1905.07558) | Random output projections to decompose a multi-output problem into univariate boosting problems |
| Au, Q.; Schalk, D.; Casalicchio, G.; Schoedel, R. "Component-Wise Boosting of Targets for Multi-Output Prediction" | [1904.03943](https://arxiv.org/abs/1904.03943) | Component-wise boosting, the direct analogue of group lasso across targets |
| Xu, D.; Shi, Y.; Tsang, I. W.; Ong, Y.-S. "A Survey on Multi-output Learning" | [1901.00248](https://arxiv.org/abs/1901.00248) | Survey and taxonomy of multi-output learning |
| Iosipoi, L.; Vakhrushev, A. "SketchBoost: Fast Gradient Boosted Decision Tree for Multioutput Problems" | [2211.12858](https://arxiv.org/abs/2211.12858) | Multi-output GBDT with a sketched approximation |
| Pliakos, P.; Geurts, P.; Vens, C. "Global multi-output decision trees for interaction prediction" | DOI [10.1007/s10994-018-5700-x](https://doi.org/10.1007/s10994-018-5700-x) | Global multi-output trees, the journal-form record |

**The distinction matters for the library.** Fitting one tree per target is embarrassingly parallel and reuses the existing engine, but it loses the shared partition and cannot exploit output correlation. A genuine multi-output tree needs a split criterion that scores a partition across several targets simultaneously, which means a multi-output extension to `GainFn`.

**How the mainstream implementations actually behave.**

- scikit-learn's `MultiOutputRegressor` wraps separate regressors, one per target. There is no multi-output tree in scikit-learn, and scikit-learn's multiclass and multioutput guide documents every estimator as either inherently multi-output or wrapped.
- XGBoost's `multi_output_tree` is off by default. When enabled, the tree structure is shared across targets while the leaf values differ per target, which is a genuine multi-output tree.
- LightGBM's multiclass handling uses one tree per class per iteration with a shared split structure, and multiclass-one-vs-all as an alternative.
- CatBoost's `MultiRMSE` and `MultiLogloss` objectives handle multi-target regression with a multi-target tree structure.

**The library's position.** The library already has a multi-output path in the tree: `predict_vector` returns `Option<Vec<Vec<f64>>>`. A multi-output leaf family would extend that. For the boosting family, a multi-output stage needs a multi-output leaf and a joint Newton step, which is the same blocker as the distributional-output question.

**Verdict: BUILD_LATER, in the forest family first.** A multi-output forest gets shared partitions for free from the existing engine. A multi-output booster needs the joint stage machinery, which is deferred.

## 12. Implementation feasibility on this library's engine

This is the section that decides what to build.

### 12.1 What the library already has

| Need | Existing | Source |
|---|---|---|
| Tree growth with a chosen split criterion | `GainFn`, 30 variants including `FriedmanMse`, `Sse`, `Sdr`, `AbsoluteError`, `Huber` | `criteria.rs` |
| Leaf model fitting with family and solver | `LeafKind`, `LeafFamily`, `LeafSolver` (Irls, Lbfgs, Ccd, CmaEs, Shade, NaturalGradient, GeneticAlgorithm) | `leaf.rs` |
| Monotone constraints on splits and leaf coefficients, separately | `tree_monotone`, `leaf_monotone` | `model_tree.rs` |
| Missing-value routing with per-node stored direction | present | `split.rs` |
| Categorical partition search | `CatOrder`, `CatSearch` | `split.rs` |
| Binned split evaluation | `SplitEval::Histogram`, 256 bins, classification gains only | `split.rs` |
| Weighted fits with fractional weights | `fit_weighted_named` | `model_tree.rs` |
| Monotone-constrained linear leaves | `leaf_monotone` plus `l1`/`l2` | `leaf.rs` |
| Shared objective, gradient, penalty, budget, and convergence semantics | Spec 26 solver seam | `solver/generic/` |
| A `Custom` split criterion hook | `GainFn::Custom(Arc<dyn CustomGain>)`, receiving lazy left and right `(y, w)` slices | `criteria.rs` |

### 12.2 Buildable on the existing engine

**`gbdt`.** Needs a gradient contract, a stage loop, and the ensemble-level monotone bound. The tree engine, the leaf machinery, the monotone constraints, the missing-value routing, the categorical search, and the binned split evaluation all exist.

**`dart`.** Everything `gbdt` needs, plus the drop selection, the $|D|/(|D|+1)$ rescale, the skip probability, and the drop cap. Self-contained.

**`linear`.** A coordinate-descent regularised linear stage. Needs a non-tree contribution to the stage update, plus the solver seam, which already has a `Lbfgs` path and a generic objective.

### 12.3 Needs new machinery

**Ordered boosting.** Needs ordered target statistics and supporting models, which changes the split engine rather than the leaf fit. The library's categorical handling is partition search, not target encoding, so the two are not obviously composable. Largest blocker.

**Oblivious trees.** Needs a level-synchronous growth mode in the tree builder, plus a level-wise evaluation path. Mechanically simple, but it changes the growth code and the leaf representation.

**Histogram binning for the regression gains.** The gate is `uses_class_histogram`, which admits only gini, entropy, gain ratio, and twoing. Lifting it to `Sse`, `FriedmanMse`, `Sdr`, `AbsoluteError`, `Huber`, and `Deviance` is a contained change, and it is a prerequisite rather than an optimisation, because a 300-stage booster over raw scans costs 300 times a single tree's fit time.

**Multi-output stages.** Needs a multi-output leaf and a joint Newton step.

**Distributional stages.** Needs a multi-parameter leaf and a joint Newton step across the parameters. Same blocker, and the reason the forest family owns distributional output instead.

### 12.4 The verdict table

| Method or feature | Verdict | One-line reason |
|---|---|---|
| `gbdt` with the existing tree engine | **BUILD_NOW** | Stage loop plus a gradient contract; the tree engine, leaf machinery, and constraints all exist |
| `dart` | **BUILD_NOW**, second wave | Self-contained addition to the stage loop; the paper's Algorithm 1 is fully specified |
| `linear` stage, the `gblinear` shape | **BUILD_LATER** | Needs a non-tree stage contribution and coordinate descent, both available in the solver seam |
| `forest_stage`, the LightGBM `rf` shape | **BUILD_LATER** | Needs Spec 28's forest machinery inside a stage |
| Ordered boosting with ordered target statistics | **BUILD_LATER** | Changes the split engine; not composable with partition search without new design |
| Oblivious symmetric trees | **BUILD_LATER** | Needs level-synchronous growth and evaluation in the tree builder |
| Histogram binning for the regression gains | **BUILD_NOW** | Contained change; prerequisite for any competitive fit time |
| Ensemble-level monotone bounds | **BUILD_NOW** | Small; a bound vector across stages with clipping and a report |
| Second-order Newton leaf weights | **BUILD_LATER** | The pseudo-residual formulation coincides for squared error; the Newton constant only matters for other losses |
| Multi-output boosting | **BUILD_LATER** | Needs a multi-output leaf and a joint Newton step |
| Distributional boosting output | **OUT_OF_SCOPE** | Needs a multi-parameter stage; the forest family covers it instead |
| Bayesian boosting | **OUT_OF_SCOPE** | Posterior sampling is a separate family with a different inferential contract |
| EBM and GA²M | **ORACLE_ONLY** | Additive models, not tree ensembles; the interpretability comparator for a linear-leaf family |
| NGBoost | **ORACLE_ONLY** | Multiparameter probabilistic output; covered by the component-analysis note's scope |
| ChimeraBoost as a citable method | **ORACLE_ONLY** | A repository, not a paper; use it as a reference implementation and benchmark oracle |
| DART under XGBoost's weighted semantics | **BUILD_NOW**, second wave | The drop condition and rescale are simple; implement the paper's uniform default and expose weighted as an option |
| LightGBM's GOSS sampling | **BUILD_LATER** | Needs gradient-ranked subsampling inside the stage; a clean standalone win over uniform subsampling |
| "Müller, Mentrup, Guestrin 2015 DART" | **OUT_OF_SCOPE** | The paper does not exist; use Rashmi and Gilad-Bachrach |
| The `gblinear` tutorial formula | **OUT_OF_SCOPE** | The tutorial URL does not exist at any released version; the source differs from the quoted form |

## 13. Citation list

1. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics* 29(5):1189–1232. [https://doi.org/10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
2. Friedman, J. H. (2002). "Stochastic Gradient Boosting." *Annals of Statistics* 30(1):115–123. [https://doi.org/10.1214/aos/1016345136](https://doi.org/10.1214/aos/1016345136)
3. Bühlmann, P.; Hothorn, T. (2007). "Boosting Algorithms: Regularization, Prediction and Model Fitting." *Statistical Science* 22:477–505. [https://doi.org/10.1214/07-STS242](https://doi.org/10.1214/07-STS242)
4. Bühlmann, P.; Yu, B. (2003). "Boosting with the L2 Loss: Regression and Classification." *JASA* 98(462):324–339. [https://doi.org/10.1198/016214503000125](https://doi.org/10.1198/016214503000125)
5. Chen, T.; Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." [arXiv:1603.02754](https://arxiv.org/pdf/1603.02754)
6. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017. [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
7. Prokhorenkova, L. et al. (2018). "CatBoost: unbiased boosting with categorical features." [arXiv:1706.09516](https://arxiv.org/pdf/1706.09516)
8. Rashmi, K. V.; Gilad-Bachrach, R. (2015). "Dropout Adaptive boosting for Machine Learning." PMLR v38. [arXiv:1505.01866](https://arxiv.org/abs/1505.01866) · [PMLR](http://proceedings.mlr.press/v38/korlakaivinayak15.html)
9. Lou, Y.; Caruana, R.; Gehrke, J.; Hooker, G. (2013). "Accurate intelligible models with pairwise interactions." KDD 2013, 623–631. [https://doi.org/10.1145/2487575.2487579](https://doi.org/10.1145/2487575.2487579) · [PDF](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)
10. Lou, Y.; Caruana, R.; Gehrke, J.; Hooker, G. (2012). "Intelligible models for classification and regression." KDD 2012. [https://doi.org/10.1145/2339530.2339556](https://doi.org/10.1145/2339530.2339556)
11. Caruana, R.; Lou, Y.; Gehrke, J.; Koch, P.; Sturm, M.; Elhadad, N. (2015). "Intelligible Models for HealthCare." KDD 2015. [https://doi.org/10.1145/2783258.2788613](https://doi.org/10.1145/2783258.2788613)
12. Nori, H. et al. (2019). "InterpretML: A Unified Framework for Machine Learning Interpretability." [arXiv:1909.09223](https://arxiv.org/pdf/1909.09223) · [docs](https://interpret.ml/docs/)
13. Shi, Y.; Li, J.; Li, Z. (2019). "Gradient Boosting With Piece-Wise Linear Regression Trees." IJCAI 2019. [arXiv:1802.05640](https://arxiv.org/pdf/1802.05640)
14. Joly, A.; Wehenkel, L.; Geurts, P. (2019). "Gradient tree boosting with random output projections for multi-label classification and multi-output regression." [arXiv:1905.07558](https://arxiv.org/abs/1905.07558)
15. Au, Q.; Schalk, D.; Casalicchio, G.; Schoedel, R. (2019). "Component-Wise Boosting of Targets for Multi-Output Prediction." [arXiv:1904.03943](https://arxiv.org/abs/1904.03943)
16. Xu, D.; Shi, Y.; Tsang, I. W.; Ong, Y.-S. (2019). "A Survey on Multi-output Learning." [arXiv:1901.00248](https://arxiv.org/abs/1901.00248)
17. Pliakos, P.; Geurts, P.; Vens, C. (2018). "Global multi-output decision trees for interaction prediction." *Machine Learning*. [https://doi.org/10.1007/s10994-018-5700-x](https://doi.org/10.1007/s10994-018-5700-x)
15. Ibragimov, S.; Gusev, G. (2019). "Minimal Variance Sampling in Stochastic Gradient Boosting." [arXiv:1910.13204](https://arxiv.org/abs/1910.13204)
16. Lu, J.; Karimireddy, S. P.; Ponomareva, E.; Mirrokni, V. (2019). "Accelerating Gradient Boosting Machine." [arXiv:1903.08708](https://arxiv.org/abs/1903.08708)
17. Biau, G.; Cadre, V.; Rouvière, S. (2019). "Accelerated Gradient Boosting." [arXiv:1803.02042](https://arxiv.org/abs/1803.02042)
18. Zhang, S.; Zhang, Y.; Fan, Z.; Luo, Y. (2022). "OpenFE: Automated Feature Generation with Expert-level Performance." [arXiv:2211.12507](https://arxiv.org/abs/2211.12507)
19. Iosipoi, A.; Vakhrushev, A. (2022). "SketchBoost: Fast Gradient Boosted Decision Tree for Multioutput Problems." [arXiv:2211.12858](https://arxiv.org/abs/2211.12858)
20. Ghosal, I.; Hooker, G. (2020). "Boosting random forests to reduce bias: one-step boosted forest and its variance estimate." *JCGS*.
21. XGBoost docs: [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) · [monotone tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html) · [source `src/gbm/gbtree.cc`](https://github.com/dmlc/xgboost/blob/master/src/gbm/gbtree.cc) · [`src/linear/coordinate_common.h`](https://github.com/dmlc/xgboost/blob/master/src/linear/coordinate_common.h) · [`src/tree/split_evaluator.h`](https://github.com/dmlc/xgboost/blob/master/src/tree/split_evaluator.h)
22. LightGBM: [parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) · [source `src/boosting/`](https://github.com/LightGBM/LightGBM/tree/master/src/boosting)
23. CatBoost: [parameters](https://catboost.ai/docs/en/features/parameters-main) · [repository](https://github.com/catboost/catboost)
24. scikit-learn: [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) · [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) · [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) · [ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html)
25. ChimeraBoost: [PyPI](https://pypi.org/pypi/chimeraboost) · [GitHub `bbstats/chimeraboost`](https://github.com/bbstats/chimeraboost) · [concepts](https://github.com/bbstats/chimeraboost/blob/main/docs/concepts.md) · [attribution](https://github.com/bbstats/chimeraboost/blob/main/docs/attribution.md)
26. Local benchmark artefacts: [`docs/benchmarks/elo_overall.md`](../docs/benchmarks/elo_overall.md) · [`docs/benchmarks/elo_board.md`](../docs/benchmarks/elo_board.md) · [`docs/benchmarks/elo_cla_board.md`](../docs/benchmarks/elo_cla_board.md)

**Unverified, do not assert:**

- Any paper titled "Fast Boosting with Outlier Detection" by Müller, Mentrup, or Guestrin. Searches across the NeurIPS 2015 accepted list, the conference book author lists, the workshop book, arXiv, Crossref, OpenAlex, Semantic Scholar, CORE, and DBLP found nothing.
- The XGBoost `gblinear` tutorial at `xgboost.readthedocs.io/.../tutorials/gblinear.html`. The page does not exist at any released version.
- A published monotone-EBM or "ExMonotone" paper from AISTATS 2024. No matching record found.
- Bühlmann & Yu's numeric rate claims. The full text is paywalled; only metadata is verified.
- The exact CatBoost release that introduced the GPU 50,000-object Ordered threshold.
- DBLP and Semantic Scholar records for "ChimeraBoost". Both were bot-challenged or rate-limited, so they are not evidence of absence.
