# Ensemble Families and API Architecture for `mltreelib`

**Date:** 2026-11-22
**Scope:** Bagging and its variants, the bagging-to-boosting boundary, voting, averaging, stacking, cross-validation machinery, Bayesian bootstrap, ordered bagging, stability selection, and the API precedent for a unified ensemble surface.
**Decision informed:** How `mltreelib` should split its ensemble capability across `ModelForest`, `ModelBoost`, and `ModelMeta`, and whether the member tree should be `ModelTree`.
**Confidence:** High for classical mechanics, API surfaces, and repository fit. Explicitly marked gaps where a cited paper could not be verified.
**Code changes:** None. Research and specification only.

## 1. Executive Summary

`mltreelib` needs an ensemble layer. The question is not which ensemble algorithm to implement. It is how to shape the surface so that the three statistically different families do not get confused with each other.

**The three families.**

| Family | What it changes | Combination rule | Members can differ? | Needs held-out data? |
|---|---|---|---|---|
| **Bagging** | The training data of each member | Uniform mean or vote | No | No |
| **Boosting** | The target each stage is fitted to | Sum of shrunk stages | No | No, but early stopping wants it |
| **Meta-estimators** | Neither | Fixed rule, or learned from out-of-fold predictions | **Yes** | **Yes**, for stacking |

Three findings drove the architecture.

**First, one umbrella class would be a mistake.** The families differ in what the user is choosing between: a variance-reduction method, a bias-reduction method, and a combination-rule learner. One class with a `kind` switch over all three would have to accept either an OOB flag or a learning rate or a CV splitter in the same constructor, and would have to answer "which family am I in" before it could validate the rest. The parameter surface explodes because `ModelTree` already exposes leaf kind, leaf family, leaf solver, two penalty terms, two monotone vectors, interaction groups, split criterion, split test, prune cost, categorical policy, honesty, alpha, imbalance penalty, col sample, and NA policy, and a forest multiplies each by a per-kind configuration.

**Second, `ModelTree` is the right member type, and the user's instinct was correct.** `mltreelib`'s tree is not CART with extra options. It carries linear leaves, survival leaves, causal leaves, monotone constraints on splits and leaf coefficients separately, categorical partition search, missing-value routing with per-node stored direction, thirty split criteria including `Sdr`, `Causal`, `LogRank`, `Mob`, `Ctree`, `MaxStat`, and `Hellinger`, rule extraction, SHAP, break-down, permutation importance, and a k-NN residual correction path. A forest of these members gets all of it. A forest of plain CART trees gets none of it, and no generic `Tree` interface would be narrow enough to avoid becoming `ModelTree` with extra indirection.

The sklearn inheritance structure supports this. The user-proposed chain `BaseDecisionTree -> DecisionTreeRegressor -> BaseForest -> RandomForestRegressor` is **not** sklearn's actual structure. sklearn has two parallel branches sharing `BaseEstimator`:

```text
BaseDecisionTree(MultiOutputMixin, BaseEstimator)
  DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree)
  DecisionTreeRegressor(RegressorMixin, BaseDecisionTree)
    ExtraTreeRegressor(DecisionTreeRegressor)

BaseEnsemble(MetaEstimatorMixin, BaseEstimator)
  BaseForest(MultiOutputMixin, BaseEnsemble)
    ForestRegressor(RegressorMixin, BaseForest)
      RandomForestRegressor(ForestRegressor)
      ExtraTreesRegressor(ForestRegressor)
```

The base tree does **not** become the ancestor of the forest. The forest is a *composer of fitted tree objects*. Forest and tree share estimator-level conventions, not concrete lineage. That is exactly right for `mltreelib`: `ModelForest` composes `ModelTree` members and never inherits from them.

**Third, the bagging-to-boosting boundary is precise, and the Cubist committee sits in an interesting place.** Bagging and boosting share the linear-ensemble framing but differ in data generation and coefficients. There is **no theorem** saying boosting is "bagging of residual space". Squared-loss boosting is functional gradient descent with pseudo-response equal to the residual, and for a linear smoother with hat matrix $S$, repeated residual fitting gives $\widehat F_m = \sum_{j=0}^{m} S(I-S)^j Y = (I - (I-S)^{m+1})Y$. That is exact boosting algebra, not bagging algebra. The closest formal bridge is iterated bagging, which Breiman explicitly says "has no connection to boosting".

The Cubist committee is sequential in *training* but averages in *combination*. That asymmetry is a real property of the method, and it places the committee in the bagging family by combination rule while looking sequential by construction.

## 2. Bagging: the exact algorithm

### 2.1 Breiman (1996)

Given a learning sample $L = \{(y_i, x_i) : i = 1, \ldots, N\}$ and a fixed learning procedure producing predictor $\phi(x, L)$:

1. For $b = 1, \ldots, B$, draw a bootstrap replicate $L^{(b)}$ of size $N$, with replacement from $L$.
2. Fit $\phi_b(x) = \phi(x, L^{(b)})$.
3. Numerical response: average predictions.

$$\phi_B(x) = \frac{1}{B} \sum_{b=1}^{B} \phi_b(x).$$

4. Classification: plurality vote.

$$N_j(x) = \sum_{b=1}^{B} \mathbf{1}\{\phi_b(x) = j\}, \qquad \phi_B(x) = \arg\max_j N_j(x).$$

**Implementation requirement that is easy to get wrong.** Every sampled index must be preserved *with its multiplicity*, and the multiplicity must be exposed. Duplicate rows change effective weights and enable out-of-bag residual estimation. Drawing unique indices silently turns bagging into subsampling.

### 2.2 The stability argument and its actual conditions

Let $\phi(x, L)$ be fitted on a random iid sample $L$, and define population aggregation $\phi_A(x) = E_L[\phi(x,L)]$. For squared error:

$$E[(Y - \phi_A(x))^2] \le E[(Y - \phi(x,L))^2],$$

with the exact decomposition

$$E[(Y - \phi_A(x))^2] = E[(Y - \phi(x,L))^2] - E_L[\operatorname{Var}_L(\phi(x,L) \mid x)].$$

Averaging removes part of the learning-set variance.

Breiman's "unstable predictor" condition: the base procedure must change materially when the learning set changes. Unstable examples are classification and regression trees, neural networks, and subset selection in linear regression. Stable examples include $k$-nearest neighbours. Bagging $k$-NN does not help.

The conditions are: iid observations; the same learning procedure for every replicate; size-$N$ bootstrap samples; uniform aggregation. Instability is the **mechanism**, not a finite-sample guarantee of improved test error.

### 2.3 Random forest's difference from bagging

Random forest adds random subsetting of candidate split variables at each node. That is the whole difference. Everything else in Breiman (2001) — bootstrap sampling, unpruned trees, averaging — is shared.

Sources: Breiman (1996) DOI 10.1023/A:1018054314350, [PDF](https://www.stat.berkeley.edu/~breiman/bagging.pdf); Bühlmann & Yu (2002), *Analyzing Bagging*, *Annals of Statistics* 30:927–961, DOI [10.1214/aos/1031689014](https://doi.org/10.1214/aos/1031689014).

## 3. The bagging-to-boosting boundary

### 3.1 Boosting is functional gradient descent

For empirical risk $\widehat C(F) = \frac{1}{n}\sum_i \rho(Y_i, F(X_i))$, boosting computes the negative gradient

$$u_i = -\left.\frac{\partial \rho(y,f)}{\partial f}\right|_{f = F_{m-1}(X_i)},$$

fits a base learner to the pseudo-responses $u_i$, and performs a line search $F_m = F_{m-1} + \hat s_m \hat g_m$.

Bagging instead fits on bootstrap samples and averages: $\hat g_{\text{Bag}} = \frac{1}{M}\sum_{k=1}^{M}\hat g_k^*$.

They share the general linear-ensemble framing but differ in data generation and coefficients.

### 3.2 Squared-loss boosting is repeated residual fitting

For $\rho(y,f) = \frac{1}{2}(y-f)^2$, the pseudo-response is the residual $y_i - F_{m-1}(X_i)$. If the base learner performs least squares on residuals, the line-search step is one:

$$F_m = F_{m-1} + \hat g_m, \qquad \hat g_m = \mathcal A(X, Y - F_{m-1}(X)).$$

For a linear smoother with hat matrix $S$, repeated residual fitting yields

$$\widehat F_m = \sum_{j=0}^{m} S(I-S)^j Y = \big(I - (I-S)^{m+1}\big)Y.$$

This is exact algebra for boosting's residual operator. It is **not** bagging algebra.

### 3.3 The closest bridge: iterated bagging

Breiman's iterated bagging (2001) is the nearest formal relative:

1. Current responses are $y_n^{(j)}$, initially $y_n$.
2. Draw $K$ bootstrap replicates from $(x_n, y_n^{(j)})$.
3. Fit predictor $f_{j,k}$ on each.
4. For each observation $n$, compute the out-of-bag mean:

$$\bar f_{n,j} = \operatorname{av}_{k:\,n \notin T_{j,k}} f_{j,k}(x_n).$$

5. Update responses: $y_n^{(j+1)} = y_n^{(j)} - \bar f_{n,j}$.
6. Update the ensemble additively: $f^{(j+1)}(x) = f^{(j)}(x) + \operatorname{av}_k f_{j,k}(x)$.

Stopping rule: compute the mean sum of squares of the current updated responses; stop when it exceeds $1.1$ times the minimum mean sum of squares from previous stages; return the predictor from the stage with the minimum mean sum of squares. Breiman reports $K = 50$ to $100$ and the multiplier $1.1$, noting $1.2$ is slightly worse and $1.05$ noticeably worse.

Iterated bagging differs from plain bagging because plain bagging averages one stage of bootstrap fits while iterated bagging changes response targets using out-of-bag predictions and then adds new stage predictors. It differs from boosting because of bootstrap sampling versus adaptive reweighting, out-of-bag residual estimates versus loss-gradient pseudo-residuals, independent member fitting versus a constrained sequential hypothesis space, and Breiman's explicit statement that it "has no connection to boosting".

**The actual formal hybrid** is bag-boosting, where a bagged estimator serves as boosting's base procedure. That is composition, not equality.

Source: Breiman (2001), "Using Iterated Bagging to Debias Regressions", *Machine Learning* 45:261–277, DOI [10.1023/A:1017934522171](https://doi.org/10.1023/A:1017934522171), [PDF](https://link.springer.com/content/pdf/10.1023/A:1017934522171.pdf).

## 4. Random-subspace methods and rotation forests

### 4.1 Ho random subspaces

Ho's random decision forest and random subspace method select a random subspace of features at each split, with a voting scheme across subspaces. Failure modes: too few selected features can make formerly separable classes ambiguous; subspace size strongly affects accuracy; if the chosen features cannot distinguish classes, multiple subspaces do not help; diversity depends on feature redundancy and subspace geometry.

Sources: Ho (1995) DOI [10.1109/ICDAR.1995.598994](https://doi.org/10.1109/ICDAR.1995.598994); Ho (1998) DOI [10.1109/34.709601](https://doi.org/10.1109/34.709601).

### 4.2 Rotation forest

**Citation correction.** The authors are **Juan J. Rodríguez, Ludmila I. Kuncheva, Carlos J. Alonso**, not Rodríguez, Grinblat, Yao. *IEEE TPAMI* 28(10):1619–1630, 2006, DOI [10.1109/TPAMI.2006.211](https://doi.org/10.1109/TPAMI.2006.211).

Per ensemble member $D_i$:

1. Randomly partition the feature set $F$ into $K$ subsets. The paper favours disjoint subsets.
2. For each subset $F_{i,j}$: randomly select a nonempty subset of classes; draw an object sample of size 75% of the data; run PCA using only $F_{i,j}$ on that selected sample and class subset; retain **all** available principal components, including zero-eigenvalue omissions.
3. Assemble component coefficient vectors into a sparse rotation matrix $R_i$.
4. Reorder columns to original feature positions, producing $R_i^a$.
5. Transform the complete training data: $XR_i^a$.
6. Fit the base classifier on the complete transformed data.
7. Repeat for all members; aggregate predictions.

All principal components are retained, so PCA acts as a rotation and diversification heuristic rather than dimension reduction. Rotation changes tree splits because trees are sensitive to axis orientation.

**Spec implication.** Keep transform-training separate from base-learner fitting. At prediction, apply the member-specific rotation matrix before the member predictor.

## 5. Double bagging and iterated bagging

### 5.1 Citation corrections

- *Analyzing bagging* is Bühlmann & Yu, **2002**, not 2014. DOI [10.1214/aos/1031689014](https://doi.org/10.1214/aos/1031689014).
- "Prediction with bagging and bagging of predictors" attributed to Bühlmann–Yu 2014 is **unverified**. The verified related publication is Bühlmann, "Bagging, Boosting and Ensemble Methods," *Handbook of Computational Statistics*, [manuscript](https://stat.ethz.ch/Manuscripts/buhlmann/handbook-cs-rev2010.pdf).
- Iterated bagging is Breiman, **2001**, DOI [10.1023/A:1017934522171](https://doi.org/10.1023/A:1017934522171).

### 5.2 Hothorn–Lausen double bagging

Per replicate:

1. Draw a bootstrap sample $L_n^*$ of size $n$, with replacement.
2. Define the out-of-bag set $L_n \setminus L_n^*$.
3. Train auxiliary classifiers $C_1, \ldots, C_K$ **only on out-of-bag observations**.
4. Apply each auxiliary classifier to in-bag observations.
5. Form extra predictors from estimated conditional class probabilities, LDA discriminant variables, or predicted classes.
6. Fit a classification tree on the original predictors plus the auxiliary predictions.
7. Repeat $B$ times.
8. Aggregate trees by majority vote.

At prediction, apply the same replicate's out-of-bag-trained auxiliary model to the new observation, then its tree.

This "double" means two classifier families are bundled: auxiliary classifier plus tree. It is **not** bootstrap-residual correction, **not** iterated bagging, and **not** ordinary stacking.

Sources: Hothorn & Lausen (2003) DOI [10.1016/S0031-3203(02)00169-3](https://doi.org/10.1016/S0031-3203(02)00169-3); journal version (2005) DOI [10.1016/j.csda.2004.06.019](https://doi.org/10.1016/j.csda.2004.06.019); Hothorn, "Bundling Predictors in R", [DSC 2003 proceedings](https://www.r-project.org/conferences/DSC-2003/Proceedings/Hothorn.pdf).

### 5.3 Ordered bagging

**Requested citation unverified.** No authoritative record was found for "Therneau, Ipelweich, Bachelin (2000), *Improvements to bagging for data analysis*". Likely a conflation among Grandvalet's *Bagging Equalizes Influence* and the tree-bagging literature. Do not cite.

The verified ordered-bagging work is Martínez-Muñoz & Suárez (2004), *Pruning in Ordered Bagging Ensembles*, ICML, DOI [10.1145/1143844.1143921](https://doi.org/10.1145/1143844.1143921).

Mechanics:

1. Train a bagged CART ensemble.
2. Replace the random member order with an order chosen by a greedy criterion.
3. At each position, add the member maximising the selected criterion.
4. Retain the prefix of the desired size.
5. Aggregate the prefix by majority vote.

Published criteria: **Reduce-Error** (the member minimising partial-ensemble error on the selection set), **Complementariness** (the member that correctly handles cases the current partial ensemble misclassifies), and **Margin Distance Minimization** (the member moving the mean correctness-margin vector toward a target point $o = (p, \ldots, p)$ with $p \in (0,1)$; the paper explores $p \approx 0.05$ to $0.25$).

Key result: ordering by individual resubstitution error or individual validation accuracy did not yield useful subsets. Complementariness matters more than standalone accuracy.

Ordering alone cannot change full-ensemble prediction. The benefit requires pruning or changing the aggregation prefix.

## 6. Stability selection, and why it is not double bagging

These share subsampling vocabulary but solve different problems.

- **Double bagging:** Hothorn & Lausen (2003), DOI [10.1016/S0031-3203(02)00169-3](https://doi.org/10.1016/S0031-3203(02)00169-3).
- **Stability selection:** Meinshausen & Bühlmann (2010), DOI [10.1111/j.1467-9868.2010.00740.x](https://doi.org/10.1111/j.1467-9868.2010.00740.x).

Let the selection procedure return a selected set $\hat S_\lambda(I)$ at regularization value $\lambda$.

1. Draw $B$ independent random subsamples $I_b$, **without replacement**, each of size $|I_b| = \lfloor n/2 \rfloor$.
2. For every $\lambda \in \Lambda$, run selection on each $I_b$.
3. Estimate the selection probability: $\hat\pi_k(\lambda) = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\{k \in \hat S_\lambda(I_b)\}$.
4. Define the stable set: $\hat S_{\text{stable}} = \{k : \max_{\lambda \in \Lambda} \hat\pi_k(\lambda) \ge \pi_{\text{thr}}\}$, with $0 < \pi_{\text{thr}} < 1$.
5. Practical threshold range approximately $0.6$ to $0.9$; $0.9$ is common.

The error theorem, under exchangeability of the selection indicators and the assumption that the base selection is "not worse than random guessing":

$$E(V) \le \frac{q_\Lambda^2}{p(2\pi_{\text{thr}} - 1)},$$

where $V$ is the number falsely selected and $q_\Lambda$ is the average number selected per subsample. With $\pi_{\text{thr}} = 0.9$, choosing $q_\Lambda = \sqrt{0.8p}$ controls $E(V) \le 1$; choosing $q_\Lambda = \sqrt{0.8\alpha p}$ controls the familywise error $P(V > 0) \le \alpha$.

Failure modes: exchangeability is strong and may fail in real data; the original paper used half-samples because they approximate bootstrap distance behaviour while being cheaper; small $B$ makes selection-frequency estimates noisy; using the same data to choose $\lambda$, estimate frequencies, and assess the final selection creates optimism; stability selection changes variables and does not automatically produce better final predictions.

## 7. Voting, averaging, and stacking

### 7.1 Voting

- **Hard voting:** count predicted labels.
- **Weighted hard voting:** count $w_m \mathbf{1}\{\hat y_m = c\}$.
- **Soft voting:** sum or mean predicted class probabilities, optionally weighted, then argmax.

Soft voting requires every estimator to support `predict_proba`. Calibration matters because probability scales must be comparable.

Failure modes: correlated members produce the same errors and majority cannot repair them; minority-class members may be drowned out; hard voting discards confidence; soft voting can be harmed by poor calibration; fixed weights are not learned from held-out data.

### 7.2 Averaging

$$\hat y(x) = \sum_m w_m \hat f_m(x), \qquad w_m \ge 0, \quad \sum_m w_m = 1.$$

Uniform weights are simplest. For classification, averaging over class probabilities is the soft-voting rule. Averaging is a variance-reduction combinator when member errors are sufficiently diverse.

### 7.3 Stacking

Wolpert's two-level formulation:

- Level 0: the original observations and base generalizers.
- Partition the learning set into training and held-out parts.
- Train base generalizers on the training part; predict the held-out part.
- Construct level-1 observations from those held-out predictions, optionally with raw inputs.
- Train the level-1 generalizer on the reduced level-1 set.
- Transform a new query through the base level-0 generalizers, then the level-1 generalizer.

Modern $K$-fold algorithm:

1. Partition rows into $K$ folds.
2. For each base learner $m$, for each fold $k$: fit a clone on the other $K-1$ folds, predict fold $k$.
3. Concatenate the out-of-fold predictions into the meta-feature matrix $Z$.
4. Fit the final estimator on $(Z, y)$.
5. For inference: fit base learners on the full training data, generate $Z_{\text{test}}$, apply the final estimator.

Out-of-fold predictions are essential. In-sample base predictions are optimistically biased, especially for flexible models, and the meta-learner learns misleading reliability.

Breiman's *Stacked Regressions* uses cross-validation data to fit a linear combination, usually with non-negative least-squares coefficients. Non-negativity regularises strongly correlated predictors.

Failure modes: meta-features must be out-of-fold; the meta-learner can overfit when base predictions are highly correlated or the sample is small; base models refit on full data create train/test distribution shift relative to the out-of-fold meta-features; tuning the meta-learner on the same CV predictions requires nested CV; multi-level stacking magnifies the issue.

### 7.4 The four combinators at a glance

| Combinator | Combines | Fixed or learned | Requires held-out data | Members must share a family? |
|---|---|---|---|---|
| Voting | Labels or probabilities | Fixed weights | No | No |
| Averaging | Measurements or probabilities | Fixed weights | No | No |
| Stacking | Out-of-fold predictions | Learned | Yes | No |
| Bagging | Trained members | Uniform | No | Yes, one base type |

Sources: Wolpert (1992) DOI [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1), [PDF](https://machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf); Breiman (1996), "Stacked Regressions", DOI [10.1007/BF00117832](https://doi.org/10.1007/BF00117832), [PDF](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf); Kuncheva, *Combining Pattern Classifiers*, DOI [10.1002/0471660264](https://doi.org/10.1002/0471660264).

## 8. Cross-validation machinery

### 8.1 $K$-fold

Partition $n$ observations into $K$ folds of approximately equal size. For each fold: fit on the other $K-1$ folds, predict the held-out fold, accumulate the held-out loss. Each training row receives one out-of-fold prediction.

### 8.2 Leave-one-out

The $K = n$ special case: train on $n-1$ rows, predict one row, repeat $n$ times. Uses nearly all data per fit, but is expensive unless the learner supports efficient deletion or update. The risk estimate is nearly unbiased but has high variance. Folds are strongly dependent, so uncertainty estimates cannot assume independent folds.

### 8.3 Monte Carlo cross-validation

Repeated random train/test splits: randomly choose the test fraction and test observations, train on the complement, predict the test set, repeat $R$ times, average the losses or aggregate the predictions. Kohavi distinguishes "complete $K$-fold", the average over all possible partitions, from one observed partition. Repeated random splits provide a Monte Carlo estimate of the complete-$K$-fold result. Variants: shuffle each repetition, stratify classification folds, keep the test set fixed versus redraw it, use repeated splits for uncertainty rather than only a point estimate.

### 8.4 Feeding stacking

$K$-fold yields one out-of-fold prediction per row with $K$ fits per base learner. Leave-one-out yields one per row with $n$ fits per base learner. Repeated Monte Carlo yields multiple per row, and the meta-training step must define how repeated predictions are reduced, often by averaging, while preserving a separate validation split to avoid tuning leakage.

**Every base learner must use identical folds** for comparable meta-feature columns. scikit-learn achieves this by copying the same CV splitter for each `cross_val_predict` call. This is the single most common silent failure in a hand-rolled stack: if member $m$ uses fold assignment $A$ and member $m'$ uses assignment $B$, the meta-feature matrix pairs row $i$'s prediction from $m$ with row $j$'s prediction from $m'$.

Sources: Arlot & Celisse (2010) DOI [10.1214/09-SS054](https://doi.org/10.1214/09-SS054); Kohavi (1995), [IJCAI PDF](https://ijcai.org/proceedings/1995/0147.pdf).

## 9. Bayesian bootstrap

Ordinary bootstrap equivalent weights:

$$C = (C_1, \ldots, C_n) \sim \operatorname{Multinomial}\left(n; \tfrac{1}{n}, \ldots, \tfrac{1}{n}\right), \qquad p_i = \frac{C_i}{n}.$$

Bayesian bootstrap:

$$(p_1, \ldots, p_n) \sim \operatorname{Dirichlet}(1, \ldots, 1), \qquad p_i \ge 0, \quad \sum_i p_i = 1.$$

Operational differences:

| | Ordinary bootstrap | Bayesian bootstrap |
|---|---|---|
| Weight values | Discrete multiples of $1/n$ | Continuous |
| Can be zero | Yes | No, normally all positive |
| Duplicate rows | Yes | No, every row gets fractional weight |
| Fit mechanism | Duplicated rows | Weights $p_i^{(b)}$ |
| Weight variance | $(n+1)/n$ times Bayesian | Lower |

Both have $E[p_i] = 1/n$ and the same pairwise weight correlation. For functionals linear in weights, the Bayesian bootstrap has strictly lower variance.

For regression, $\hat\beta^{(b)} = (X^\top W^{(b)} X)^{-1} X^\top W^{(b)} y$ with $W^{(b)} = \operatorname{diag}(p^{(b)})$, then average $\hat y^{(b)} = X\hat\beta^{(b)}$.

Failure mode: many tree and linear-model APIs do not expose fractional observation weights. `mltreelib` does through `fit_weighted`, so the Bayesian bootstrap is reachable once the sampler supports it.

Sources: Rubin (1981) DOI [10.1214/aos/1176345338](https://doi.org/10.1214/aos/1176345338); Clyde & Lee (2001), [NIPS 13 PDF](https://proceedings.mlr.press/r3/clyde01a/clyde01a.pdf).

## 10. R ecosystems: `caret` and `baguette`

### 10.1 `caret::bag`

```r
bag(x, y, B = 10, vars = ncol(x), bagControl = NULL, ...)

bagControl(
  fit = NULL,
  predict = NULL,
  aggregate = NULL,
  downSample = FALSE,
  oob = TRUE,
  allowParallel = TRUE
)
```

Exact behaviour:

- `B`: number of bootstrap samples.
- `vars`: random predictor count per iteration; `NULL` uses all. Predictor subsetting occurs **before** the user's `fit` and `predict` functions.
- `fit(x, y, ...)`: fit a member.
- `predict(object, x)`: member prediction.
- `aggregate(x, type)`: reduce member outputs.
- `oob`: compute out-of-bag statistics and retain predictions.
- `downSample`: for classification, downsample classes inside each bag to the smallest class count.
- `allowParallel`: use the registered `foreach` backend.

`bagEarth()` fits one MARS/`earth` model per bootstrap sample and aggregates with a supplied `summary`, default mean, with `keepX = TRUE` retaining the original data.

`train()` methods include `method = "bag"` with tuning parameter `vars`, `method = "bagEarth"` with `nprune` and `degree`, and `method = "bagEarthGCV"` with `degree`. There is **no single `method = "bagged"` generic workflow**; the `bagged*` names identify specific registered models.

Source: [`caret::bag` reference](https://search.r-project.org/CRAN/refmans/caret/html/bag.html), [source](https://github.com/cran/caret/blob/master/R/bag.R).

### 10.2 `baguette`

**There is no `fit_bag()` in `baguette`.** The public API is `bagger()`, `control_bag()`, `class_cost()`, `predict()`, `var_imp()`.

```r
bagger(
  x, y,
  weights = NULL,
  base_model = "CART",
  times = 11L,
  control = control_bag(),
  cost = NULL,
  ...
)
```

Exact behaviour:

- Fits separate models to bootstrap samples.
- `times` is the maximum ensemble size; some fits may fail.
- Base models: `"CART"`, `"MARS"`, `"nnet"`, `"C5.0"` classification only.
- Case weights are **not used during bootstrap resampling**.
- Stores each member's prediction function as an R expression and discards the original model object.
- Evaluates the member prediction expressions and aggregates using the mean.
- Supports parallel fitting through `future`; memory use may grow sharply.
- `control_bag(sampling = "down")` can downsample each class within each bootstrap sample; default `"none"`.

The parsnip workflow uses ordinary `fit()`:

```r
bag_tree() |>
  set_engine("rpart", times = 25) |>
  set_mode("regression") |>
  fit(mpg ~ ., data = mtcars)
```

Supported specifications: `bag_tree()` with `rpart` or `C5.0`, `bag_mars()` with `earth`, `bag_mlp()` with `nnet`.

Sources: [`baguette` index](https://baguette.tidymodels.org/reference/index.html), [`bagger()`](https://baguette.tidymodels.org/reference/bagger.html), [`control_bag()`](https://baguette.tidymodels.org/reference/control_bag.html).

## 11. scikit-learn exact semantics

### 11.1 `BaggingClassifier` and `BaggingRegressor`

```python
BaggingClassifier(
    estimator=None,
    n_estimators=10,
    *,
    max_samples=None,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=False,
    warm_start=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
)
```

**Naming correction.** `base_estimator` was renamed to `estimator` in scikit-learn 1.2. The fitted `base_estimator_` became `estimator_`. Older API documentation asked for `base_estimator`.

Parameter semantics:

- `estimator=None`: default `DecisionTreeClassifier` or `DecisionTreeRegressor`.
- `n_estimators=10`: member count.
- `max_samples=None`: draw `X.shape[0]` samples irrespective of `sample_weight`.
- `max_samples=int`: exact sample count.
- `max_samples=float`: `max_samples * n_samples` unweighted, or `max_samples * sum(sample_weight)` weighted.
- `max_features=1.0`: all features.
- `max_features=int`: exact feature count.
- `max_features=float`: `max(1, int(max_features * n_features_in_))`.
- `bootstrap=True`: sample rows with replacement.
- `bootstrap=False`: sample rows without replacement, known as **pasting**.
- `bootstrap_features=False`: sample features without replacement.
- `bootstrap_features=True`: sample features with replacement. Feature subset plus sample subset gives random patches.
- `random_state`: controls row and feature sampling; each member receives a derived seed. Base estimators accepting `random_state` get distinct seeds.
- `n_jobs`: parallel fitting and prediction; `-1` uses all processors.

Aggregation: regressor uses the arithmetic mean of member predictions; a classifier with `predict_proba` averages aligned class probabilities; a classifier without `predict_proba` votes over predicted labels; `decision_function` averages member decision functions.

Out-of-bag semantics: `oob_score=True` estimates the training-set score from predictions made by members that did not contain each row. It requires `bootstrap=True`. The implementation rejects `warm_start=True` with out-of-bag enabled. A classifier exposes `oob_decision_function_`; a regressor exposes `oob_prediction_`. Small member counts can leave a row never out-of-bag, producing `NaN`.

`warm_start=False` fits a new ensemble from scratch. `warm_start=True` reuses existing members and adds more only when the requested `n_estimators` exceeds the current count. Increasing `n_estimators` after fitting with `warm_start=True` grows the ensemble; decreasing the count does not truncate the existing fitted ensemble.

Source: official scikit-learn reference pages and [source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_bagging.py).

### 11.2 `VotingClassifier` and `VotingRegressor`

```python
VotingClassifier(
    estimators,
    *,
    voting="hard",
    weights=None,
    n_jobs=None,
    flatten_transform=True,
    verbose=False,
)

VotingRegressor(
    estimators,
    *,
    weights=None,
    n_jobs=None,
    verbose=False,
)
```

- `estimators`: list of `(name, estimator)` tuples.
- Members are cloned and fitted on the same full dataset.
- `"drop"` removes a member.
- `"hard"`: plurality over predicted labels.
- `"soft"`: argmax over the weighted sum of predicted probabilities; requires `predict_proba`.
- `weights`: per-member weights; uniform when `None`.
- `flatten_transform` changes only the `transform()` shape: soft and flattened gives `(n_samples, n_classifiers * n_classes)`; soft and unflattened gives `(n_classifiers, n_samples, n_classes)`; hard gives `(n_samples, n_classifiers)`. It does not change `predict()`.

### 11.3 `StackingClassifier` and `StackingRegressor`

```python
StackingClassifier(
    estimators,
    final_estimator=None,
    *,
    cv=None,
    stack_method="auto",
    n_jobs=None,
    passthrough=False,
    verbose=0,
)
```

- Base estimators are fit on the full `X` for inference.
- The final estimator trains on `cross_val_predict` outputs from the base estimators.
- `cv=None`: default 5-fold. Integer `cv`: number of folds.
- Classifier with binary or multiclass labels uses `StratifiedKFold`; regressor uses `KFold`.
- Default splitters use `shuffle=False`. A custom CV generator or an iterable of `(train, test)` is accepted.
- `"prefit"`: assume the base estimators are already fitted; the meta-features become in-sample predictions. scikit-learn warns of very high overfitting risk. Support added in scikit-learn 1.1.
- `stack_method="auto"`: per estimator try `predict_proba`, then `decision_function`, then `predict`. An explicit method errors if the estimator lacks it.
- `passthrough=False`: the final estimator sees only the base predictions. `passthrough=True`: it sees the concatenated predictions and the original `X`.
- Default final classifier: `LogisticRegression`. Default final regressor: `RidgeCV`.
- Binary `predict_proba` stacking drops one probability column per estimator, because probabilities sum to one and the columns are perfectly collinear.
- `cv` exists to generate meta-predictions, **not** to evaluate model performance.
- All base learners use copies of the same CV splitter, ensuring aligned fold assignment.

Source: official reference pages and [source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_stacking.py).

### 11.4 The meta-estimator list contract

For `VotingRegressor`, `VotingClassifier`, `StackingRegressor`, and `StackingClassifier`:

1. Non-empty list of `(name, estimator)` tuples.
2. Names must be unique.
3. Names cannot collide with wrapper constructor parameters.
4. Names cannot contain `__`.
5. Normal `fit` clones the supplied estimator instances before fitting.
6. `'drop'` removes a member from the active ensemble.
7. `estimators_` contains the fitted active members, excluding dropped members.
8. `named_estimators_` is a fitted-estimator `Bunch`, keyed by the supplied name.
9. `get_params(deep=True)` exposes nested parameters under `name__parameter`.
10. `set_params(name__parameter=value)` tunes a member; `set_params(name=other_estimator)` replaces it; `set_params(name='drop')` drops it.

Current source nuance: dropped entries remain visible in `named_estimators_` as `"drop"` placeholders in voting and stacking, while `estimators_` excludes them. `mltreelib` should keep two explicit lists instead, so a fitted member list never contains a string.

`MetaEstimatorMixin` is now an empty marker. It says the estimator contains other estimators; it does not implement parameter forwarding.

Sources: [ensemble base source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_base.py), [composition source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/metaestimators.py), [MetaEstimatorMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.MetaEstimatorMixin.html).

## 12. API architecture: the precedent

### 12.1 Precedent table

| Library | Base tree | Ensemble surface | Pattern |
|---|---|---|---|
| scikit-learn | `BaseDecisionTree`, `DecisionTreeClassifier`, `DecisionTreeRegressor` | `RandomForestClassifier`, `GradientBoostingClassifier`, `BaggingClassifier`, `VotingClassifier`, `StackingClassifier` | Algorithm-specific classes; heterogeneous meta-estimators use a named list |
| XGBoost | No public per-leaf tree base class | `Booster`; `booster = "gbtree"`, `"gblinear"`, `"dart"` | One model object with a kind enum |
| LightGBM | Internal tree learner | `LGBMClassifier`, `LGBMRegressor`; `boosting_type = "gbdt"`, `"dart"`, `"rf"` | One estimator with a kind enum |
| Julia MLJ | `DecisionTreeClassifier` from DecisionTree.jl | Generic `EnsembleModel(model = tree, ...)` | Generic wrapper around a selected atomic model |
| R `rpart` | `rpart` object | No forest wrapper | Algorithm-specific object |
| R `ranger` | Internal tree | `ranger()` | Algorithm-specific family; `splitrule` enum |
| R `randomForest` | Internal tree | `randomForest()` | Algorithm-specific family |
| R `gbm` | Internal trees | `gbm()` | Algorithm-specific family |
| R `party` / `partykit` | `ctree()` | `cforest()` | Forest fixed around conditional-inference trees |
| tidymodels `parsnip` | `decision_tree()` spec | `rand_forest()`, `boost_tree()`, `bag_tree()` | Model specs separate from engines |
| R `caret` | Package-specific | `train(method = "rf" \| "gbm" \| "bag")` | String-registry dispatch |
| CatBoost | Internal trees | `CatBoostClassifier`; `boosting_type` option | Algorithm-specific estimator with an internal mode enum |
| Rust `sklears-tree` | `ModelTree` | Ordinary RF uses a separate tree implementation | Confirms the `ModelTree` name for an M5-style tree |

**No established library uses one public class named literally `Ensemble` to cover every ensemble algorithm.** scikit-learn publishes algorithm-specific classes and has no generic public `Ensemble` class. XGBoost uses `Booster`, but it represents XGBoost internals. Julia MLJ comes closest with a generic `EnsembleModel`, but it wraps clones of one atomic model. tidymodels uses separate model specifications plus pluggable engines. `caret` uses one training entry point backed by a string-dispatch registry.

### 12.2 Rich base tree precedent

**scikit-learn** does not support model trees. The upstream model-tree request remains open and the proposed M5P pull request was closed. sklearn therefore has no native per-leaf linear model or arbitrary leaf estimator.

**XGBoost's `gblinear`** fits one global linear function. It does not fit linear models in leaves.

**LightGBM's `linear_tree = true`** changes the leaf value from constant to linear model: usual tree split selection remains; each leaf fits a linear model over the numerical features in that branch; the first tree uses constant leaf values; categorical features may split but do not enter leaf linear models; only CPU or GPU with the serial tree learner; `regression_l1` unsupported; memory use rises significantly; monotonic constraints guide split selection, not the fitted leaf linear models.

**ranger** does not expose a selected tree class. `treetype` appears in the fitted object and records the classification, regression, or survival family. `splitrule` selects the internal criterion.

**partykit**'s `cforest()` uses conditional-inference trees as fixed base learners, configured through `ctree_control()`. No user-selected arbitrary tree class.

**grf** does not expose a user-selected tree class either. It exposes an internal C++ strategy architecture:

```text
ForestTrainer
  RelabelingStrategy
  SplittingRuleFactory -> SplittingRule
  OptimizedPredictionStrategy

ForestPredictor
  DefaultPredictionStrategy
  or OptimizedPredictionStrategy
```

`RelabelingStrategy` has `relabel(samples, data, responses_by_sample) -> bool`, where the boolean requests an early stop, and `get_response_length()` supporting scalar or vector outcomes. `SplittingRule` has `find_best_split(...) -> bool`, where the boolean means no valid split was found, and it chooses the split variable, split value, and missing-value direction. `DefaultPredictionStrategy` predicts from weighted training-sample neighbours sharing the test leaf and supports variance estimates through grouped trees. `OptimizedPredictionStrategy` precomputes leaf summary values during training and computes prediction, variance, out-of-bag debiased error, and Monte Carlo error.

**The `grf` plugin architecture is the direct precedent for the deferred GRF splitting work.** It shows the seam a Rust implementation would need: a node-level relabeling strategy, a splitting rule, and a prediction strategy. `mltreelib`'s current `GainFn::Custom` cannot express GRF's split because `Custom` receives only the left and right `(y, w)` slices, while GRF needs the parent-level moment solve to derive the pseudo-outcome.

**Julia MLJ** is the strongest selected-base-model precedent: `EnsembleModel(model = tree, n = 300, bagging_fraction = 0.8)` clones a selected atomic model.

### 12.3 Kind switch versus meta-estimator list

**Kind switch** appears in LightGBM (`boosting_type`), XGBoost (`booster`), CatBoost (`boosting_type`), and ranger (`splitrule`, though narrower: an internal split criterion rather than a whole model family).

Strengths: one fit and predict lifecycle; one report contract; shared seed, parallelism, persistence, and diagnostics machinery; efficient Rust enum dispatch; users compare families without changing workflow code; good for related algorithms sharing a base learner and member representation.

Costs: invalid parameter combinations become possible; capability sets differ sharply; one flat constructor can accumulate irrelevant knobs; diagnostics need conditional sections; algorithm-specific deep behaviour can exceed the common facade.

Mitigations: typed family config enums rather than one giant flat config; a Python facade accepting `kind` plus validated family config; rejection of irrelevant options at construction; capability traits gating operations; a report that identifies family, objective, aggregation, and missing capability.

**Meta-estimator list** appears in sklearn voting and stacking, not in the concrete RF and GBDT classes.

Strengths: natural heterogeneous members; each member keeps its own algorithm-specific API; named access, per-member tuning, and inspection; supports voting, stacking, and arbitrary prediction aggregation.

Costs: members must share a weak contract of fit, predict, and often a probability method; cross-validation, sample weights, metadata routing, and missing-value behaviour need agreement; forest-specific diagnostics such as leaf paths, out-of-bag membership, forest weights, and training samples do not generalise; GRF-like members cannot work through a generic prediction interface because their contribution is topology and weights, not standalone prediction; generic stacking needs the full estimator lifecycle, cloning, persistence, parameter routing, and prefit semantics.

**Which pattern for `mltreelib`.** The official APIs do not support a quantified frequency claim. The qualitative pattern is: generic heterogeneous ensembles use the named-list pattern; same-engine algorithm families use a kind enum; broad scientific libraries often use separate public classes plus internal shared machinery. `mltreelib`'s current scope is same-engine tree families, not generic heterogeneous ensembles.

The decision: **kind switch within each family, separate public class per family, named-list meta-estimator surface as its own class.** That is exactly the three-class split.

### 12.4 Naming landscape

| Pattern | Examples | Meaning |
|---|---|---|
| Vendor prefix plus task | `XGBClassifier`, `XGBRegressor` | Vendor-owned estimator |
| Vendor prefix plus algorithm | `LGBMClassifier`, `CatBoostClassifier` | Vendor-owned estimator family |
| Algorithm plus task | `RandomForestClassifier`, `GradientBoostingClassifier` | Algorithm family, explicit task |
| Combination plus task | `BaggingClassifier`, `VotingClassifier`, `StackingClassifier` | Explicit combination rule |
| Generic wrapper | Julia `EnsembleModel` | Wrapper around an atomic model |
| Generic training facade | R `caret::train(method = ...)` | Registry-dispatch facade |
| Generic model spec | tidymodels `decision_tree()`, `rand_forest()`, `boost_tree()` | Spec plus engine |

No surveyed established library uses a generic `Model` prefix as a public estimator family name. `Model` often means a fitted internal representation rather than a public facade. That is a mild caution against the `Model` prefix in general, but `mltreelib` already uses it: `ModelTree` is the public name of the tree class, and the Rust ecosystem has precedent for the same name in `sklears-tree`'s `ModelTree` module.

### 12.5 Report and diagnostics conventions

No library has one universal ensemble report.

**scikit-learn** uses trailing underscores for fitted attributes: `estimators_`, `estimator_`, `feature_importances_`, `oob_score_`, `oob_prediction_`, `estimators_samples_`, `estimators_features_`, `n_features_in_`, `feature_names_in_`.

**Julia MLJ** separates `fitted_params(mach)` from `report(mach)`, and guards accessors with traits such as `supports_training_losses` and `reports_feature_importances`. Composite models expose named report sections.

**R `caret`** separates `results`, `bestTune`, `finalModel`, `resample`, `control`, and `times`.

**XGBoost, LightGBM, and CatBoost** expose the raw booster, evaluation results, the best iteration, best score, and feature importances.

`mltreelib` already has a report contract from Spec 25. The ensemble families extend it with family-specific sections rather than inventing a new universal schema.

## 13. The Cubist committee's family assignment

The committee trains members sequentially but combines them by an unweighted average. That asymmetry is worth stating as a finding rather than a convenience.

**Training.** Member $m \ge 2$ is trained on the relabelled target $y' = 2y - \hat y_{m-1}$, where $\hat y_{m-1}$ is the previous member's prediction on the training rows.

**Combination.** The final prediction is the simple average over members, with no stage weights, no learning rate, and no row resampling:

$$\hat y(x) = \frac{1}{M}\sum_{m=1}^{M} \hat y^{(m)}(x).$$

**Why this is not boosting.** Under equal averaging with target $2y - \hat y$, member $m+1$ receives weight $1/2$ implicitly, so the effective correction rate is $1/2$ with no decay across members. It is closest to a two-point Newton correction of the ensemble mean. That is why Cubist recommends about 5 members, why the benefit appears only when the base model is already reasonably accurate, and why the gain saturates rather than shrinking monotonically with $M$.

**Family assignment.** By combination rule, the committee is a bagging-family member. By training procedure, it looks like a booster. The two facts sit in different columns of the same table, and both are true. The family follows the combination rule because the combination rule determines the error characteristics and the aggregation machinery.

**Where the Cubist mechanics come from.** The Cubist GPL Edition C source, release 2.07, was read directly for the committee, smoothing, clamping, and composite mechanics. The CRAN `Cubist` vignette states the same averaging rule and contrasts it with boosting: "Unlike traditional boosting, stage weights for each committee are not used to average the predictions from each model tree; the final prediction is a simple average of the predictions from each model tree." Full details are in `references/research/2026-11-22-forest-methods-landscape.md`.

## 14. Verdicts

### BUILD_NOW

1. **Generic bootstrap aggregation.** Preserve sampled indices and multiplicities; uniform numeric averaging; hard class voting and probability averaging; optional weighted combiners.
2. **Out-of-bag infrastructure.** Store member sample indices; compute per-row out-of-bag member counts; out-of-bag numeric and class predictions and score; explicit missing-out-of-bag state rather than a silent zero.
3. **Combinators.** Average, hard voting, soft voting, fixed weights.
4. **Cross-validation split generator.** $K$-fold, leave-one-out, Monte Carlo splits, stratification and shuffle controls, stable fold identities so stacking columns stay aligned.
5. **scikit-learn-compatible semantics for the parts that are genuinely shared.** `estimator`, `n_estimators`, `max_samples`, `max_features`, `bootstrap`, `bootstrap_features`, `oob_score`, `warm_start`; `VotingClassifier` and `VotingRegressor`; the stacking API reuses the same split generator.
6. **Three separate public classes.** `ModelForest`, `ModelBoost`, `ModelMeta`, over one shared Rust `ensemble/` module.
7. **`ModelTree` as the member type.** `ModelForest` holds `Vec<ModelTree>`; every leaf family, monotone constraint, categorical policy, missing-value rule, rule set, and attribution reaches the forest.
8. **Common report contract with family-specific sections.**

### BUILD_LATER

1. **Stacking**, once the fold generator exists.
2. **Random-subspace ensemble.** Per-member feature mask, fit and predict through the mask.
3. **Rotation forest.** Per-member feature partitions, per-subset PCA, sparse rotation matrix and transform.
4. **Iterated bagging.** Out-of-bag pseudo-targets, stage accumulation, Breiman's stopping rule.
5. **Double bagging.** Out-of-bag auxiliary model, auxiliary outputs as tree features, member-specific prediction pipeline.
6. **Bayesian bootstrap.** Dirichlet weights, weighted learner interface.
7. **Stability selection.** Half-subsampling without replacement, selection frequencies across the regularization path, threshold and false-positive calibration.
8. **Ordered aggregation and pruning.** Member prediction storage, complementarity-based order, separate validation for the prefix size.
9. **Prefit stacking**, with an explicit leakage warning.
10. **Nested cross-validation** for an honest stack performance estimate.
11. **Learned weights** as an alternative to a fixed combination rule.

### ORACLE_ONLY

1. A universal "unstable predictor will improve" promise.
2. Perfectly unbiased out-of-bag estimate at finite member count.
3. A meta-learner that automatically eliminates train/test distribution shift.
4. A universal optimal ordered prefix.
5. An automatic equivalence between boosting and bagging.
6. The full sklearn generic estimator protocol as a runtime dependency or compatibility layer.
7. The `grf` public C++ strategy plugin ecosystem. Its internal split remains a strong design oracle; exposing equivalent low-level plugins adds ABI and maintenance cost.

### OUT_OF_SCOPE

1. Voting, averaging, and stacking inside `ModelForest` or `ModelBoost`. They belong to `ModelMeta`.
2. `grf` inside the bagging family. Its splitting rule needs a per-node relabeling strategy that today's `GainFn::Custom` cannot express, so it gets its own specification.
3. Bayesian additive regression trees and their GPU variants. Not a bagging method and not a boosting method. They are posterior samplers with a different inferential contract, and they belong with the probabilistic tree work.
4. R package runtime dependencies (`caret`, `baguette`) in a Rust library.
5. The unverified Therneau–Ipelweich–Bachelin 2000 source.
6. Exact claims about the unverified "Bühlmann–Yu 2014" paper.
7. Class-ordering interpretation without an explicit ordinal-response requirement.
8. A single umbrella class covering all three families.

## 15. Citation list

1. Breiman, L. (1996). "Bagging Predictors." *Machine Learning* 24:123–140. DOI [10.1023/A:1018054314350](https://doi.org/10.1023/A:1018054314350) · [PDF](https://www.stat.berkeley.edu/~breiman/bagging.pdf)
2. Breiman, L. (1996). "Stacked Regressions." *Machine Learning* 24:49–64. DOI [10.1007/BF00117832](https://doi.org/10.1007/BF00117832) · [PDF](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)
3. Breiman, L. (2001). "Using Iterated Bagging to Debias Regressions." *Machine Learning* 45:261–277. DOI [10.1023/A:1017934522171](https://doi.org/10.1023/A:1017934522171) · [PDF](https://link.springer.com/content/pdf/10.1023/A:1017934522171.pdf)
4. Bühlmann, P.; Yu, B. (2002). "Analyzing Bagging." *Annals of Statistics* 30:927–961. DOI [10.1214/aos/1031689014](https://doi.org/10.1214/aos/1031689014)
5. Bühlmann, P.; Yu, B. (2003). "Boosting with the L2 Loss: Regression and Classification." *JASA* 98(462):324–339. DOI [10.1198/016214503000125](https://doi.org/10.1198/016214503000125)
6. Bühlmann, P.; Hothorn, T. (2007). "Boosting Algorithms: Regularization, Prediction and Model Fitting." *Statistical Science* 22:477–505. DOI [10.1214/07-STS242](https://doi.org/10.1214/07-STS242) · [PDF](https://stat.ethz.ch/Manuscripts/buhlmann/BuehlmannHothorn_Boosting-rev.pdf)
7. Bühlmann, P. (2014). "Bagging, Boosting and Ensemble Methods." *Handbook of Computational Statistics*. [Manuscript](https://stat.ethz.ch/Manuscripts/buhlmann/handbook-cs-rev2010.pdf)
8. Hothorn, T.; Lausen, B. (2003). "Double-Bagging: Combining Classifiers by Bootstrap Aggregation." *Pattern Recognition* 36:1303–1309. DOI [10.1016/S0031-3203(02)00169-3](https://doi.org/10.1016/S0031-3203(02)00169-3)
9. Hothorn, T.; Lausen, B. (2005). "Bundling Classifiers by Bagging Trees." *Computational Statistics & Data Analysis* 49:1068–1078. DOI [10.1016/j.csda.2004.06.019](https://doi.org/10.1016/j.csda.2004.06.019)
10. Hothorn, T. (2003). "Bundling Predictors in R." DSC 2003. [PDF](https://www.r-project.org/conferences/DSC-2003/Proceedings/Hothorn.pdf)
11. Ho, T. K. (1995). "Random Decision Forests." ICDAR, 278–282. DOI [10.1109/ICDAR.1995.598994](https://doi.org/10.1109/ICDAR.1995.598994)
12. Ho, T. K. (1998). "The Random Subspace Method for Constructing Decision Forests." *IEEE TPAMI* 20:832–844. DOI [10.1109/34.709601](https://doi.org/10.1109/34.709601)
13. Rodríguez, J. J.; Kuncheva, L. I.; Alonso, C. J. (2006). "Rotation Forest: A New Classifier Ensemble Method." *IEEE TPAMI* 28(10):1619–1630. DOI [10.1109/TPAMI.2006.211](https://doi.org/10.1109/TPAMI.2006.211) · [PDF](https://lucykuncheva.co.uk/papers/jrlkcatpami06.pdf)
14. García-Pedrajas, N.; García-Osorio, C.; Fyfe, C. (2007). "Nonlinear Boosting Projections for Ensemble Construction." *JMLR* 8:1–33. [PDF](https://jmlr.org/papers/volume8/garcia-pedrajas07a/garcia-pedrajas07a.pdf)
15. García-Pedrajas, N. (2008). "Boosting Random Subspace Method." *Neural Networks* 21. DOI [10.1016/j.neunet.2008.01.002](https://doi.org/10.1016/j.neunet.2008.01.002)
16. Martínez-Muñoz, G.; Suárez, A. (2004). "Pruning in Ordered Bagging Ensembles." ICML. DOI [10.1145/1143844.1143921](https://doi.org/10.1145/1143844.1143921) · [PDF](http://arantxa.ii.uam.es/~gonzalo/publications/MarSua-aia2004_preprint.pdf)
17. Tutz, G.; Hechenbichler, K. (2005). "Aggregating Classifiers with Ordinal Response Structure." DOI [10.1080/00949650410001729481](https://doi.org/10.1080/00949650410001729481)
18. Meinshausen, N.; Bühlmann, P. (2010). "Stability Selection." *JRSS-B* 72:417–473. DOI [10.1111/j.1467-9868.2010.00740.x](https://doi.org/10.1111/j.1467-9868.2010.00740.x) · [PDF](https://stat.ethz.ch/Manuscripts/buhlmann/stability.pdf)
19. Wolpert, D. H. (1992). "Stacked Generalization." *Neural Networks* 5:241–259. DOI [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1) · [PDF](https://machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)
20. Arlot, S.; Celisse, A. (2010). "A Survey of Cross-Validation Procedures for Model Selection." *Statistics Surveys* 4:40–79. DOI [10.1214/09-SS054](https://doi.org/10.1214/09-SS054)
21. Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection." IJCAI. [PDF](https://ijcai.org/proceedings/1995/0147.pdf)
22. Rubin, D. B. (1981). "The Bayesian Bootstrap." *Annals of Statistics* 9:130–134. DOI [10.1214/aos/1176345338](https://doi.org/10.1214/aos/1176345338)
23. Clyde, M. A.; Lee, H. K. H. (2001). "Bagging and the Bayesian Bootstrap." *NIPS 13*. [PDF](https://proceedings.mlr.press/r3/clyde01a/clyde01a.pdf)
24. Kuncheva, L. I. (2004/2014). *Combining Pattern Classifiers: Methods and Algorithms*. DOI [10.1002/0471660264](https://doi.org/10.1002/0471660264)
25. Friedman, J. H.; Hastie, T.; Tibshirani, R. (2000). "Additive Logistic Regression: A Statistical View of Boosting." *Annals of Statistics* 28:337–407. DOI [10.1214/aos/1016218223](https://doi.org/10.1214/aos/1016218223)
26. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics* 29:1189–1232. DOI [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
27. `caret` reference: [`bag`](https://search.r-project.org/CRAN/refmans/caret/html/bag.html), [`bagEarth`](https://search.r-project.org/CRAN/refmans/caret/html/bagEarth.html), [`train` model list](https://search.r-project.org/CRAN/refmans/caret/html/models.html), [source](https://github.com/cran/caret/blob/master/R/bag.R).
28. `baguette` reference: [index](https://baguette.tidymodels.org/reference/index.html), [`bagger()`](https://baguette.tidymodels.org/reference/bagger.html), [`control_bag()`](https://baguette.tidymodels.org/reference/control_bag.html), [package home](https://baguette.tidymodels.org/).
29. scikit-learn: [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html), [`BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html), [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html), [`VotingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html), [`StackingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), [`StackingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html), [ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html), [tree source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py), [forest source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py), [ensemble base source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_base.py), [metaestimators source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/metaestimators.py), [stacking source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_stacking.py), [bagging source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_bagging.py).
30. Julia MLJ: [EnsembleModel](https://juliaai.github.io/MLJ.jl/stable/models/EnsembleModel_MLJEnsembles/), [DecisionTreeClassifier](https://juliaai.github.io/MLJ.jl/stable/models/DecisionTreeClassifier_DecisionTree/), [machines](https://juliaai.github.io/MLJ.jl/stable/machines/), [model search](https://juliaai.github.io/MLJ.jl/stable/model_search/).
31. tidymodels `parsnip`: [decision_tree](https://parsnip.tidymodels.org/reference/decision_tree.html), [rand_forest](https://parsnip.tidymodels.org/reference/rand_forest.html), [boost_tree](https://parsnip.tidymodels.org/reference/boost_tree.html), [bag_tree](https://parsnip.tidymodels.org/reference/bag_tree.html).
32. R [partykit](https://search.r-project.org/CRAN/refmans/partykit/html/partykit.html), [ranger](https://search.r-project.org/CRAN/refmans/ranger/html/ranger.html), [randomForest](https://search.r-project.org/CRAN/refmans/randomForest/html/randomForest.html), [gbm](https://search.r-project.org/CRAN/refmans/gbm/html/gbm.html), [rpart](https://search.r-project.org/CRAN/refmans/rpart/html/rpart.html).
33. XGBoost [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), LightGBM [parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html), CatBoost [Python reference](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).
34. `grf` algorithm reference [REFERENCE.md](https://github.com/grf-labs/grf/blob/master/REFERENCE.md), [developing guide](https://grf-labs.github.io/grf/DEVELOPING.html), [ForestTrainer header](https://github.com/grf-labs/grf/blob/master/core/src/forest/ForestTrainer.h), [RelabelingStrategy header](https://github.com/grf-labs/grf/blob/master/core/src/relabeling/RelabelingStrategy.h), [SplittingRule header](https://github.com/grf-labs/grf/blob/master/core/src/splitting/SplittingRule.h).
35. Rust [`sklears-tree` ModelTree](https://docs.rs/sklears-tree/latest/sklears_tree/model_tree/index.html).
36. sklearn upstream [model tree issue #13106](https://github.com/scikit-learn/scikit-learn/issues/13106) and [closed M5P PR #13732](https://github.com/scikit-learn/scikit-learn/pull/13732).
37. Cubist GPL source, RuleQuest (2010), release 2.07. [https://github.com/topepo/Cubist](https://github.com/topepo/Cubist)
38. CRAN [`Cubist` vignette](https://cran.r-project.org/web/packages/Cubist/vignettes/cubist.html), [reference manual](https://cran.r-project.org/web/packages/Cubist/refman/Cubist.html).
