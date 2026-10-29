# Tree-growth split criteria survey — 2026-10-29 (scout report; citations verified against sources)

All evidence is gathered. Here is the complete deliverable.

---

# Tree-growth split criteria survey — grounding for mltreelib's split-criterion & custom-objective design

## A. Per-target findings

### 1. C4.5 / C5.0 (Quinlan)

**Split selection — information gain and gain ratio.** Node impurity is entropy $i(t) = -\sum_j p(j|t)\log_2 p(j|t)$ ([Loh, *Tree-Structured Classifiers*, p.2](https://pages.stat.wisc.edu/~loh/treeprogs/guide/wires.pdf)). For a split $s$ on attribute $X$ into subnodes $t_1..t_m$ (plus $t_0$ = missing), with $f(t)$ = fraction non-missing and $q(t_k)=N(t_k)/N(t)$, the gain is $g_s(t) = f(t)\{i(t) - \sum_k q(t_k)i(t_k)\}$ (missing observations excluded from the entropy difference; same source, p.2–3). The selected split maximizes the **gain ratio** $\dfrac{g_s(t)}{-\sum_{k=0}^{m} q(t_k)\log_2 q(t_k)}$ (Loh p.3; the denominator is the "split information" / intrinsic value $-\sum |S_v|/|S|\log(|S_v|/|S|)$ — [Rokach & Maimon, *Data Mining & Knowledge Discovery Handbook*, ch. 9 §3.8](https://www.ise.bgu.ac.il/faculty/liorr/hbchap9.pdf), which also gives the two-stage selection rule: keep only attributes whose information gain is ≥ average gain, then take the best ratio — "not defined when the denominator is zero… may favor attributes for which the denominator is very small").

**Multiway splits.** "C4.5 uses a binary split of the form X ≤ c if X is an ordered variable. But if X is categorical, C4.5 creates one branch for each category" — very fast on categoricals, but thinly populated branches; missing values are *distributed* to every child with weight proportional to $q(t_k)$ after the split is chosen (Loh p.4).

**Stopping / pruning — the MDL question.** Shipped C4.5/C5.0 does **not** use an MDL stop rule: C4.5 grows then prunes with a pessimistic binomial error estimate — "a node is pruned if its error rate increases with splitting. Further, it is replaced by its most frequently used branch if the error rate of the branch is smaller" (Loh p.4; the error-based-pruning formulas $\varepsilon^{UB} = \varepsilon + Z_\beta\sqrt{\varepsilon(1-\varepsilon)/|S|}$ are in Rokach §6.6). C5.0: "A large tree is first grown… then pruned by removing parts predicted to have a relatively high error rate" — per-subtree pruning + global stage (`-g` disables), `-c CF` controls error-estimate severity, `-m cases` is pre-pruning ([RuleQuest C5.0 tutorial, "Advanced pruning options"](http://www.rulequest.com/see5-unix.html); full-text grep of the tutorial finds no "MDL"). The **MDL-based stopping/pruning rule in this lineage** is Quinlan & Rivest (1989), *Inferring decision trees using the MDL principle*, Inf. & Comput. 80:227–248 ([PDF](https://people.csail.mit.edu/rivest/pubs/QR89.pdf), scanned; formulation corroborated by [evtree's JSS paper](https://rdrr.io/cran/evtree/f/inst/doc/evtree.pdf) p.9: "evaluation functions include BIC and *minimum description length* (MDL, as in Quinlan and Rivest 1989)"). MDL scoring = bits to encode tree + data; a split is worth keeping only if encoding cost decreases. Rokach §6.8 gives an operational split-cost formula (Mehta, Rissanen & Agrawal, KDD'95): $\mathrm{Cost}(t) = \sum_i |y=c_i S_t|\ln\frac{|S_t|}{|y=c_iS_t|} + \frac{|\mathrm{dom}(y)|-1}{2}\ln\frac{|S_t|}{2} + \frac{|\mathrm{dom}(y)|}{2}\ln\frac{|\mathrm{dom}(y)|}{2}$ — cheap enough to implement as an optional pruning criterion.

### 2. OneR (baseline floor)

Holte (1993), *Very Simple Classification Rules Perform Well on Most Commonly Used Datasets*, ML 11:63–90 ([PDF](https://gwern.net/doc/ai/tabular/1993-holte.pdf)). Criterion is minimal: for each attribute, build a **1-level tree** (rule = majority class per attribute value), score = per-attribute **classification error rate**; pick the single attribute with the lowest error ([OneR CRAN vignette](https://cran.r-project.org/web/packages/OneR/vignettes/OneR.html): "basically 1-level decision trees… `Attribute Accuracy` table… *Chosen attribute due to accuracy and ties method*"; ties = first attribute, optionally χ² tie-break on contingency tables; missing values = separate level or dropped). This is our floor: any custom-criterion ABI must at minimum express "score a whole attribute by majority-error" without code changes.

### 3. grf (Athey/Tibshirani/Wager) — the closest analogue to our design

**Default splitting rule** — [core/src/splitting/RegressionSplittingRule.cpp](https://raw.githubusercontent.com/grf-labs/grf/master/core/src/splitting/RegressionSplittingRule.cpp):
- `min_child_size = max(ceil(size_node * alpha), 1)` (line 58) — `alpha` "controls the maximum imbalance of a split", default 0.05 ([ll_regression_forest.R roxygen](https://raw.githubusercontent.com/grf-labs/grf/master/r-package/grf/R/ll_regression_forest.R)).
- Per candidate split (lines 186–192):
  ```cpp
  double decrease = sum_left * sum_left / weight_sum_left + sum_right * sum_right / weight_sum_right;
  double penalty = imbalance_penalty * (1.0 / n_left + 1.0 / n_right);
  decrease -= penalty;
  ```
  i.e. weighted variance reduction ($\Sigma y^2$-form, parent term constant) minus an edge-imbalance penalty (`imbalance.penalty`, default 0). Split accepted only if `best_decrease > 0` (lines 81–83). Same pattern with $(\sum y)^2$-form in `MultiCausalSplittingRule.cpp:276-281` and `CausalSurvivalSplittingRule.cpp:309-312`.
- Missing values are handled *inside* the scan (both "send left" directions evaluated, lines 152–184).

**`ll.split.*` local-linear split scoring** — implemented as a **relabeling strategy**, not a new criterion ([core/src/relabeling/LLRegressionRelabelingStrategy.cpp](https://raw.githubusercontent.com/grf-labs/grf/master/core/src/relabeling/LLRegressionRelabelingStrategy.cpp)): at each node, fit ridge regression of $y$ on the `ll.split.variables`; replace each $y_i$ by residual $\hat y_i - y_i$ (lines 84–88), then the *standard* variance-reduction rule runs on the residuals. Knobs: `ll.split.lambda` (default 0.1), `ll.split.weight.penalty` (standard ridge $\lambda\cdot\mathrm{tr}(M)/(p{+}1)$ vs. covariance ridge $\lambda M_{jj}$, lines 63–74), `ll.split.cutoff` (leaf size below which the **overall** $\beta$ from the full data is used, lines 58–62); `overall.beta = (D^\top D + \lambda J)^{-1}D^\top Y$ with $J_{11}=0$ (intercept unpenalized) computed in R ([ll_regression_forest.R:146-159](https://raw.githubusercontent.com/grf-labs/grf/master/r-package/grf/R/ll_regression_forest.R)). **Takeaway: the criterion itself stayed fixed; only the y the criterion sees was transformed.** That is exactly our "lazy y slice + fast-path y-transform" idea, shipped in production.

### 4. mboost (R) — trees on negative gradients

`btree` is a ctree stump: `btree(..., tree_controls = partykit::ctree_control(teststat = "quad", testtype = "Teststatistic", mincriterion = 0, minsplit = 10, minbucket = 4, maxdepth = 1, saveinfo = FALSE))` ([mboost baselearners.Rd](https://rdrr.io/cran/mboost/man/baselearners.html), "tree based base-learner"). So on gradient-targets (negative gradients of the loss at current fit), the split criterion is **not** an impurity at all — it is ctree's *quadratic maximally-selected test statistic* with the raw statistic as the acceptance criterion (`testtype="Teststatistic"`, `mincriterion=0`). **There is no `splitrule` parameter in mboost** — no "SS"/"twoing" knob exists in `btree`/`bbs` (checked the full baselearners man page). "Twoing" is Breiman's CART binary criterion (below); "SS" is rpart-speak for the sum-of-squares deviance. Extensibility: `buser(X, K)` lets you define a custom *base learner* (design + penalty matrix), but not a custom split rule.

### 5. ngboost — base-tree growth only

`default_tree_learner = DecisionTreeRegressor(criterion="squared_error", min_samples_split=2, min_samples_leaf=1, max_depth=3, splitter="best")` ([ngboost/learners.py](https://raw.githubusercontent.com/stanfordmlgroup/ngboost/master/ngboost/learners.py)). I.e., plain CART variance reduction on natural-gradient pseudo-residuals — a second data point that boosting libraries grow trees with the ordinary MSE criterion on transformed targets rather than custom split criteria.

### 6. M5 / M5P / Cubist — model trees (most relevant to us)

**SDR (standard deviation reduction)**, the model-tree growth criterion (Quinlan 1992, *Learning with Continuous Classes*):

$$\mathrm{SDR} = \sigma(D) - \left(\frac{|D_1|}{|D|}\sigma(D_1) + \frac{|D_2|}{|D|}\sigma(D_2)\right)$$

(stated verbatim as "the split rule of our causal M5 model tree… based on standard deviation reduction (SDR)" with $\sigma$ = std. dev. of the target in the node — [Yang & Noor-E-Alam, arXiv:2604.27307, Eq. 1](https://arxiv.org/html/2604.27307v1), Algorithm 1 confirms split selection = argmax SDR over attributes × cutpoints). Corroborating description: "splits… using tests on single attributes that maximally reduce variance in the target variable" ([Landwehr, Hall & Frank, *Logistic Model Trees*, §2.2](https://link.springer.com/content/pdf/10.1007/978-3-540-39857-8_23.pdf)).

**M5' pipeline** (same §2.2): grow regression tree by SDR → fit **linear models at every inner node** on the attributes used in its subtree → simplify models by dropping attributes when "the decrease in the number of parameters outweighs the increase in observed training error" → prune subtrees when the root model's estimated error ≤ subtree's expected error → **smoothing**: "combines the model at a leaf with the models on the path to the root to form the final model placed at the leaf" (the classic $w' = (nw + mw_{\text{parent}})/(n{+}m)$ shrinkage is in Wang & Witten 1997; I could not open a copy of that PDF — treat the exact formula as [INFERENCE] until checked against Weka's `M5Base`). Weka `M5P` exposes `-N` unpruned, `-U` **unsmoothed**, `-R` plain regression tree instead of model tree, `-M` min instances (default 4) ([Weka M5P docs](https://weka.sourceforge.io/doc.stable-3-8/weka/classifiers/trees/M5P.html)). **Cubist** is the rule/instance-based descendant: same M5 fitting plus committees and nearest-neighbor corrections ([CRAN Cubist manual](http://ftp.gwdg.de/pub/misc/cran/web/packages/Cubist/Cubist.pdf): "rule-based model described in Quinlan (1992) (aka M5) with additional corrections based on nearest neighbors in the training set, as described in Quinlan (1993a)") — committees touch leaf fitting, not split selection.

### 7. Conditional-inference / statistical splitting

**ctree (partykit)** ([ctree_control docs](https://rdrr.io/cran/partykit/man/ctree_control.html)): two statistics — `teststat = "quadratic" | "maximum"` for **variable selection** (a linear or maximally selected statistic testing $H_0$: independence of $Y$ and $X_j$), and `splitstat = "quadratic" | "maximum"` for **splitpoint selection** (prior to partykit 1.2-0 only `maximum`). `testtype = "Bonferroni" | "MonteCarlo" | "Univariate" | "Teststatistic"`: p-values from the asymptotic distribution, Bonferroni-adjusted; or Monte-Carlo (`nresample=9999` permutations; "Bonferroni-adjusted Monte-Carlo p-values are computed when both `Bonferroni` and `MonteCarlo` are given"); `Teststatistic` uses the raw statistic. A split is implemented only if `mincriterion = 1 - alpha` (default 0.95) is exceeded and sample-size constraints (minsplit/minbucket/minprob) hold. `multiway=TRUE` enables multiway splits on unordered factors. Purpose: kill selection bias toward variables with many cutpoints (Loh's simulation table shows CART/C4.5 wildly biased vs. GUIDE ≈ 0.167 uniform — wires.pdf, Table 2).

**CHAID** (Kass 1980) ([R CHAID man page](https://rdrr.io/rforge/CHAID/man/chaid.html), which reproduces the IBM SPSS Algorithms spec): multiway, χ²-based, three-step node processing — (1) **merging**: repeatedly merge the least-significantly-different pair of categories (largest p-value) while p > `alpha2`; adjacent pairs only for ordinal predictors; (2) **splitting**: pick the predictor with the smallest **Bonferroni-adjusted** p-value; split only if ≤ `alpha4`, else terminal node; (3) **stopping**: pure node / no predictor variation / max depth / min node size / too-few-cases children merged. Each surviving category of the chosen predictor = one child → inherently multiway.

**GUIDE** (Loh; wires.pdf p.5): two-stage — variable selection by χ² tests on residuals of a linear fit (**residual independence tests**; main effects + pairwise interactions, multiplicity-controlled), then split-point by LDA/QDA on the selected variable; missing values get their own category during the test and go left ($c=-\infty$) afterwards; CART-style CV pruning. Designed specifically to remove the selection bias of CART/C4.5.

### 8. rpart — the classic extensible criterion API (our ABI evidence)

**Built-in deviance formulas** (source of [cran/rpart](https://github.com/cran/rpart)):
- `anova.c`: node deviance = weighted SS $\sum w_i(y_i-\bar y_w)^2$ (`anovass`, lines 30–50); split improvement = between-group SS $w_l(\bar y_l-\bar y)^2 + w_r(\bar y_r-\bar y)^2$ computed incrementally after centering (comment block, lines 63–76; scan lines 100–127), **reported as a fraction of the parent risk**: `*improve = best / myrisk` (line 139). Categorical predictors: categories ordered by mean y, then **all** $2^{k-1}$ subsets enumerated by Gray code (lines 155–200).
- `gini.c`: node risk = expected misclassification cost under `prior` and `loss` matrix: $\min_c \sum_k n_k\,\mathrm{loss}(c,k)\,\mathrm{prior}_k$ (`ginidev`, lines 115–152); split gain = absolute impurity decrease $\mathrm{total\_ss} - \mathrm{best}$ with $\mathrm{total\_ss}=\sum \mathrm{rwt}\cdot I(p)$ and $I$ = `gini_impure1(p)=p(1-p)` or `gini_impure2(p)=-p\log p` (switch = `parms$split` "gini"/"information", `giniinit` lines 37–42; scan lines 218–255). Priors are folded in as "altered weights" `aprior`.

**Custom-method interface** — [`method` = list(init, split, eval)](https://rdrr.io/cran/rpart/man/rpart.html) ("Alternatively, `method` can be a list of functions named `init`, `split` and `eval`… see the vignette 'User Written Split Functions'"); full spec in the [usercode vignette](https://cran.r-project.org/web/packages/rpart/vignettes/usercode.pdf) (Therneau):
- `init(y, offset, parms, wt)` → `list(y, parms, numresp, numy, summary, text)` — raw response + user params in, transformed y (e.g. `cbind(y, offset)` to smuggle state into the split function) and arbitrary `parms` out;
- `eval(y, wt, parms)` → `list(label, deviance)` — called once per node; "the deviance value… does not need to be an actual deviance… any impurity measure… pruning will be most efficient if the value 0 corresponds to a perfectly pure node";
- `split(y, wt, x, parms, continuous)` → `list(goodness, direction)` — called **once per covariate per node**; `y` is the node's response slice, `x` pre-sorted (missing rows already removed), and it returns a **vector over all n−1 cutpoints** ("goodness: the utility of the split, where larger numbers are better… the parent code is responsible for handling the minimum node size constraint, and also for dealing with ties"). The vignette's worked example reproduces built-in anova exactly (`all.equal(fit1$frame, fit2$frame) == TRUE`), and its logistic-regression example shows the limit case: the eval function *fits a GLM per node*.
- Cost: user rules are "many times slower than the built-in ones"; C-level extension = 4 functions in `func_table.h` `{init_split, choose_split, eval, error}` + recompile. CV for user rules is manual (`xpred.rpart`).

**evtree (brief)**: no per-split criterion at all — evolutionary search; candidate trees scored by `loss + α·comp`: classification $2N\cdot MC + \alpha M\log N$, regression $N\log \mathrm{MSE} + \alpha\,4(M{+}1)\log N$ (BIC-type) ([evtree JSS vignette §3.4](https://rdrr.io/cran/evtree/f/inst/doc/evtree.pdf); `evtree.control(alpha, minbucket, minsplit, maxdepth, …)` has no `splitrule`). Useful as the counterpoint: whole-tree fitness, not greedy splits.

---

## B. Comparison table

| Library | Criterion (formula anchor) | What the criterion receives | Multiway? | Extensible? |
|---|---|---|---|---|
| **C4.5/C5.0** | Entropy gain $g_s = f(t)(i(t)-\sum_k q_k i_k)$; select by **gain ratio** $g_s / (-\sum q_k\log_2 q_k)$, pre-filtered to gain ≥ avg (Loh wires.pdf p.2–3; Rokach §3.8) | class counts per child (impurity over $p(j|t)$) | **Yes**, one branch per category (categorical) | No (commercial/GPL binaries; no hook) |
| **OneR** | majority-class error per attribute value; pick min error (Holte 1993; OneR vignette) | contingency table of attribute × class | Yes (one branch per value) | No |
| **CART / rpart anova** | improvement $= w_l(\bar y_l-\bar y)^2 + w_r(\bar y_r-\bar y)^2$, as fraction of parent risk (`anova.c:63-76,139`) | node y-slice, weights, sorted x; deviation computed from means | No (binary; categorical via Gray-code 2^{k−1} subsets) | **Yes** — `method=list(init,split,eval)`: raw y + wt + sorted x + free-form `parms` → per-cut `goodness`+`direction` (usercode.pdf) |
| **rpart class** | impurity decrease with $I = p(1-p)$ or $-p\log p$ over altered priors; node risk = min misclassification cost (`gini.c:37-42,115-255`) | as above + `prior`, `loss`, `split` via `parms` | No | Yes (same hook) |
| **ctree/partykit** | quadratic/maximum test statistic on linear statistic of $Y$; accept if $1-p$ > `mincriterion`; `testtype` Bonferroni/MonteCarlo(n=9999)/Univariate/Teststatistic; `splitstat` for cutpoint (ctree_control docs) | full node sample for hypothesis test + permutations | `multiway=TRUE` for factors | Partially — `ytrafo`/`splitflavour`, `update` arg; no arbitrary criterion |
| **CHAID** | χ² significance: merge categories (p>α₂), choose predictor by smallest Bonferroni-adjusted p, split if ≤ α₄ (Kass 1980; CHAID man page) | contingency tables, p-values | **Yes** (each merged category = child) | No |
| **GUIDE** | χ² tests on residuals (independence), multiplicity-controlled; then LDA split point (Loh wires.pdf p.5) | residuals from node fit, per-predictor tests | Yes (CRUISE-style) | No |
| **M5/M5P/Cubist** | SDR $= \sigma(T) - \sum_i \frac{|T_i|}{|T|}\sigma(T_i)$ (arXiv:2604.27307 Eq.1) | target values per child (sd), i.e. raw y-slices | No (binary) | No (Weka `-R/-U/-N/-M` knobs only) |
| **grf** | $\frac{(\sum_L y)^2}{W_L} + \frac{(\sum_R y)^2}{W_R} - \mathrm{imbalance\_penalty}(\frac1{n_L}+\frac1{n_R})$ (RegressionSplittingRule.cpp:186-192) | weighted sums/counts per candidate bucket; missing handled in-scan; `ll.split` sees **ridge-residual relabeled y** (LLRegressionRelabelingStrategy.cpp) | No | Criterion closed; but **y-transform hook** (`enable.ll.split`) + `alpha`/`imbalance.penalty` split acceptance |
| **mboost btree** | ctree quadratic teststat on negative-gradient pseudo-residuals, `mincriterion=0` (baselearners.Rd) | pseudo-residuals via ctree machinery | via ctree `multiway` | `buser` replaces the base learner, not the split rule |
| **ngboost** | sklearn `squared_error` on natural gradients (learners.py) | y = pseudo-residuals | No | sklearn's own limits |
| **xgboost** (one row, tree-growth view) | regularized gradient gain (formula below) | per-row $g_i,h_i$ aggregated to $G,H$ per bucket | No | Custom **objective** (raw preds → grad/hess), never the split criterion |
| **evtree** | whole-tree fitness $2N\!\cdot\!MC + \alpha M\log N$ / $N\log\mathrm{MSE} + \alpha 4(M{+}1)\log N$ (JSS §3.4) | full-tree predictions vs y | n/a (global search) | `alpha` weight only |

sklearn reference enum (parity target): classifier `{"gini","entropy","log_loss"}`, regressor `{"squared_error","absolute_error","poisson"}`; **no custom criterion is accepted** — the parameter is a validated string enum ([sklearn/tree/_classes.py:916, 1332](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/tree/_classes.py)); extension requires subclassing the Cython `Criterion` in `_criterion.pyx` and recompiling. Notably `"friedman_mse"` is deprecated in 1.9 and **"maps to `squared_error` as both were always equivalent"** (`_classes.py:1585-1592`) — evidence to *not* ship Friedman MSE as a separate built-in.

## C. Exact split-gain formulas

**XGBoost (regularized, boosting trees)** ([Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/tutorials/model.html), "The Structure Score"/"Learn the tree structure"): with $\omega(f) = \gamma T + \tfrac12\lambda\sum_j w_j^2$, $G_j=\sum_{i\in I_j}g_i$, $H_j=\sum_{i\in I_j}h_i$, optimal leaf weight $w_j^* = -\frac{G_j}{H_j+\lambda}$ and split gain

$$\mathrm{Gain} = \frac12\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

("if the gain is smaller than $\gamma$, we would do better not to add that branch" — that is its pruning rule; $\lambda$=`reg_lambda`, $\gamma$=`min_split_loss` per [parameter docs](https://xgboost.readthedocs.io/en/stable/parameter.html)).

**rpart anova**: node deviance $\mathrm{SS}=\sum w_i(y_i-\bar y_w)^2$; split improvement $= w_l(\bar y_l - \bar y)^2 + w_r(\bar y_r - \bar y)^2$; reported relative to parent: `improve = best / myrisk` (`src/anova.c:63-76,139`). Equivalent to grf's $\frac{(\sum y)^2}{W}$-form.

**rpart class**: node deviance $= \min_c \sum_k n_k\,\mathrm{loss}(c,k)\,\mathrm{prior}_k$ (expected cost, `src/gini.c:115-152`); split gain $= \mathrm{total\_ss} - \min_{\text{splits}}\big[\sum_{\text{child}} \mathrm{rwt}\cdot I(p(\cdot|\text{child}))\big]$ with $I(p)=p(1-p)$ or $-p\log p$ and prior-altered weights (`src/gini.c:206-255`).

Also, for the table's M5/grf/C4.5 rows: SDR and gain-ratio formulas are in §A.1/A.6/A.3 above with anchors.

---

## D. Recommendations for mltreelib

**Built-ins worth shipping** (ordered by evidence strength + our linear-leaf needs):

1. **Variance reduction / MSE** (`squared_error`) — universal workhorse (sklearn default, ngboost/mboost trees, grf's criterion is its $(\sum y)^2$ form). Needs only incremental `(n, Σwy, Σwy²)` per candidate bucket — cheap in a histogram scan.
2. **SDR** for model trees — the canonical criterion for exactly our leaf type (linear leaves, Quinlan 1992). Mathematically it is a monotone rescaling of variance reduction in most cases, but not identical (sd vs. variance weighting); shipping it buys M5/Cubist parity and a documented story. Pair it with the M5'-style **smoothing** of leaf coefficients along the root path and the expected-error-based simplification/pruning of leaf models (Landwehr §2.2) — those are leaf-fitting, but the criterion only makes sense with them.
3. **Gini and entropy/log_loss** for classification — sklearn enum parity (`gini`/`entropy`/`log_loss` as alias). Same incremental stats as MSE.
4. **Poisson deviance** — sklearn + rpart `poisson` both have it; only needs $(\sum wy, n)$ per node.
5. Skip `friedman_mse` (sklearn proves it's an alias of squared_error), skip MAE in v1 (sklearn docs: 3–6× slower fit — the order-statistic machinery is real cost), skip twoing/DKM/ORT/KS/AUC-split (Rokach §3.6–3.14 catalog, §3.16: "in most of the cases, the choice of splitting criteria will not make much difference"). Optional later: an MDL **pruning** criterion (Mehta et al. cost formula, Rokach §6.8) — it's a stopping/pruning hook, not a split scorer.

**Custom-criterion ABI — recommended shape.** The evidence converges hard:

- rpart's 30-year-old interface is the only *true* custom split-criterion API in the wild, and its contract is: **per node, per candidate feature — raw y slice + weights + the x values (sorted) + user params in; a vector of per-cut scores + optional direction out; plus a separate `eval` returning (leaf value, node deviance)**. It deliberately passes raw values, not aggregates (its logistic example fits a GLM inside `eval`), and it explicitly keeps min-node-size and tie-handling *out* of the user function. Its cost is interpreted-call overhead — fine as an escape hatch.
- grf demonstrates the other half: many "custom criteria" are really **y-transforms** (ridge residuals) feeding a stock criterion. That is exactly what a fast-path expression buys us.
- Boosting libs show the aggregate-only ABI (grad/hess sums) is sufficient *for boosting*, but it cannot express SDR/MAE/quantile/GLM criteria — wrong default for a tabular single-tree core.

So keep the planned two-tier design, specified as:

```rust
trait SplitCriterion {
    fn init(&self, y: &[f64], w: &[f64], parms: &Parms) -> NodeState; // rpart init
    fn node_risk(&self, y: &[f64], w: &[f64]) -> f64;                 // rpart eval deviance; 0 = pure
    fn leaf_value(&self, y: &[f64], w: &[f64]) -> LeafValue;          // scalar or linear model
    // once per (node, feature): all cutpoints scored at once (rpart split contract)
    fn score_cuts(&self, y: LazySlice<'_>, w: LazySlice<'_>, x_sorted: &[f64]) -> Vec<f64>;
}
```

with (a) **lazy y slices** materialized per (node, feature) — the rpart-proven sufficient interface for *any* criterion, including ones that fit models; (b) the **fast-path expression string** interpreted as a *y-transform* (like grf's relabeling) or as an arithmetic score over node aggregates `(n, Σwy, Σwy², Σw)` — covering the built-ins and ~all user cases at compiled speed; (c) full closure criterion as the escape hatch, documented as "interpreted overhead" exactly as rpart does. Do **not** make grad/hess the ABI; if we later want boosted trees, add an adapter that turns `(g,h)` slices into a criterion. Borrow grf's `alpha`-min-child + `imbalance.penalty` as generic split-acceptance parameters (they are criterion-agnostic and demonstrably useful), and rpart's convention that scores are "larger is better, 0 = no split" so the parent retains pruning/min-size control.

**Checks for main agent:** none (research-only, no repo edits). One gap to note: the exact M5' smoothing formula $w'=(nw+mw_p)/(n{+}m)$ is [INFERENCE] — Wang & Witten (1997) PDFs were inaccessible; verify against Weka `M5Base.java` if we implement smoothing.