# NGBoost leaf & training behavior — 2026-10-29 (scout report; citations verified against source)

# NGBoost (StanfordMLGroup/ngboost) source deep-dive — ground truth for "distributional leaves" in mltreelib

All citations are to the **master branch** of github.com/StanfordMLGroup/ngboost as read 2026-10-29 (raw.githubusercontent.com/StanfordMLGroup/ngboost/master/…); line numbers verified against the raw files. Two naming corrections to the ticket: distributions live in **`ngboost/distns/`** (a package, 18 modules — not `ngboost/dist/`), and the per-round fit helper is **`fit_base`** (not `_fit_base`) at `ngboost/ngboost.py:226`. "TDist" is class `T` (`ngboost/distns/t.py:46`); "Multinomial" is the factory `k_categorical(K)` (`ngboost/distns/categorical.py:44-90`); `Bernoulli = k_categorical(2)` (`ngboost/distns/categorical.py:93`).

---

## 1. Leaf fitting — what is actually fit inside a leaf

**Nothing distributional is fit inside a leaf.** The base learner is `sklearn.tree.DecisionTreeRegressor` (`ngboost/learners.py:4-12`); each tree is fit by ordinary least squares to **one parameter's natural-gradient components**, so each leaf holds **one constant per (round, parameter): the (weighted) mean of that gradient column** over the rows routed there (sklearn squared-error leaves store the mean of the training targets — sklearn tree docs, https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation). The base learners are cloned per parameter and fit to `grads.T` columns — `zip(base_learners, grads.T, seeds)` (`ngboost/ngboost.py:251-254`, `learner.fit(X, g)` at `:242-248`); `_base_learners()` returns `[self.Base] * n_params` (`ngboost/ngboost.py:213-215`), so **one tree per distribution parameter per boosting round** (e.g. Normal = 2 trees/round; T = 3; `k_categorical(K)` = K−1; MVN = k(k+3)/2). Every round updates **all** parameters; no tree predicts "all of loc/scale/shape" at once.

The full distribution lives at the **ensemble** level: the per-row parameter vector is the global init plus additive tree increments:

$$\theta_k(x) \;=\; \theta_{0,k} \;-\; \nu \sum_{r} s_r \, f_{r,k}(x)$$

(`params = init broadcast` `ngboost/ngboost.py:177`; `params -= self.learning_rate * resids * s` `:183-184` at predict, `:544-547` at train). Sign convention: trees are fit to the **gradient** `g = ∇_θ L` and the update **subtracts** `lr·scale·f(x)` — algebraically the same as fitting the negative gradient and adding.

**Precision on "natural parameters":** NGBoost does **not** generally store canonical exponential-family natural parameters. Each dist defines an unconstrained **working chart** (log-links for positive params) and the natural gradient $F(\theta)^{-1}\nabla L$ is computed *in that chart*; since natural gradients are invariant to smooth reparameterization (Amari; noted in the NGBoost paper, https://arxiv.org/abs/1910.03225 §2), the direction matches the canonical one. Where the chart *is* canonical: Poisson stores $\log\mu$ (`ngboost/distns/poisson.py:43-44`) and categorical stores logits vs. class 0 (`ngboost/distns/categorical.py:59-61`). Where it is not: Normal stores $(\mu, \log\sigma)$, not $(\mu/\sigma^2, -1/2\sigma^2)$ (`ngboost/distns/normal.py:68-69`). The "additive update on the natural parameter" of the ticket is therefore **an additive update on the working chart along the natural-gradient direction**.

### Deliverable A — the boosting loop (10 lines, file:line)

1. **Init (global, once):** $\theta_0$ = marginal MLE/moments of the dist on all Y: `self.init_params = self.Manifold.fit(Y)` — `ngboost/ngboost.py:170-173`, called at `:499`; e.g. Normal `fit(Y) → [mean, log sd]` `ngboost/distns/normal.py:73-75`, Poisson NLE fit `ngboost/distns/poisson.py:47-58`.
2. **Working params per row:** `params = np.ones((m, n_params)) * self.init_params` (`ngboost/ngboost.py:175-177`), likewise `val_params` for validation (`:515`).
3. **Per round `itr`** (`for itr in range(len(self.col_idxs), …)` `ngboost/ngboost.py:529`): subsample rows (`minibatch_frac`, w/o replacement `:191-193`) and columns (`col_sample` `:195-200`); record `col_idx` (`:533`).
4. **Instantiate manifold:** `D = self.Manifold(P_batch.T)` — the class `Manifold(Distribution.implementation(Score), Distribution)` mixing the dist-specific score into the dist (`ngboost/ngboost.py:535`, `ngboost/manifold.py:12`; score resolution `{S.__bases__[-1]: S}` `ngboost/distns/distn.py:45`).
5. **Score (loss):** `train_loss_monitor(D, Y_batch, W)` = `D.total_score` = mean of the proper scoring rule (`ngboost/ngboost.py:537`, monitor default `:519-522`; `total_score` = weighted average `ngboost/scores.py:4-5`).
6. **Natural gradient per parameter:** `grads = D.grad(Y_batch, natural=True)` (`ngboost/ngboost.py:539`) → `grad = d_score(Y)` then `grad = F(θ)^{-1} grad` via `np.linalg.solve(metric, grad)` (`ngboost/scores.py:7-11`, `:64`; pinv fallback `:76`). Shape `(n_batch, n_params)`.
7. **Project onto trees:** `proj_grad = self.fit_base(X_batch, grads, weight_batch)` (`ngboost/ngboost.py:541`) — one cloned `DecisionTreeRegressor` per parameter fit to its gradient column (`:242-254`, `ngboost/learners.py:4-12`); `proj_grad` = the trees' leaf constants on the batch (`:265`).
8. **Line search (scalar step on the projected direction):** `scale = self.line_search(proj_grad, P_batch, Y_batch, weight_batch)` (`ngboost/ngboost.py:542`) — start `scale=1`; **double** while the batch `total_score` of `start − scale·resids` is not worse (cap `scale > 256`) (`:275-283`); then **halve** while still worse and `norm ≥ tol` (`:285-295`); store `self.scalings.append(scale)` (`:296`).
9. **Additive update on the chart:** `params -= learning_rate * scale * [m.predict(X[:, col_idx]) for m in base_models[-1]]` for **all** rows (`ngboost/ngboost.py:544-547`), and identically `val_params -= …` (`:552-558`).
10. **Validation / stopping:** `val_loss = val_loss_monitor(Manifold(val_params.T), Y_val)` (`:559-560`); track `best_val_loss` (`:561-562`); early-stop when the best loss is older than `early_stopping_rounds` rounds (`:563-575`); also stop when mean `‖proj_grad‖ < tol` (`:587-590`). History in `evals_result` keyed by `Score.__name__.upper()` (`:592-596`).
11. **Predict:** `pred_dist` → `self.Dist(params.T)` — a full distribution object per row (`:728-731`); `predict` → `pred_dist(...).predict()` = `self.mean()` for regression (`:775`, `ngboost/distns/distn.py:68-70`) or `argmax class_probs` for classification (`distn.py:72-74`); `predict_proba` → `class_probs()` (`ngboost/api.py:319`).

---

## 2. Distributions and scores — deliverable B

"How one tree predicts" is the same for every row of this table: **each leaf holds one constant = the mean natural-gradient component for exactly one parameter; there are `n_params` trees per round, and the update subtracts `lr·scale·constant` from that parameter's accumulator.** The chart below is the *stored* parameter vector (what gets accumulated); links are applied in `__init__`.

| Distribution (class, file) | Stored params ("working chart" = natural-gradient coordinates) | n_params | Score implementations (exact class names) | Trees / round |
|---|---|---|---|---|
| `Normal` (`distns/normal.py:54-84`) | $(\mu,\ \log\sigma)$ (`:68-69`) | 2 | `NormalLogScore` (`:10-24`), `NormalCRPScore` (`:26-52`) — MLE+CRPS | 2 |
| `NormalFixedVar` (`normal.py:127-148`) | $(\mu)$, σ≡1 (`:139-143`) | 1 | `NormalFixedVarLogScore` (`:92-104`), `NormalFixedVarCRPScore` (`:106-125`) | 1 |
| `NormalFixedMean` (`normal.py:186-207`) | $(\log\sigma)$, μ≡0 (`:198-203`) | 1 | `NormalFixedMeanLogScore` (`:151-163`), `NormalFixedMeanCRPScore` (`:165-184`) | 2→1 |
| `LogNormal` (`distns/lognormal.py:82-115`) | $(\mu_{\log},\ \log\sigma_{\log})$ (`:97-98`) | 2 | censored-only: `LogNormalLogScoreCensored` (`:10-39`), `LogNormalCRPScoreCensored` (`:42-79`); plain use via `uncensor()` (`distns/distn.py:53-65`, `api.py:92-93`) — MLE+CRPS | 2 |
| `Exponential` (`distns/exponential.py:51-81`) | $(\log \text{scale})$ (`:65`) | 1 | censored-only `ExponentialLogScore` (`:12-26`), `ExponentialCRPScore` (`:29-48`) | 1 |
| `Gamma` (`distns/gamma.py:32-60`) | $(\log\alpha_{\text{shape}},\ \log\beta_{\text{rate}})$ (`:38-39`) | 2 | `GammaLogScore` only (`:10-30`, `:34`) | 2 |
| `Poisson` (`distns/poisson.py:27-70`) | $(\log\mu)$ — canonical natural param (`:43-44`) | 1 | `PoissonLogScore` only (`:13-24`, `:35`) | 1 |
| `T` (`distns/t.py:46-77`) | $(\mu,\ \log\sigma,\ \log\nu)$ (`:60-63`) | 3 | `TLogScore` only (`:10-43`, `:56`) | 3 |
| `TFixedDf` (`t.py:106-139`, df=3) | $(\mu,\ \log\sigma)$ (`:121-122`) | 2 | `TFixedDfLogScore` only (`:80-104`) | 2 |
| `TFixedDfFixedVar` (`t.py:161-196`) | $(\mu)$ (`:174`) | 1 | `TFixedDfFixedVarLogScore` (`:142-159`) | 1 |
| `Cauchy` (`distns/cauchy.py:12-13` = `TFixedDf`, df=1); `CauchyFixedVar` (`cauchy.py:17-18`) | $(\mu,\ \log\sigma)$ / $(\mu)$ | 2 / 1 | `CauchyLogScore = TFixedDfLogScore` (`cauchy.py:10`), `CauchyFixedVarLogScore` (`:15`) | 2 / 1 |
| `k_categorical(K)` / `Bernoulli = k_categorical(2)` (`distns/categorical.py:44-93`) | K−1 logits vs. class 0 — canonical (`:53`, `:59-61`) | K−1 (Bernoulli: 1) | `CategoricalLogScore` only (`:10-29`, `:51`); `CategoricalCRPScore` exists (`:32-38`) but `d_score`/`metric` return `None` → unusable for boosting | K−1 |
| `Weibull` (`distns/weibull.py:31-60`) | $(\log c_{\text{shape}},\ \log\lambda_{\text{scale}})$ (`:46-47`) | 2 | `WeibullLogScore` only (`:9-28`, `:41`) | 2 |
| `Laplace` (`distns/laplace.py:51-78`) | $(\mu,\ \log b)$ (`:59-61`) | 2 | `LaplaceLogScore` (`:9-22`), `LaplaceCRPScore` (`:25-48`) — MLE+CRPS | 2 |
| `HalfNormal` (`distns/halfnormal.py:22-52`) | $(\log\sigma)$ (`:37`) | 1 | `HalfNormalLogScore` only (`:9-19`, `:32`) | 1 |
| `MultivariateNormal(k)` factory → `MVN` (`distns/multivariate_normal.py:155-304`) | $k$ means + $k(k{+}1)/2$ entries of $L$, $\Sigma^{-1}{=}LL^\top$, diag exp'd (`:183`, `:150`) | $k(k{+}3)/2$ (`:183`) | `MVNLogScore` only (`:24-120`, `:184`) | k(k+3)/2 |
| `Beta` (`distns/beta.py:20-26`) | $(\log\alpha,\ \log\beta)$ (log-link flag `True`, `sympy_utils.py:190-193`; exp decode `:648`) | 2 | auto `BetaLogScore` (factory names it `f"{name}LogScore"`, `sympy_utils.py:612-620`) | 2 |
| `LogitNormal` (`distns/logitnormal.py:37-45`) | $(\mu,\ \log\sigma)$ (identity/log links `:38`) | 2 | auto `LogitNormalLogScore`; Fisher via MC (`sympy_utils.py:465-470`) | 2 |
| `BetaBernoulli` (`distns/beta_bernoulli.py:41-50`, classification) | $(\log\alpha,\ \log\beta)$ (`:42`) | 2 | auto `BetaBernoulliLogScore` (`:52`) | 2 |
| `BetaBinomial` (`distns/betabinomial.py:69-82`) | $(\log\alpha,\ \log\beta)$ + fixed `n` extra param (`:70-71`) | 2 | auto `BetaBinomialLogScore` | 2 |
| `BetaBinomialEstN` (`betabinomial.py:119-128`) | $(\log\alpha,\ \log\beta,\ \log n)$ (`:120`) | 3 | auto `BetaBinomialEstNLogScore` | 3 |

**Score registry pattern** (the "losses/enums"): every dist carries `scores = [<Dist>LogScore, <Dist>CRPScore…]` (or `censored_scores`, `ngboost/distns/lognormal.py:92`, `distns/utils.py:23-25`); the generic user-facing classes are `LogScore` (alias `MLE = LogScore`, `ngboost/scores.py:80-95`) and `CRPScore` (alias `CRPS = CRPScore`, `scores.py:97-102`). `Dist.implementation(Score)` maps generic→dist-specific via the score's **base class** (`{S.__bases__[-1]: S for S in scores}[Score]`, `distns/distn.py:30-51`), and `manifold(Score, Dist)` mixes the chosen score into the dist class (`ngboost/manifold.py:1-15`). API defaults: `NGBRegressor(Dist=Normal, Score=LogScore, Base=default_tree_learner)` (`ngboost/api.py:70-72`), `NGBClassifier(Dist=Bernoulli…)` (`api.py:174-176`), `NGBSurvival(Dist=LogNormal…)` requiring `censored_scores` (`api.py:383-405`).

---

## 3. Natural gradients — exact mechanics (`ngboost/scores.py`)

- `Score.grad(Y, natural=True)`: `grad = self.d_score(Y)` (ordinary gradient of the loss w.r.t. the stored chart, chain rule through log-links included), then `metric = self.metric()` and `grad = self._natural_gradient(grad, metric)` (`ngboost/scores.py:7-11`).
- `_natural_gradient` solves **per row** $F(\theta_i)\,\tilde g_i = g_i$, i.e. $\tilde g = F^{-1}\nabla$ — `np.linalg.solve(metric, grad[..., None])` (`scores.py:64`), per-row solve fallback (`:73`) and `np.linalg.pinv(metric[i]) @ grad[i]` for singular metrics (`:76`). The metric tensor has shape `(n, n_params, n_params)` (`:15-22`, checks `:24-61`).
- **LogScore metric = Fisher information.** Default is a Monte-Carlo estimate: draw `n_mc_samples=100` from `self.sample(100)`, stack `d_score` at each draw, return the mean outer product `np.mean(np.einsum("sik,sij->sijk", grads, grads), axis=0)` (`scores.py:89-91`) — i.e. $\hat I(\theta)=\mathbb{E}_{\hat Y}[\nabla\log p\,\nabla\log p^\top]$. Most dists override with the analytic Fisher in their chart: Normal $\mathrm{diag}(1/\sigma^2,\ 2)$ in $(\mu,\log\sigma)$ (`distns/normal.py:20-24`); Poisson $\mu$ in $\log\mu$ (`distns/poisson.py:22-24`); Gamma full $2{\times}2$ with $\psi_1$ (`distns/gamma.py:23-29`); Categorical softmax Fisher (`distns/categorical.py:20-27`); TFixedDf $\mathrm{diag}(\frac{\nu+1}{(\nu+3)\sigma^2},\ \frac{2\nu}{\nu+3})$ (`distns/t.py:100-104`); MVN = E[Hessian] (`distns/multivariate_normal.py:66-72`); sympy factory tries analytic FI then falls back to the MC metric with a ridge (`distns/sympy_utils.py:92-111`, `:433-470`).
- **CRPS metric** is the CRPS-induced Riemannian metric, not the Fisher: Normal's is $\frac{1}{2\sqrt{\pi}\sigma}\begin{bmatrix}2&0\\0&\sigma^2\end{bmatrix}$, commented "2 \* integral of grad F grad F^T, which scales like 1 / scale" (`distns/normal.py:42-52`); Laplace $\mathrm{diag}(\tfrac{0.5}{b},\ 0.25\,b)$ (`distns/laplace.py:44-48`); Exponential $0.5\,\sigma$ (`distns/exponential.py:46-48`).
- Every score subclass implements exactly three methods: `score(Y)` (loss value, e.g. `−logpdf`, `normal.py:11-12`), `d_score(Y)` (chart gradient, e.g. $d_0=(\mu-y)/\sigma^2$, $d_1=1-(\mu-y)^2/\sigma^2$ in $(\mu,\log\sigma)$, `normal.py:14-18`), `metric()` ($F$). That triple is the entire per-row math the boosting loop consumes.

---

## 4. Leaf shape / base learner / prediction assembly

| Aspect | Value | Citation |
|---|---|---|
| Default base | `DecisionTreeRegressor(criterion="squared_error", min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, splitter="best")` — `max_depth=3` is the only non-sklearn-default | `ngboost/learners.py:4-12` |
| Alternative base | `default_linear_learner = Ridge(alpha=0.0)` (linear leaves = per-param ridge on gradients) | `learners.py:14` |
| Per-parameter bases | `Base` may be a list of `n_params` estimators, order = dist param order | `ngboost/ngboost.py:213-223`, `api.py:36-45` |
| Trees per round | `n_params` (one per parameter column of `grads.T`) | `ngboost.py:251-254` |
| Leaf payload | one scalar constant per tree (mean of gradient targets); optionally weighted | `ngboost.py:242-248` + sklearn docs (URL above) |
| Prediction assembly | row → $\theta(x)=\theta_0-\sum_r \nu\, s_r f_{r}(x)$ per parameter, then dist link (e.g. `scale=np.exp(params[1])`, `normal.py:69`) | `ngboost.py:175-185` |
| `pred_dist` | returns `self.Dist(params.T)` — a distribution object exposing `params`, `mean()`, `logpdf()`, `class_probs()` etc. | `ngboost.py:728-731`, `distn.py:18-27` |
| `predict` | `pred_dist(...).predict()` = `mean()` (regression) / `argmax class_probs` (classification) | `ngboost.py:775`, `distn.py:68-74` |

---

## 5. Training behavior

| Mechanism | Behavior | Citation |
|---|---|---|
| Learning rate | **constant** `learning_rate=0.01`; **no decay schedule** | `ngboost.py:103`, `:184`, `:544-547` |
| Row/col subsampling | `minibatch_frac=1.0` rows and `col_sample=1.0` cols per round, drawn w/o replacement; `col_size = max(1, int(col_sample·p))` | `ngboost.py:104-105`, `:187-200` |
| Line-search step | per-round **scalar**, shared across all parameter trees: `scale_init=1`, ×2 up while batch score not worse (break at `scale>256`), then ×0.5 down while worse and `‖resids‖≥tol`; stored in `self.scalings` and **replayed at predict** | `ngboost.py:270-297`, `:183-184` |
| Early stopping | metric = `val_loss_monitor`, default `D.total_score` on validation = mean of the chosen Score (LogScore ⇒ NLL; or CRPS) | `ngboost.py:525-527`, `:559-560` |
| Patience | `early_stopping_rounds` (default **None** = disabled, no validation at all); stop when `len(val_loss_list) > R` and `best_val_loss < min(last R val losses)` | `ngboost.py:111`, `:563-575` |
| Validation split | if `early_stopping_rounds` set and no `X_val/Y_val`, auto-split `validation_fraction=0.1` via `train_test_split` | `ngboost.py:110`, `:428-455` |
| Gradient-norm stop | `mean ‖proj_grad‖ < tol` (`tol=1e-4`) quits ("GRAD") | `ngboost.py:108`, `:587-590` |
| Defaults | `n_estimators=500`, `natural_gradient=True` (plain gradient if False, `scores.py:7-11`), `n_jobs` parallelizes the per-param tree fits only (`:250-264`) | `ngboost.py:101-112` |

---

### Deliverable C — invariant per leaf vs. global

**Per leaf (the only leaf-level state):** one scalar per tree = the weighted mean of a single parameter's natural-gradient component over the routed rows; identical in kind to a mean leaf, just fit to gradient targets. No distribution params, no scale, no shape are fit or stored per leaf (`ngboost/learners.py:4-12`, `ngboost.py:242-254`).

**Global / ensemble level:** (1) $\theta_0$ = one constant parameter vector from `Dist.fit(Y)` on the whole marginal, shared by all rows (`ngboost.py:170-173`, `:177`; per-dist `fit` e.g. `normal.py:73-75`); (2) the per-round line-search scalar `scale`, shared by all parameter trees of that round and stored for prediction (`ngboost.py:296`, `:183-184`); (3) constant `learning_rate`; (4) the dist's link transforms and the Score's metric (Fisher or CRPS) — chart mechanics, not learned; (5) per-round row/column subsample indices (`:187-200`).

---

### Deliverable D — translation notes for mltreelib (mean/linear leaves)

- **"Dist leaf" for a standalone tree = `Dist::fit` of the leaf's targets**, literally NGBoost's init step run per leaf: for Normal that is `(mean, log sd)` (`normal.py:73-75`) — and `LeafStats` already computes the required moments (`mltreelib-core/src/node_stats.rs:13-52`); the current mean leaf is the 1-parameter fixed-variance Normal case. Prediction returns per-row parameter vectors (a "dist predict", like `pred_dist`, `ngboost.py:728-731`), with point predict = `mean()` (`distn.py:68-70`).
- **Do not confuse this with NGBoost's leaf**: in boosting, leaves stay constants holding *gradient increments* — one tree per distribution parameter per round — and the distribution exists only as `θ₀ − Σ ν·s·leaf_const` per row (`ngboost.py:183-184`, `:544-547`). If you later add dist boosting, "vector-valued constant leaves" (n_params columns per round) is the faithful design.
- **Trait seam mirrors NGBoost's manifold mixin** (`manifold.py:12`, `distn.py:30-51`): `Dist { fn fit(y)->θ₀; fn link(θ) (exp for positive params, normal.py:69); }` + `Score { fn score/d_score/metric }`; natural gradient = `F⁻¹·g` (`scores.py:64`) — diagonal F for Normal/Poisson/HalfNormal makes the per-row solve O(n_params); only T/Gamma/MVN need a small dense solve.
- **Step-size control is one scalar per round** on the training-batch total score: double ≤256 then halve (`ngboost.py:270-297`); the fitted scalings must be stored on the ensemble and reused at predict (`:183-184`). Cheap to port; needs only score evaluation, no gradients.
- **Defaults worth copying:** depth-3 gradient trees (`learners.py:9`), `lr=0.01`, `n_estimators=500` (`ngboost.py:102-103`), optional per-round row/col subsampling (`:191-200`), and early stopping on validation total score with patience (`:563-575`) — `LeafKind::{Mean,Linear}` (`mltreelib-py/src/lib.rs:1962-1986`) extends naturally: linear leaves just hold a per-parameter `LinearModel` (`node_stats.rs:65-92`) producing the same increment.

**Bottom line for the mltreelib spec:** for a single tree predicting full distributions, fit the distribution *per leaf* (MLE params in a log-linked chart — a strict generalization of mean leaves, with linear leaves fitting per-param linear models over those chart coordinates). NGBoost's score/natural-gradient/line-search machinery is only the *boosting* half of the story; its leaves themselves are nothing but gradient means.