# Bayesian Tree Ensembles Landscape for `mltreelib`

**Date:** 2026-11-22
**Scope:** BART and XBART, Bayesian CART, monotone BART, variable selection under BART priors, GP and treed-GP hybrids, Dirichlet-process variants, causal and survival BART, and the Bayesian boosting literature. Deliberately separate from the bagging, boosting, and meta-estimator notes.
**Decision informed:** Whether the Bayesian tree family belongs in `mltreelib`, and if so which parts are reachable without an MCMC layer.
**Confidence:** High for the model specification, priors, and sampler mechanics. Explicitly marked list of unverified or non-existent citations at the end.
**Code changes:** None. Research and specification only.

## 1. Executive Summary

The direct question was: are BART and XBART a bagging method, a boosting method, or neither?

**Neither, in the algorithmic sense.** BART fits no bootstrap samples, so it is not bagging. It fits trees jointly rather than greedily and sequentially, so it is not boosting. It is a Bayesian **sum-of-trees** model fitted by Bayesian backfitting MCMC, producing a posterior over the whole sum.

That answer matters for the library, because it means BART is not a fourth family alongside `ModelForest`, `ModelBoost`, and `ModelMeta`. It is a different kind of object: an estimator whose primary output is a **distribution over functions**, not a point prediction with attached diagnostics.

**Finding 1: BART is the most-cited strong tabular model that the local Elo board does not yet rank against properly.** On the local 56-dataset arena, `bart` scores 1313.9 overall, 1243.3 regression Elo, 1228.6 classification Elo, with a 25.49 s mean fit time. That places it seventh of 47 entrants, ahead of ranger (1298.2), ahead of `ModelTree(all)` (1290.1), and ahead of every boosting implementation except CatBoost(200). It is the only entrant whose output is a posterior.

**Finding 2: most of BART's value is reachable without an MCMC layer.** BART's tree-structure prior, its leaf-shrinkage prior, and its error-variance prior are all *growth-criterion regularisers*. The library can adopt them as configuration over the existing split engine and get BART-shaped trees deterministically, with none of the posterior machinery. The full posterior, credible bands, and the variable-selection posterior all need MCMC, and those are out of scope for now.

**Finding 3: monotone BART's semantics already match this library's.** mBART constrains *leaf values*, not split rules, and every tree being constrained makes the sum monotone. The library's `leaf_monotone` already constrains leaf coefficients, so the audit is a semantics check rather than a new capability.

**Finding 4: six commonly cited Bayesian-tree papers do not exist as described.** "Bayesian boosting" (Geurst, van der Vaart, van der Velden, AoS 2021), "Safe Bayesian boosting" (Simin, Grinsztajn, JRSS-B 2022), the JCGS 2001 Friedman-Hastie-Tibshirani "Bayesian backfitting" paper, the JASA 2001 "Additive Bayesian Regression" paper, the Scheibler et al. spike-and-slab BART paper, and the Kim and Oh "EBART" paper. Verified substitutes appear in every case.

**Finding 5: XBART is not the BART posterior, and the official implementation is CPU-only.** Standalone XBART is a stochastic hill-climbing approximation that the authors themselves call "BART-inspired" rather than exact. The official repo has a C++ core, Rcpp and SWIG bindings, and a CPU thread pool, with no CUDA sources and no GPU build target. The GPU claim belongs to a separate line of work, `bartz`.

## 2. Citation Audit

The requested primaries do not all exist as described.

| Requested attribution | Status | Verified substitute |
|---|---|---|
| Chipman, McCulloch, Raftery (BART) | Wrong third author | **Chipman, George, McCulloch (2010)**, "BART: Bayesian additive regression trees," *AOAS* 4(1):266–298, DOI [10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285), arXiv:[0806.3286](https://arxiv.org/abs/0806.3286) |
| Friedman, Hastie, Tibshirani, JCGS 2001, "Bayesian backfitting" | **Does not exist** | **Hastie & Tibshirani (2000)**, *Statistical Science* 15(3):196–223, DOI [10.1214/ss/1009212815](https://doi.org/10.1214/ss/1009212815). Two authors, 2000, not JCGS, no Friedman. The actual JCGS Bayesian-tree paper is Wu, Tjelmeland & West (2007), DOI [10.1198/106186007X180426](https://doi.org/10.1198/106186007X180426) |
| "JASA 2001 Additive Bayesian Regression" | **Not verified** | Treat as non-existent. Closest real item is a 2001 book by Silverman |
| Scheibler, Carvalho, Tewari, Boyd, spike-and-slab BART, JMLR 2019 | **Not verified** | DART (Linero & Yang 2018, DOI [10.1111/rssb.12293](https://doi.org/10.1111/rssb.12293)) and ABC Bayesian forests (Liu, Ročková & Wang 2021, DOI [10.1111/rssb.12423](https://doi.org/10.1111/rssb.12423)) |
| EBART, Kim & Oh, JMLR 2024 | **Not verified** | Linero (2017) review, DOI [10.29220/CSAM.2017.24.6.543](https://doi.org/10.29220/CSAM.2017.24.6.543), and Goedhart et al. (2025) EBcoBART, DOI [10.1002/sim.70004](https://doi.org/10.1002/sim.70004) |
| Yuan & Lin (2009), treed GP / T-REX | **Wrong attribution** | **Gramacy & Lee (2008)**, "Bayesian treed Gaussian process models," *JASA* 103(483):1119–1130, DOI [10.1198/016214508000000689](https://doi.org/10.1198/016214508000000689), arXiv:[0710.4536](https://arxiv.org/abs/0710.4536). Yuan & Lin (2007) is a Gaussian graphical model paper. The 2009 "T-REX" paper is unrelated microbial software |
| Denison, Mallick, Rao, "Bayesian backfitting" | **Wrong third author** | **Denison, Mallick, Smith (1998)**, "A Bayesian CART algorithm," *Biometrika* 85(2):363–377, DOI [10.1093/biomet/85.2.363](https://doi.org/10.1093/biomet/85.2.363) |
| "Bayesian boosting," Geurst, van der Vaart, van der Velden, AoS 2021 | **Does not exist** | The described mechanics split into arXiv:[2206.05608](https://arxiv.org/abs/2206.05608) (boosting as GP posterior inference) and BART itself (shrinkage prior plus Gibbs) |
| "Safe Bayesian boosting," Simin, Grinsztajn, JRSS-B 2022 | **Does not exist** | Grünwald (2012), DOI [10.1007/978-3-642-34106-9_16](https://doi.org/10.1007/978-3-642-34106-9_16); de Heide et al. (2020), [PMLR 108](https://proceedings.mlr.press/v108/heide20a.html); Malinin et al., arXiv:[2006.10562](https://arxiv.org/abs/2006.10562) |
| Künzel, Sekhon, Bickel, Pollard, "Nonparametric survival analysis for causal inference," Biometrics 2021 | **Not verified** | Both offered DOIs resolve to other papers. Substitute Sparapani et al. (2016), DOI [10.1002/sim.6893](https://doi.org/10.1002/sim.6893) |
| `dbbart` package, `PyBART` Bayesian-trees package | **Do not exist** | `dbarts` (Dorie, Chipman, McCulloch). PyPI `pybart` is a transit CLI |
| `BayesTree` by "Kass, Wu, Gao" | **Wrong attribution** | `BayesTree` is Chipman & McCulloch |
| `bartMachine` by "Wada & Jiang" | **Wrong authors** | bartMachine is Kapelner & Bleich. "Jiang" maps to BARTSIMP, a spatial BART application |

The `Bayesian boosting` negative is worth spelling out because it is thorough: the exhaustive Crossref listing of *Annals of Statistics* 2021 (151 works) contains no title with "Bayesian boosting" and none combining "Bayesian" with "boost"; OpenAlex `title.search` restricted to the AoS source returns zero; arXiv author searches for Geurst and van der Velden return zero; and van der Vaart's own publication list has no such paper. The OpenAlex author lookup for "Geurst" returns only a condensed-matter physicist.

One arXiv-id trap worth recording: arXiv:1904.01730 is **not** a BART paper. It resolves to Alvi et al., "Sequencing and Scheduling for Multi-User Machine-Type Communication". Every arXiv id in this note was re-verified by title and authors.

## 3. BART: The Model

### 3.1 Specification

Continuous response:

$$Y = \sum_{j=1}^{m} g(X; T_j, M_j) + \varepsilon, \qquad \varepsilon \sim N(0, \sigma^2).$$

Each $T_j$ is a binary tree of split rules. Each terminal node $i$ of tree $j$ carries a scalar leaf mean $\mu_{ij}$. The joint prior factorises:

$$p(T_1, M_1, \ldots, T_m, M_m, \sigma) = \prod_{j=1}^{m} p(T_j)\, p(M_j \mid T_j)\, p(\sigma).$$

Binary response uses a probit link, $P(Y=1 \mid x) = \Phi(f(x))$.

Source: Chipman, George & McCulloch (2010), [author PDF](https://www.acadiau.ca/~hchipman/papers/AOAS285.pdf).

### 3.2 Priors, with the correction that matters

**Leaf means are Gaussian, not scaled inverse chi-square.**

$$\mu_{ij} \mid T_j \sim N(\mu_\mu, \sigma_\mu^2), \qquad \mu_\mu = 0, \qquad \sigma_\mu = \frac{0.5}{k\sqrt{m}},$$

after shifting and scaling $Y$ to $[-0.5, 0.5]$. The leaf-mean variance scales as $1/m$, **not** as $1/n_j$ and **not** as a function of depth. $k$ is the prior-SD count: $k = 2$ means the prior puts 95% of its mass on $E(Y \mid x) \in (-0.5, 0.5)$ for a numeric response.

**There is no inverse-gamma leaf-variance prior in standard BART.** Leaf-specific inverse-gamma variances belong to the *ancestor* single-tree model, Bayesian CART model search, where $\mu_i \mid \sigma_i^2, T \sim N(\bar\mu, \sigma_i^2/a)$ and $\sigma_i^2 \mid T \sim IG(\nu/2, \nu\lambda/2)$. Hierarchical CART shrinkage likewise gives bottom-node variances inverse-gamma priors with depth-indexed shifts.

**Scaled inverse chi-square is the error-variance prior, not a leaf prior.**

$$\sigma^2 \sim \frac{\nu\lambda}{\chi_\nu^2} \;\equiv\; IG\!\left(\frac{\nu}{2}, \frac{\nu\lambda}{2}\right).$$

Calibration: pick a rough $\hat\sigma$ (the sample or least-squares residual standard deviation), pick $\nu \in [3, 10]$, choose $\lambda$ so that $P(\sigma < \hat\sigma) = q$. The paper's settings are conservative $(\nu, q) = (10, 0.75)$, **default $(3, 0.90)$**, and aggressive $(3, 0.99)$. The authors reject $\nu < 3$ because it places too much mass near $\sigma = 0$ and encourages overfitting. Neither $\nu$ nor $\lambda$ depends on leaf sample size or tree depth.

### 3.3 Where depth and leaf sample size actually enter

This is the precise mechanism, and it differs from the common description.

The **depth** $d$ enters only the tree-size prior. The **leaf sample size** $n_j$ enters only the posterior precision:

$$\mu_j \mid T, R, \sigma^2 \sim N\!\left(\frac{S_j/\sigma^2}{n_j/\sigma^2 + 1/\sigma_\mu^2},\; \frac{1}{n_j/\sigma^2 + 1/\sigma_\mu^2}\right).$$

So small leaves shrink harder *through $n_j$*, not through a depth-scaled leaf prior.

The **tree-structure prior**, at depth $d = 0, 1, \ldots$:

$$P(\text{node is nonterminal} \mid d) = \alpha(1+d)^{-\beta}, \qquad \alpha \in (0,1), \ \beta \in [0,\infty).$$

Default $\alpha = 0.95$, $\beta = 2$. The splitting variable is uniform over available variables, and the splitting value is uniform over available discrete split values. That gives monotone-transformation invariance, which is not full Bayesian coherence. Prior probability on tree sizes $1, 2, 3, 4, \ge 5$ leaves is $0.05, 0.55, 0.28, 0.09, 0.03$.

### 3.4 The sampler

The paper: "At a general level, our algorithm is a Gibbs sampler." For tree $j$, with residual $R_j = y - \sum_{k \neq j} g(x; T_k, M_k)$:

1. $T_j \mid R_j, \sigma$ via **Metropolis-Hastings with the leaf means integrated out**, using the marginal likelihood $\int p(R_j \mid M_j, T_j, \sigma)\, p(M_j \mid T_j, \sigma)\, dM_j$.
2. $M_j \mid T_j, R_j, \sigma$ via independent Gaussian leaf draws.
3. After all trees, $\sigma \mid \cdots, y$ from its inverse-gamma full conditional.

**The move set is four moves, not three.** Grow a terminal node, probability 0.25; prune a pair of terminal nodes, 0.25; change a nonterminal split rule, 0.40; swap parent and child rules, 0.10. The same values appear in the 2006 precursor.

The acceptance ratio:

$$\alpha(T, T^*) = \min\!\left\{1,\; \frac{q(T \mid T^*)\, p(T^*)\, p(Y \mid X, T^*)}{q(T^* \mid T)\, p(T)\, p(Y \mid X, T)}\right\}.$$

With $s_d = \alpha(1+d)^{-\beta}$, growing a terminal at depth $d$ contributes a structural prior ratio $\frac{s_d}{1-s_d}(1-s_{d+1})^2$, and pruning gives the reciprocal. The full acceptance must also carry the marginal-likelihood ratio **and** the forward and reverse proposal ratio. Dropping the proposal ratio makes the proposal non-reversible.

**Initialisation:** "We initialize the chain with $m$ simple single node trees." The authors report that restarts give similar results, so they "run one long chain with BART rather than multiple starts."

### 3.5 Burn-in, thinning, and sample counts

| Source | Burn-in | Retained draws | Thinning |
|---|---:|---:|---:|
| BART paper bake-off | 200 | 1000 | — |
| BART paper, Friedman-function example | 1000 | 5000 | — |
| `BayesTree` and `BART` package defaults | 100 (`nskip`) | 1000 (`ndpost`) | 1 (`keepevery`) |
| `dbarts::bart2` formula API | 500 (`n.burn`) | 500 (`n.samples`) | 1 (`n.thin`) |

The package burn-in default of 100 differs from the paper's 200. `bart2` also uses 4 chains by default.

## 4. Hyperparameter Names Across Sources

The same concept has a different name in every package. This table is the practical reference.

| Concept | `BayesTree` / `BART` | `dbarts` | `bartMachine` | `XBART` | Meaning |
|---|---|---|---|---|---|
| Tree prior base | `base=0.95` | `base=0.95` | `alpha=0.95` | `alpha=0.95` | Nonterminal probability at depth 0 |
| Tree prior exponent | `power=2.0` | `power=2.0` | `beta=2` | `beta=1.25` | Depth decay |
| Leaf shrinkage | `k=2.0` | `k=2.0` | `k=2` | n/a (`tau`) | Prior SDs of $E(Y\mid x)$ from the boundary |
| Error variance | `sigest`+`sigdf`+`sigquant` | same | `nu=3`, `q=0.9` | `kap=16`, `s=4` | Inverse-gamma shape and scale |
| Variables per split | **absent** (uniform prior) | `splitprobs=1/numvars` | `cov_prior_vec` | `mtry=NULL` (all) | Split-variable weighting |
| Trees in the sum | `ntree=200L` | `ntree=200` / `n.trees=75L` | `num_trees=50` | required | Ensemble size |
| Burn-in | `nskip=100` | `nskip=100` / `n.burn=500L` | `num_burn_in=250` | `burnin=1L` (R), 15 (Py) | Discarded iterations |
| Retained draws | `ndpost=1000L` | `ndpost=1000` / `n.samples=500L` | `num_iterations_after_burn_in=1000` | `num_sweeps` required | Posterior draws |
| Thinning | `keepevery=1` | `keepevery=1` / `n.thin=1L` | n/a | n/a | Keep every $k$-th |
| Candidate cuts | `numcut=100L` | `numcut=100` | n/a | `num_cutpoints=100` | Split-value grid |

Paper-recommended values: $\alpha = 0.95$, $\beta = 2$, $k = 2$, $(\nu, q) = (3, 0.90)$, $m = 200$.

The `BART` package's exact `wbart` defaults: `sparse=FALSE, theta=0, omega=1, a=0.5, b=1, augment=FALSE, rho=NULL, xinfo=matrix(0.0,0,0), usequants=FALSE, cont=FALSE, rm.const=TRUE, sigest=NA, sigdf=3, sigquant=.90, k=2.0, power=2.0, base=.95, sigmaf=NA, lambda=NA, fmean=mean(y.train), w=rep(1,length(y.train)), ntree=200L, numcut=100L, ndpost=1000L, nskip=100L, keepevery=1L, printevery=100L, transposed=FALSE`.

`bartMachine`'s defaults: `num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000, alpha=0.95, beta=2, k=2, q=0.9, nu=3, prob_rule_class=0.5, mh_prob_steps=c(2.5,2.5,4)/9, s_sq_y="mse", num_rand_samps_in_library=10000, use_missing_data=FALSE, mem_cache_for_speed=TRUE, flush_indices_to_save_RAM=TRUE, use_xoshiro=FALSE`. The `mh_prob_steps` are the grow, prune, and change probabilities $(2.5, 2.5, 4)/9$.

`dbarts`'s internal proposal mix: `birth_death` 0.5, `change` 0.1, `swap` 0.4, and within `birth_death`, `birth` 0.5.

## 5. XBART

### 5.1 Verified publication

He, Jingyu; Yalov, Saar; Hahn, P. Richard (2019), "XBART: Accelerated Bayesian Additive Regression Trees," AISTATS 2019, *PMLR* 89:1130–1138, [record](https://proceedings.mlr.press/v89/he19a.html), [PDF](https://proceedings.mlr.press/v89/he19a/he19a.pdf), [supplement](https://proceedings.mlr.press/v89/he19a/he19a-supp.pdf), arXiv:[1810.02215](https://arxiv.org/abs/1810.02215).

Expanded version: He & Hahn (2023), "Stochastic tree ensembles for regularized nonlinear regression," *JASA* 118(541):551–570, DOI [10.1080/01621459.2021.1942012](https://doi.org/10.1080/01621459.2021.1942012), arXiv:[2002.03375](https://arxiv.org/abs/2002.03375).

The "Chen & Vadapalli" attribution is wrong. arXiv:1803.09583 is an unrelated "Fast Bayesian Additive Regression Trees" preprint.

### 5.2 What changes

**The proposal.** XBART discards the current tree and grows a replacement from the root each sweep. At each node it evaluates all eligible cutpoints under BART's integrated marginal likelihood, adds one null "no split" candidate, samples split-or-null *probabilistically* rather than by argmax, and recurses. The split criterion:

$$L(c) \propto \exp\!\left[\tfrac{1}{2}\sum_{b=1}^{2} n\left\{\log\frac{\sigma^2}{\sigma^2 + \tau n_b} + \frac{\tau s_b^2}{\sigma^2(\sigma^2 + \tau n_b)}\right\}\right].$$

The null candidate's weight is $|\mathcal{C}|\left((1+d)^\beta/\alpha - 1\right)\times(\text{leaf term})$, so when marginal likelihoods tie, $P(\text{split}) = 1 - \alpha(1+d)^{-\beta}$ recovers the BART tree prior.

**The data structure.** A predictor order matrix $O \in \{1..n\}^{V \times n}$ gives prefix residual sums $s(\le, v, c) = \sum_{h \le c} r_{o_{vh}}$. After each split, "sifting" partitions the other variables' sorted orders into left and right arrays, avoiding a re-sort. The cutpoint grid is adaptive: only $C$ cutpoints per variable, $C = \min(n, 100)$ by default, with eligible index step $j = \lfloor (n_b - 2)/C \rfloor$.

**Sparse variable proposal.** $m \le V$ variables are sampled per split without replacement, proportional to an adaptive Dirichlet weight $w$. Split counts update the Dirichlet hyperparameters, and all variables are used during burn-in to estimate $w$.

### 5.3 Reported speed and sample counts

The AISTATS design used $d = 30$ predictors and $n \in \{10{,}000, 50{,}000, 250{,}000\}$ with 5 replications. XBART's recommendation is $L \propto (\log n)/\log\log n$, $\alpha = 0.95$, $\beta = 1.25$, $\tau = 0.3\,\mathrm{Var}(y)/L$, **$K = 40$ sweeps with $I = 15$ burn-in**. The BART comparison at $n = 10{,}000$ used 200 trees, $\alpha = 0.95$, $\beta = 2$, **5,000 burn-in and 2,000 retained samples**.

XBART runtimes in seconds, from Table 3: $n = 10$k gives 20/16/17/17 and $n = 250$k gives 1,774/1,188/1,484/1,324 for Linear, Max, Single index, and Trig+poly respectively.

The supplement at $n = 10{,}000$ reports XBART at 16 to 20 s against `dbarts` at 110 to 117 s, summarised as **approximately 10× faster**, with broadly similar RMSE.

Across 40 real datasets and 800 fits: average relative RMSE was BART 1.067, XBART 1.128, XGB with CV 1.137, RF 1.139, XGB 1.170. Average runtime was BART 16.930 s against XBART 5.680 s, roughly 3×, with RF 0.385 s and XGB 0.117 s.

Warm-start BART, from the expanded paper: XBART with 40 sweeps and the first 15 discarded, BART with 1,000 burn-in and 2,500 retained, and WS-BART as 25 chains of 100 retained with no burn-in. Nominal 95% coverage for Linear $\kappa = 1$ was XBART 0.78, BART 0.77, **WS-BART 0.99**; for Linear $\kappa = 2$, 0.50, 0.83, and **0.98**. A later characterisation summarises prior XBART acceleration as at most 20 to 30× while emphasising that XBART changed the algorithm.

### 5.4 Is XBART the BART posterior?

**No, for standalone XBART.** The authors describe it as stochastic hill climbing, and the 2019 paper calls it a greedy stochastic approximation when the exact full conditional is absent. Only 40 sweeps are recommended, with a warning that posterior uncertainty may be understated. Petrillo calls XBART a "BART-inspired non-Bayesian algorithm". Stationarity of the ensemble chain and single-tree consistency do **not** establish posterior equality.

The 2019 paper does define a corrected full-forest Metropolis-Hastings variant that regrows all $L$ trees, computes the joint proposal density, and accepts or rejects the whole sweep, but states that efficacy evaluation remained future work.

Warm-start BART *does* target the BART posterior.

### 5.5 The GPU claim is false for official XBART

The official repository [JingyuHe/XBART](https://github.com/JingyuHe/XBART) has a C and C++ core, Rcpp and SWIG bindings, and a CPU thread pool in `src/thread_pool.cpp`. There are no CUDA sources and no GPU build target, and neither paper reports GPU work.

GPU BART is a separate line: Petrillo, "Very fast Bayesian Additive Regression Trees on GPU," arXiv:[2410.23244](https://arxiv.org/abs/2410.23244), package `bartz`. Its design uses fixed depth $D = 5$ with perfect breadth-first heap arrays, quantised predictors, and a leaf-index cache whose removal slows training by 2 to 3×. Only GROW and PRUNE moves are retained, on JAX and XLA, with no custom CUDA. Petrillo proposes GPU XBART as future work only.

## 6. Packages: What Actually Exists

| Requested | Verified reality |
|---|---|
| R package `bart` by Friedman | **No CRAN package named `bart`.** Two real packages: `BayesTree` (Chipman & McCulloch, first CRAN release 2006-01-05) and `BART` (McCulloch & Sparapani, first CRAN 2017-04-07). Friedman appears only as a cited reference and as the origin of the standard simulated DGP $10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5$ |
| `BayesTree` by Kass, Wu, Gao | **Wrong.** `BayesTree` is Chipman & McCulloch. The Kass, Wu & Gao ORDB 2014 paper is a Bayesian residual-error tree paper, a different method |
| `bartMachine` by Wada & Jiang | **Wrong authors.** bartMachine is Kapelner & Bleich. "Wada" has zero BART records. "Jiang" maps to BARTSIMP, a spatial BART application |
| `dbbart` | **No such package.** CRAN and PyPI both 404. The real package is `dbarts` |
| `PyBART` | PyPI `pybart` is a **BART transit CLI**. `pybart-nlp` is a UD-tree-to-BART syntactic-graph converter. Real Python BART packages are `bartpy`, `bartz`, `rbartpackages`, `xbart`, and `stochtree` |
| `XBART` R and Python | **Correct**, but not on CRAN. `install_github("JingyuHe/XBART")`, PyPI `xbart` 0.1.8. The maintained successor is CRAN `stochtree` 0.4.5, repo [StochasticTree/stochtree](https://github.com/StochasticTree/stochtree), which provides grow-from-root BART plus BCF in R and Python |

**Package capability summary.**

- **`BayesTree`** (R and C/C++, 0.3-1.5): pure BART MCMC, probit binary, dummy expansion with all levels kept, 1-D and 2-D partial dependence, `plot.bart` showing the sigma trace and posterior interval plot. **No monotone constraints, no variable selection, no missing-data handling.**
- **`BART`** (2.9.10): no `bart()` function. Entry points `wbart`, `gbart` with sparse Dirichlet variable selection, `pbart` for probit, `lbart` for logit, `mbart` for multinomial, `abart`/`surv.bart`/`recur.bart`/`crisk.bart`/`crisk2.bart`, `pwbart` plus `predict.*`, and `mc.*` parallel twins. Adds survival, competing risks, and recurrent-event BART, the sparse Dirichlet split prior, and data augmentation.
- **`dbarts`** (0.9-34): positioned as BART usable *inside* another Gibbs or Metropolis-Hastings sampler, through the `dbartsSampler` S4 class. Supports a `k` hyperprior, multi-chain operation, and `extract(type = "trees")` giving `sample, chain, tree, n, var, value` in depth-first order with `var = -1` marking a leaf.
- **`bartMachine`** (1.4.2): Java and TornadoVM GPU backend. Adds variable selection, interaction investigation, credible and prediction intervals, convergence diagnostics, $k$-fold CV, missing-data handling, and interaction constraints.
- **`XBART`** (1.1): R defaults `max_depth=250, Nmin=1, num_cutpoints=100, alpha=0.95, beta=1.25, tau=NULL, no_split_penalty=NULL, burnin=1L, mtry=NULL, p_categorical=0L, kap=16, s=4, tau_kap=3, tau_s=0.5, verbose=FALSE, update_tau=TRUE, parallel=TRUE, random_seed=NULL, sample_weights=TRUE, nthread=0`, with `num_trees` and `num_sweeps` required. The R default `burnin = 1L` differs from the paper's recommendation of 15. The Python constructor (`xbart` 0.1.8) uses `num_trees=100, num_sweeps=40, n_min=1, num_cutpoints=100, alpha=0.95, beta=1.25, tau="auto", burnin=15, mtry="auto", max_depth=250, kap=16.0, s=4.0`.

**Adjacent CRAN family**: `bartisan` for generalised non-Gaussian likelihoods via a Laplace-approximation reversible-jump sampler, `bartCause`, `stan4bart`, `stochtree`, `nftbart` for nonparametric failure-time BART, and `bartcs` for confounder selection.

## 7. Variable Selection Under Bayesian Tree Priors

Three distinct mechanisms, all verified.

### 7.1 Inclusion-proportion thresholding

Fit BART; permute the response $L$ times preserving predictor dependence; refit on each null response; compare observed inclusion proportions against null quantiles. The rules are a local threshold, a global maximum, and a global standard error.

Source: Bleich, Kapelner, Scheibe & Vavrek (2014), arXiv:[1310.4887](https://arxiv.org/abs/1310.4887). Verified defaults in Luo & Daniels (2024), *Statistical Science* 39(2):286–304, DOI [10.1214/23-STS900](https://doi.org/10.1214/23-STS900): $L = 100$ permuted-response fits and $L_{\text{rep}} = 10$ original-data fits, with the DART median-probability model selecting predictors whose marginal posterior variable-inclusion probability is at least 0.5.

### 7.2 Shrinkage at the split level: DART

$$s = (s_1, \ldots, s_P) \sim \text{Dirichlet}(a/P^\xi, \ldots, a/P^\xi), \qquad \xi \ge 1,$$

with hyperprior $\frac{a}{a+\lambda_a} \sim \text{Beta}(a_a, b_a)$ and $(a_a, b_a, \lambda_a) = (0.5, 1, P)$. The conjugate update adds branch counts. Frequent predictors gain probability, so the mechanism is "rich get richer". For $\xi = 1$ the active-predictor count is approximately $1 + \text{Poisson}(\theta)$ with $\theta = a\sum_{i=1}^{B-1}(a+i)^{-1}$. The theory needs $\xi > 1$; $\xi = 1$ works in practice.

Source: Linero & Yang (2018), *JRSS-B* 80(5):1087–1110, DOI [10.1111/rssb.12293](https://doi.org/10.1111/rssb.12293), arXiv:[1707.09461](https://arxiv.org/abs/1707.09461).

### 7.3 Spike-and-forest ABC Bayesian forests

A prior over active sets $S$; a forest prior conditional on $S$; variables outside $S$ cannot enter splits; posterior inclusion probability $\Pr(j \in S \mid Y)$ gives model-based selection. The ABC sampler trains on a random subset and scores the held-out subset.

Source: Liu, Ročková & Wang (2021), *JRSS-B* 83(3):453–481, DOI [10.1111/rssb.12423](https://doi.org/10.1111/rssb.12423), arXiv:[1806.02304](https://arxiv.org/abs/1806.02304).

### 7.4 Empirical-Bayes hyperparameter learning

Monte Carlo EM maximises the BART marginal likelihood; split-probability weights come from posterior split usage; an optional logistic co-data model compresses the $p$ covariate weights to $\kappa + 1$ parameters. The same framework estimates $\alpha, \beta, k, \nu, \lambda$ as a cross-validation alternative.

Source: Goedhart, Klausch, Janssen & van de Wiel (2025), *Statistics in Medicine* 44(5):e70004, DOI [10.1002/sim.70004](https://doi.org/10.1002/sim.70004), arXiv:[2311.09997](https://arxiv.org/abs/2311.09997).

## 8. Monotone BART: A Semantics Match

mBART imposes monotonicity on a predictor subset $S$: a leaf value may not exceed the minimum above-neighbour value, nor fall below the maximum below-neighbour value, for each constrained coordinate. Every tree being constrained makes the sum monotone. The prior support is restricted to admissible leaf vectors.

**The original leaf conjugacy is lost.** The Metropolis-Hastings sampler must sample constrained leaf levels rather than draw them conjugately. A variance inflation factor $c^2 = \pi/(\pi-1) \approx 1.4669$ compensates for the order constraint. The tree prior defaults remain $\alpha = 0.95$, $\beta = 2$, and the error prior $(\nu, q) = (3, 0.90)$.

Source: Chipman, George, McCulloch & Shively (2022), *Bayesian Analysis* 17(2):515–544, DOI [10.1214/21-BA1259](https://doi.org/10.1214/21-BA1259), arXiv:[1612.01619](https://arxiv.org/abs/1612.01619).

**Why this matches the library.** mBART constrains *leaf values*, not split rules. `mltreelib`'s `leaf_monotone` already constrains leaf coefficients, and `tree_monotone` constrains split direction. The mBART semantics is therefore an **audit** of the existing constraint, not a new capability: the check is that a monotone-boosted or monotone-fitted model satisfies the constraint on the summed or fitted prediction, and that the variance inflation is documented when the constraint binds.

This is the same finding that reached Spec 29's Decision 4 from the opposite direction.

## 9. Bayesian Trees Beyond BART

### 9.1 Bayesian CART

- Precursor: Chipman, George & McCulloch (1997), "A Bayesian approach to CART," *PMLR* R1:91–102, [record](https://proceedings.mlr.press/r1/chipman97a.html), no DOI.
- Canonical: Chipman, George & McCulloch (1998), "Bayesian CART model search," *JASA* 93(443):935–948, DOI [10.1080/01621459.1998.10473750](https://doi.org/10.1080/01621459.1998.10473750). The model is $p(T, \theta \mid X, Y) \propto p(Y \mid X, \theta, T)\, p(\theta \mid T)\, p(T)$; the tree prior is generative; MH moves are grow, prune, change rule, and swap rules; acceptance is the posterior ratio times the proposal ratio.
- Denison, Mallick & Smith (1998), *Biometrika* 85(2):363–377, DOI [10.1093/biomet/85.2.363](https://doi.org/10.1093/biomet/85.2.363): reversible-jump MCMC changes tree dimension.
- Wu, Tjelmeland & West (2007), *JCGS* 16(1):44–66, DOI [10.1198/106186007X180426](https://doi.org/10.1198/106186007X180426): a "pinball" prior jointly controls tree size and shape.

### 9.2 Bayesian backfitting in general

Hastie & Tibshirani's model is $y_i = \alpha + \sum_j f_j(x_{ij}) + \varepsilon_i$ with identifiability $\sum_i f_j(x_{ij}) = 0$. Algorithm 3.1 defines the partial residual $r_j^t = y - \sum_{k<j} f_k^t - \sum_{k>j} f_k^{t-1}$, draws $z_j^t \sim N(0,1)$, updates $f_j^t \leftarrow S_j r_j^t + \sigma S_j^{1/2} z_j^t$, and cycles $j = 0..p$ until the joint distribution stabilises. $S_j$ is a user-chosen linear smoother and $S_j^{1/2}$ its matrix square root. There are no tree priors, no leaf parameters, and no grow/prune moves.

### 9.3 Interaction detection

Marvald & Love (2024), "Detecting interactions using Bayesian additive regression trees," *Pattern Analysis and Applications* 27(4):153–168, DOI [10.1007/s10044-024-01357-x](https://doi.org/10.1007/s10044-024-01357-x), software `BARTselect`. The inclusion proportion is $p_i$, the conditional inclusion $p_{i \mid j}$, and the conditional marginal difference $d_{i \mid j} = p_{i \mid j} - p_i$. A pair is flagged only when both conditional inclusions and both positive CMDs exceed permutation-calibrated thresholds, with local and global threshold variants.

### 9.4 GP and treed-GP hybrids

**Treed GP** is Gramacy & Lee (2008), *JASA* 103(483):1119–1130, DOI [10.1198/016214508000000689](https://doi.org/10.1198/016214508000000689), arXiv:[0710.4536](https://arxiv.org/abs/0710.4536). The tree partitions the input; each leaf carries its own Gaussian process plus a regional linear trend; tree structure and leaf GP parameters are sampled jointly with reversible-jump MCMC over grow, prune, change, swap, and rotate.

**GP-BART** is Maia, Murphy & Parnell (2024), *CSDA* 190:107858, DOI [10.1016/j.csda.2023.107858](https://doi.org/10.1016/j.csda.2023.107858), arXiv:[2204.02112](https://arxiv.org/abs/2204.02112). Each terminal node gets a GP over its observations, replacing the constant leaf, with node-specific stationary anisotropic exponentiated-quadratic covariance and leaf means marginalised. Rotated splits are supported with 20 angles in $[0, \pi]$. Move probabilities are grow-rotate 0.3, change-rotate 0.4, prune 0.3. Defaults are 20 trees, 3,500 MCMC iterations, 1,500 burn-in.

**Local GP extrapolation** is Wang, He & Hahn (2024), *JCGS* 33(2):724–735, DOI [10.1080/10618600.2023.2240384](https://doi.org/10.1080/10618600.2023.2240384), arXiv:[2204.10963](https://arxiv.org/abs/2204.10963). The standard BART posterior is unchanged; a training-data hypercube per leaf defines interior and exterior; exterior test points get a leaf-local GP predictive while interior points keep the constant leaf prediction. The kernel is squared-exponential with default smoothness $\theta = 0.1$ and GP scale $\tau = \mathrm{Var}(y)/L$; only active variables count; at most 100 leaf training observations are sampled; the hypercube uses 95% training quantiles.

**Targeted smoothing BART** is Starling, Murray, Carvalho, Bukowski & Scott (2020), *AOAS* 14(1):28–50, DOI [10.1214/19-AOAS1268](https://doi.org/10.1214/19-AOAS1268), arXiv:[1805.07656](https://arxiv.org/abs/1805.07656).

### 9.5 Dirichlet-process variants

**DPMBART** is George, Laud, Logan, McCulloch & Sparapani (2019), *Advances in Econometrics* 40B:89–110, DOI [10.1108/S0731-90532019000040B006](https://doi.org/10.1108/S0731-90532019000040B006), arXiv:[1807.00068](https://arxiv.org/abs/1807.00068). The sum-of-trees structure is retained for the mean, and the **error law** is replaced by a Dirichlet-process mixture of normals, $\varepsilon_i \sim N(\mu_i, \sigma_i^2)$ with $(\mu_i, \sigma_i) \sim G$ and $G \sim DP(G_0, \alpha)$.

There is **no verified canonical paper** for a Dirichlet-process prior placing separate infinite mixtures inside each BART leaf. The nearest ingredients are DPMBART (a DP over the errors), Hannah et al. (2011) "Dirichlet process mixtures of generalized linear models," *JMLR* 12:307–339, and a DP-over-trees construction with Gaussian-process splits.

### 9.6 Bayesian threshold selection

Two distinct meanings. Thresholding noisy signals using tree partitions is Evers & Heaton (2009), *JCGS* 18(4):961–977, DOI [10.1198/jcgs.2009.07109](https://doi.org/10.1198/jcgs.2009.07109), with package `treethresh`. Bayesian extreme-value threshold selection is Lee, Fan & Sisson (2015), *CSDA* 85:84–99, DOI [10.1016/j.csda.2014.12.004](https://doi.org/10.1016/j.csda.2014.12.004).

## 10. Causal, Survival, and Mediation

### 10.1 Causal BART

**Hahn, P. Richard; Murray, Jared S.; Carvalho, Carlos M. (2020)**, "Bayesian regression tree models for causal inference: regularization, confounding, and heterogeneous effects," *Bayesian Analysis* 15(3):965–1056, DOI [10.1214/19-ba1195](https://doi.org/10.1214/19-ba1195).

**The widely repeated DOI `10.1214/19-BA1231` is wrong.** It 404s. Software: CRAN `bcf` (Murray, Hahn, Carvalho, DOI [10.32614/cran.package.bcf](https://doi.org/10.32614/cran.package.bcf)) and `bartCause` (Dorie, Hill, DOI [10.32614/cran.package.bartcause](https://doi.org/10.32614/cran.package.bartcause)). The outcome-model route is Hill (2011), *JCGS* 20(2):217–240, DOI [10.1198/jcgs.2010.08162](https://doi.org/10.1198/jcgs.2010.08162).

### 10.2 Survival BART

**Sparapani, Rodney A.; Logan, Brent R.; McCulloch, Robert E.; Laud, Paul W. (2016)**, "Nonparametric survival analysis using Bayesian additive regression trees," *Statistics in Medicine* 35(16):2741–2753, DOI [10.1002/sim.6893](https://doi.org/10.1002/sim.6893). Software `BART::surv.bart`, plus `abart` for AFT, `recur.bart`, and `crisk.bart`.

Follow-up for heterogeneous survival treatment effects: Hu, Ji & Li (2021), *Statistics in Medicine*, DOI [10.1002/sim.9090](https://doi.org/10.1002/sim.9090).

### 10.3 Mediation BART

- Liu, Qin, Talisa & Wang (2026), *Biometrics* 82, DOI [10.1093/biomtc/ujag079](https://doi.org/10.1093/biomtc/ujag079).
- Yu & Li (2026), *Environmetrics* 37, DOI [10.1002/env.70125](https://doi.org/10.1002/env.70125).
- Ting & Linero (2023), arXiv:[2303.01620](https://arxiv.org/abs/2303.01620).
- Linero & Zhang (2025), *Psychological Methods* 30(1):60–82, DOI [10.1037/met0000504](https://doi.org/10.1037/met0000504).
- Kim, Daniels, Marcus & Roy (2017), *Biometrics* 73(2):401–409, DOI [10.1111/biom.12575](https://doi.org/10.1111/biom.12575).

## 11. Is BART Bagging, Boosting, or Neither?

**Neither, in the algorithmic sense. Its closest relative is boosting by motivation only.**

**Not bagging.** Bagging fits independent trees on bootstrap resamples and averages them. BART fits no bootstrap samples. Its trees are conditionally dependent through the shared residual $R_j$ in the backfitting Gibbs sweep, and its leaves are shrunk by a prior rather than by resampling variance reduction. The BART paper explicitly contrasts itself with bagging and random forests, whose trees are independent and whose variance reduction comes from randomization.

**Not boosting in algorithm.** Boosting adds trees greedily and sequentially, each fit to the residual of the previous ensemble, and never revisits earlier trees. BART samples all trees jointly: each tree's conditional given the others, then the leaf values, then $\sigma$, repeatedly. The paper says "At a general level, our algorithm is a Gibbs sampler" and "Motivated by ensemble methods in general, and boosting algorithms in particular, BART is defined by a statistical model: a prior and a likelihood." Both "Bayesian backfitting" and "boosting" appear in its keywords as motivation, not mechanism.

**Correct label:** a Bayesian **sum-of-trees** model fitted by Bayesian backfitting MCMC, an additive ensemble whose members are jointly updated weak learners, producing a posterior over the whole sum.

**What this means for the library.** BART is not a fourth ensemble family next to `ModelForest`, `ModelBoost`, and `ModelMeta`. It is a different kind of estimator: one whose primary output is a distribution over functions. If `mltreelib` ever exposes a Bayesian tree family, it is a **fourth task facade**, not a fourth `kind` value, because the output contract differs fundamentally from a point predictor with diagnostics.

## 12. What the Posterior Adds, and What It Costs

### 12.1 What a single fit cannot give

- A full posterior over the regression function $f$, giving credible intervals for $f(x)$ and credible bands for the function.
- A posterior over $\sigma$, giving predictive intervals that separate aleatoric from epistemic uncertainty. `BART` returns `sigma` and `first.sigma` draws.
- A posterior over which predictors matter: per-draw split counts (`varcount`), inclusion proportions, and their permutation-calibrated null distribution, plus posterior inclusion probabilities under DART or spike-and-forest.
- A posterior over model structure: how many trees, which splits, which depths.
- Stability across draws: variability of $f^{(s)}$ over draws, tree-wise contribution variability, and posterior overlap of partial-dependence curves.

### 12.2 Costs

Each retained draw requires a full sweep over $m$ trees with Metropolis-Hastings structure moves, conjugate leaf updates, and a variance draw. Runtime scales with draws times trees times proposals per tree, and multi-chain runs multiply that by the chain count. Thinning trades draws for memory but does not increase effective sample size; thinning discards information, so effective samples per wall-clock second are *lower* with thinning. Convergence diagnosis therefore requires thinning-aware diagnostics, and credible intervals from few effective draws understate uncertainty.

### 12.3 Available diagnostics

- `BART::gewekediag(x, frac1=0.1, frac2=0.5)` — Geweke's test of equality of means between the first 10% and last 50% of the chain, with a spectral-density standard error, asymptotic standard normal under stationarity.
- `plot.bart` and `plot.pdbart` — two panels: the kept $\sigma$ draws including burn-in, and $y$ against posterior intervals for $f(x)$.
- Multi-chain: `dbarts` `nchain`/`nthread`/`combinechains`; `BART` `mc.*` parallel twins; `bartMachine` `plot_convergence_diagnostics` and `rmse_by_num_trees`.

**R-hat and ESS are not computed by the `BART` or `dbarts` packages.** No built-in split-R-hat or ESS function was verified. `bartMachine`'s `plot_convergence_diagnostics` is the closest dedicated tool, and standard R-hat and ESS tooling such as `coda`, `posterior`, or `mcmc` would be needed. `BART` ships Geweke only.

## 13. Verdicts

### BUILD_NOW

1. **BART-style priors as growth-criterion configuration.** The tree prior base and exponent ($\alpha = 0.95$, $\beta = 2$), the leaf-shrinkage SD count ($k = 2$), the error-variance prior shape and scale ($(\nu, q) = (3, 0.90)$), and the candidate-cut grid (100) are all regularisers inside existing split scoring. No MCMC needed.
2. **Monotone-constraint audit against mBART ensemble semantics.** The library already has monotone constraints; the audit is that constrained leaf values, not only split rules, are enforced on the summed or fitted prediction, and that the variance inflation $c^2 = \pi/(\pi-1)$ is documented when the constraint binds.
3. **Hyperparameter naming normalisation.** `base`, `power`, `k`, `sigdf`, `sigquant`, with documented $\alpha, \beta, k, \nu, q$ aliases, so that BART-style configuration maps cleanly onto the library's typed controls.

### BUILD_LATER

4. **Grow-from-root and XBART-style stochastic hill climbing.** A marginal-likelihood split score, a null candidate, probabilistic split sampling, adaptive variable weights, and a presorted split-statistics path.
5. **DART sparse Dirichlet split-weight prior.** Needs split-prior plumbing across the 30 split criteria.
6. **Permutation-null variable-importance thresholds.** Local, global-max, and global-SE variants, reusing the existing split counts, with $L = 100$ and $L_{\text{rep}} = 10$ defaults.
7. **Interaction diagnostics.** TIP, CIP, and CMD with permutation calibration.
8. **Empirical-Bayes prior-weight estimation.** MC-EM in the style of EBcoBART; plug-in estimates understate hyperparameter uncertainty.
9. **Local GP extrapolation leaves.** For out-of-support prediction, preserving the standard behaviour inside the support.
10. **DPMBART error model.** For heteroskedastic and heavy-tailed residuals.
11. **Categorical level-subset splits under BART-style priors.** The library has categorical partition search; the R BART packages use dummy expansion except in the `dbarts` development branch.
12. **Deterministic causal-tree and survival analogues.** Propensity-score splitting plus honest splitting for CATE, and AFT-based MRL or RMST effect measures for survival.

### ORACLE_ONLY

13. **The full BART posterior.** Gibbs and MH sampler, structure moves, leaf and $\sigma$ draws, credible intervals, posterior predictive intervals.
14. **mBART and probit-monotone BART posteriors.** The constrained space breaks the conjugacy.
15. **Bayesian CART and posteriors over tree structures.** Requires reversible-jump MCMC.
16. **Exact end-of-sweep forest MH.** The fully Bayesian XBART variant.
17. **Bayesian causal forests, XBCF, mediation BART, and BART-survival posteriors.** All need posterior draws, and mediation additionally needs a structural-equation identification step.
18. **GP-BART, treed GP, DP-over-tree mixtures, and DP leaf mixtures.**

### OUT_OF_SCOPE

19. **"Bayesian boosting" and "Safe Bayesian boosting."** The papers do not exist.
20. **Sparse additive backfitting and signal or extreme-value thresholding.** Not tree-ensemble methods.
21. **A GPU port before CPU algorithm validation.** Official XBART has no GPU implementation.
22. **Porting the R, Python, or Java packages.** Different language and runtime; the algorithms are covered by the items above.
23. **Survival and competing-risks domain packages** as a scope expansion of the tree-ensemble regression core.

## 14. Relationship to the Other Ensemble Families

This section answers the question the user asked directly, and it is the one that should drive any future spec.

| | Bagging (`ModelForest`) | Boosting (`ModelBoost`) | Meta (`ModelMeta`) | BART |
|---|---|---|---|---|
| What varies | Training data per member | Target per stage | Combination rule | Everything, jointly |
| Members independent | Yes | No | Yes | No, conditionally dependent |
| Sequential fitting | No | Yes | No | Gibbs, joint |
| Sampling | Bootstrap | Optional per stage | Cross-validation folds | None |
| Output | Point prediction | Point prediction | Point prediction | **Posterior over functions** |
| Uncertainty | Diagnostics only | Diagnostics only | Diagnostics only | **Credible intervals and bands** |
| Variable selection | Permutation importance | Permutation importance | Member weights | Posterior inclusion probability |
| Cost driver | Member count | Stage count | Members times folds | Draws times trees times proposals |
| Fits the library's tree | Yes, `Vec<ModelTree>` | Yes, stage via `ModelTree` | Members are any fitted model | Only with a new growth rule |
| Fits the library's leaves | Yes | Yes | Members' leaves | No, leaves are MCMC draws |

**The conclusion for the architecture.** BART is not a `kind` value for `ModelForest`, `ModelBoost`, or `ModelMeta`. It changes the **output contract**, from a point prediction to a posterior. If the library builds a Bayesian tree family, it is a fourth task facade, in the same pattern as Spec 25's `AnomalyModel`, `ClusterModel`, `RuleModel`, and `ComponentModel`.

**But the deterministic priors are reachable now.** BART's tree prior, leaf-shrinkage prior, and error-variance prior can be adopted as growth-criterion configuration over the existing split engine. That gives BART-*shaped* trees with none of the posterior machinery, and it is a smaller project than any boosting variant.

## 15. Citation List

**Core BART and Bayesian CART**
1. Chipman, H. A.; George, E. I.; McCulloch, R. E. (2010). "BART: Bayesian additive regression trees." *AOAS* 4(1):266–298. [https://doi.org/10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285) · [arXiv:0806.3286](https://arxiv.org/abs/0806.3286) · [PDF](https://www.acadiau.ca/~hchipman/papers/AOAS285.pdf)
2. Chipman, George, McCulloch (1998). "Bayesian CART model search." *JASA* 93(443):935–948. [https://doi.org/10.1080/01621459.1998.10473750](https://doi.org/10.1080/01621459.1998.10473750)
3. Chipman, George, McCulloch (1997). "A Bayesian approach for CART." *PMLR* R1:91–102. [https://proceedings.mlr.press/r1/chipman97a.html](https://proceedings.mlr.press/r1/chipman97a.html)
4. Chipman, George, McCulloch (2001). "The practical implementation of Bayesian model selection." *IMS LNS* 38:65–116. [https://doi.org/10.1214/lnms/1215540964](https://doi.org/10.1214/lnms/1215540964)
5. Chipman, George, McCulloch (2000). "Hierarchical priors for Bayesian CART shrinkage." *Statistics and Computing* 10:17–24. [https://doi.org/10.1023/A:1008980332240](https://doi.org/10.1023/A:1008980332240)
6. Denison, D. G. T.; Mallick, B. K.; Smith, A. F. M. (1998). "A Bayesian CART algorithm." *Biometrika* 85(2):363–377. [https://doi.org/10.1093/biomet/85.2.363](https://doi.org/10.1093/biomet/85.2.363)
7. Denison, Mallick, Smith (1998). "Automatic Bayesian curve fitting." *JRSS-B* 60(2):333–350. [https://doi.org/10.1111/1467-9868.00128](https://doi.org/10.1111/1467-9868.00128)
8. Wu, Q.; Tjelmeland, H.; West, M. (2007). "Bayesian CART: prior specification and posterior simulation." *JCGS* 16(1):44–66. [https://doi.org/10.1198/106186007X180426](https://doi.org/10.1198/106186007X180426)
9. Hastie, T. J.; Tibshirani, R. J. (2000). "Bayesian backfitting." *Statistical Science* 15(3):196–223. [https://doi.org/10.1214/ss/1009212815](https://doi.org/10.1214/ss/1009212815) · [PDF](https://hastie.su.domains/public/Papers/BayesianBackfitting.pdf)
10. Hill, J. L.; Linero, D.; Murray, J. S. (2020). "Bayesian additive regression trees: a review and look forward." *Annu. Rev. Statist. Appl.* 7:251–278. [https://doi.org/10.1146/annurev-statistics-031219-041110](https://doi.org/10.1146/annurev-statistics-031219-041110)
11. Linero, D. (2017). "A review of tree-based Bayesian methods." *CSAM* 24(6):543–559. [https://doi.org/10.29220/CSAM.2017.24.6.543](https://doi.org/10.29220/CSAM.2017.24.6.543)
12. Tan, Y.; Roy, S. (2019). "Bayesian additive regression trees and the General BART model." *Statistics in Medicine* 38(25):5048–5069. [https://doi.org/10.1002/sim.8347](https://doi.org/10.1002/sim.8347)
13. Linero, D. (2018). "Bayesian regression trees for high-dimensional prediction and variable selection." *JASA* 113(522):2678–2695. [https://doi.org/10.1080/01621459.2016.1264957](https://doi.org/10.1080/01621459.2016.1264957)
14. Linero, D.; Yang, S. (2018). "Bayesian regression tree ensembles that adapt to smoothness and sparsity." *JRSS-B* 80(5):1087–1110. [https://doi.org/10.1111/rssb.12293](https://doi.org/10.1111/rssb.12293) · [arXiv:1707.09461](https://arxiv.org/abs/1707.09461)
15. Linero, D. (2025). "Generalized BART via Laplace reversible-jump sampler." *JASA*. [https://doi.org/10.1080/01621459.2024.2337156](https://doi.org/10.1080/01621459.2024.2337156)
16. Chipman, George, McCulloch, Shively (2022). "mBART: multidimensional monotone BART." *Bayesian Analysis* 17(2):515–544. [https://doi.org/10.1214/21-BA1259](https://doi.org/10.1214/21-BA1259) · [arXiv:1612.01619](https://arxiv.org/abs/1612.01619)
17. Fisher, T. (2025). "Probit monotone BART." [arXiv:2509.00263](https://arxiv.org/abs/2509.00263)
18. Marvald, C.; Love, E. (2024). "Detecting interactions using Bayesian additive regression trees." *Pattern Analysis and Applications* 27(4):153–168. [https://doi.org/10.1007/s10044-024-01357-x](https://doi.org/10.1007/s10044-024-01357-x)
19. Carvalho, George, Hahn, McCulloch (2020). "Variable selection with Bayesian additive regression trees." [https://doi.org/10.1201/9781003089018-17](https://doi.org/10.1201/9781003089018-17)
20. Bleich, J.; Kapelner, A.; Scheibe, M.; Vavrek, D. (2014). "Variable selection for BART." [arXiv:1310.4887](https://arxiv.org/abs/1310.4887)
21. Luo, Y.; Daniels, K. (2024). "Variable selection using Bayesian additive regression trees." *Statistical Science* 39(2):286–304. [https://doi.org/10.1214/23-STS900](https://doi.org/10.1214/23-STS900)
22. Liu, Y.; Ročková, V.; Wang, S. (2021). "Variable selection with ABC Bayesian forests." *JRSS-B* 83(3):453–481. [https://doi.org/10.1111/rssb.12423](https://doi.org/10.1111/rssb.12423) · [arXiv:1806.02304](https://arxiv.org/abs/1806.02304)
23. Goedhart, P.; Klausch, U.; Janssen, R.; van de Wiel, S. (2025). "Adaptive use of co-data through empirical Bayes for BART." *Statistics in Medicine* 44(5):e70004. [https://doi.org/10.1002/sim.70004](https://doi.org/10.1002/sim.70004)
24. Rocková, V.; van der Pas, S. (2020). "Posterior concentration for Bayesian regression trees and forests." *Annals of Statistics* 48(4):2108–2131.
25. George, N.; Laud, P.; Logan, B.; McCulloch, R.; Sparapani, R. (2019). "Fully nonparametric Bayesian additive regression trees." *Adv. Econometrics* 40B:89–110. [https://doi.org/10.1108/S0731-90532019000040B006](https://doi.org/10.1108/S0731-90532019000040B006) · [arXiv:1807.00068](https://arxiv.org/abs/1807.00068)
26. Hannah, L.; Kottas, D.; Jones, I.; Gavaghan, S. (2011). "Dirichlet process mixtures of generalized linear models." *JMLR* 12:307–339. [https://jmlr.org/papers/volume12/hannah11a/hannah11a.pdf](https://jmlr.org/papers/volume12/hannah11a/hannah11a.pdf)
27. Evers, M.; Heaton, T. (2009). "Locally adaptive tree-based thresholding." *JCGS* 18(4):961–977. [https://doi.org/10.1198/jcgs.2009.07109](https://doi.org/10.1198/jcgs.2009.07109)
28. Lee, J.; Fan, Y.; Sisson, S. (2015). "Bayesian threshold selection for extremal models using measures of surprise." *CSDA* 85:84–99. [https://doi.org/10.1016/j.csda.2014.12.004](https://doi.org/10.1016/j.csda.2014.12.004)
29. Starling, D.; Murray, J.; Carvalho, C.; Bukowski, J.; Scott, A. (2020). "BART with targeted smoothing." *AOAS* 14(1):28–50. [https://doi.org/10.1214/19-AOAS1268](https://doi.org/10.1214/19-AOAS1268) · [arXiv:1805.07656](https://arxiv.org/abs/1805.07656)

**XBART and scalable BART**
30. He, J.; Yalov, S.; Hahn, P. R. (2019). "XBART: Accelerated Bayesian Additive Regression Trees." *PMLR* 89:1130–1138. [https://proceedings.mlr.press/v89/he19a.html](https://proceedings.mlr.press/v89/he19a.html) · [arXiv:1810.02215](https://arxiv.org/abs/1810.02215)
31. He, J.; Hahn, P. R. (2023). "Stochastic tree ensembles for regularized nonlinear regression." *JASA* 118(541):551–570. [https://doi.org/10.1080/01621459.2021.1942012](https://doi.org/10.1080/01621459.2021.1942012) · [arXiv:2002.03375](https://arxiv.org/abs/2002.03375)
32. He, J. (2020). *XBART: a scalable stochastic algorithm for supervised machine learning with additive tree ensembles*. Dissertation. [https://doi.org/10.6082/766sx-2ea62](https://doi.org/10.6082/766sx-2ea62)
33. Petrillo, A. (2024). "Very fast Bayesian Additive Regression Trees on GPU." [arXiv:2410.23244](https://arxiv.org/abs/2410.23244)
34. Krantsevich, A. et al. (2023). "Accelerated Bayesian causal forests." *PMLR* 206.
35. Implementations: [JingyuHe/XBART](https://github.com/JingyuHe/XBART) · [StochasticTree/stochtree](https://github.com/StochasticTree/stochtree)

**GP hybrids and treed GP**
36. Gramacy, R. B.; Lee, H. K. H. (2008). "Bayesian treed Gaussian process models." *JASA* 103(483):1119–1130. [https://doi.org/10.1198/016214508000000689](https://doi.org/10.1198/016214508000000689) · [arXiv:0710.4536](https://arxiv.org/abs/0710.4536)
37. Maia, A.; Murphy, C.; Parnell, C. (2024). "GP-BART." *CSDA* 190:107858. [https://doi.org/10.1016/j.csda.2023.107858](https://doi.org/10.1016/j.csda.2023.107858) · [arXiv:2204.02112](https://arxiv.org/abs/2204.02112)
38. Wang, C.; He, X.; Hahn, P. R. (2024). "Local Gaussian process extrapolation for BART models." *JCGS* 33(2):724–735. [https://doi.org/10.1080/10618600.2023.2240384](https://doi.org/10.1080/10618600.2023.2240384) · [arXiv:2204.10963](https://arxiv.org/abs/2204.10963)

**Causal, survival, mediation**
39. Hahn, P. R.; Murray, J. S.; Carvalho, C. M. (2020). "Bayesian regression tree models for causal inference." *Bayesian Analysis* 15(3):965–1056. [https://doi.org/10.1214/19-ba1195](https://doi.org/10.1214/19-ba1195)
40. Hill, J. L. (2011). "Bayesian nonparametric modeling for causal inference." *JCGS* 20(2):217–240. [https://doi.org/10.1198/jcgs.2010.08162](https://doi.org/10.1198/jcgs.2010.08162)
41. Sparapani, R. A.; Logan, B. R.; McCulloch, R. E.; Laud, P. W. (2016). "Nonparametric survival analysis using BART." *Statistics in Medicine* 35(16):2741–2753. [https://doi.org/10.1002/sim.6893](https://doi.org/10.1002/sim.6893)
42. Hu, X.; Ji, J.; Li, Q. (2021). "Estimating heterogeneous survival treatment effect in observational data using machine learning." *Statistics in Medicine*. [https://doi.org/10.1002/sim.9090](https://doi.org/10.1002/sim.9090)
43. Sparapani, R. A.; Logan, B. R.; Maiers, M.; Laud, P. W.; McCulloch, R. E. (2023). "Nonparametric failure time." *Biometrics*. [https://doi.org/10.1111/biom.13857](https://doi.org/10.1111/biom.13857)
44. Liu, Y.; Qin, T.; Talisa, S.; Wang, Y. (2026). "Heterogeneous causal mediation analysis using BART." *Biometrics* 82. [https://doi.org/10.1093/biomtc/ujag079](https://doi.org/10.1093/biomtc/ujag079)
45. Ting, J.; Linero, D. (2023). "Estimating heterogeneous causal mediation effects with Bayesian decision tree ensembles." [arXiv:2303.01620](https://arxiv.org/abs/2303.01620)
46. Linero, D.; Zhang, Y. (2025). "Mediation analysis using Bayesian tree ensembles." *Psychological Methods* 30(1):60–82. [https://doi.org/10.1037/met0000504](https://doi.org/10.1037/met0000504)
47. Kim, S.; Daniels, M.; Marcus, J.; Roy, D. (2017). "A framework for Bayesian nonparametric inference for causal effects of mediation." *Biometrics* 73(2):401–409. [https://doi.org/10.1111/biom.12575](https://doi.org/10.1111/biom.12575)

**Bayesian boosting and safe Bayes**
48. Friedman (2001). "Greedy function approximation." *Annals of Statistics* 29(5):1189–1232. [https://doi.org/10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
49. Friedman, Hastie, Tibshirani (2000). "Additive logistic regression: a statistical view of boosting." *Annals of Statistics* 28(2):337–407. [https://doi.org/10.1214/aos/1016218223](https://doi.org/10.1214/aos/1016218223)
50. Ehrlinger, J.; Ishwaran, H. (2012). "Characterizing L2Boosting." *Annals of Statistics* 40(2):1074–1101. [https://doi.org/10.1214/12-aos997](https://doi.org/10.1214/12-aos997)
51. Ustimenko, D.; Beliakov, G.; Prokhorenkova, L. (2022). "Gradient Boosting Performs Gaussian Process Inference." [arXiv:2206.05608](https://arxiv.org/abs/2206.05608)
52. Lorbert, A.; Blei, D.; Schapire, R.; Ramadge, R. (2012). "A Bayesian boosting model." [arXiv:1209.1996](https://arxiv.org/abs/1209.1996)
53. Miller, J.; Foti, N.; Adams, R. (2016). "Variational boosting." [arXiv:1611.06585](https://arxiv.org/abs/1611.06585)
54. Xu, B.; Yu, B. (2021). "Adaptive Bayesian learning via boosting." *Annals of Statistics* 49(4):2104–2132. [https://doi.org/10.1214/20-AOS1997](https://doi.org/10.1214/20-AOS1997)
55. Denison, D. (2001). "Boosting with Bayesian stumps." *Statistics and Computing* 11(2):171–178. [https://doi.org/10.1023/a:1008931416845](https://doi.org/10.1023/a:1008931416845)
56. Nock, R.; Sebban, R. (2001). "A Bayesian boosting theorem." *Pattern Recognition Letters* 22(3–4):413–419. [https://doi.org/10.1016/s0167-8655(00)00137-9](https://doi.org/10.1016/s0167-8655(00)00137-9)
57. Grünwald, P. (2012). "The safe Bayesian." [https://doi.org/10.1007/978-3-642-34106-9_16](https://doi.org/10.1007/978-3-642-34106-9_16)
58. de Heide, M.; Kirichenko, P.; Mehta, S.; Grünwald, P. (2020). "Safe-Bayesian generalized linear regression." *PMLR* 108:2623–2633. [https://proceedings.mlr.press/v108/heide20a.html](https://proceedings.mlr.press/v108/heide20a.html) · [arXiv:1910.09227](https://arxiv.org/abs/1910.09227)
59. Malinin, N.; Prokhorenkova, L.; Ustimenko, D. (2020). "Uncertainty in gradient boosting via ensembles." [arXiv:2006.10562](https://arxiv.org/abs/2006.10562)

**Packages and software**
60. `BayesTree` — [https://cran.r-project.org/package=BayesTree](https://cran.r-project.org/package=BayesTree)
61. `BART` (McCulloch, Sparapani) 2.9.10 — [https://cran.r-project.org/package=BART](https://cran.r-project.org/package=BART) · [reference manual](https://cran.r-project.org/web/packages/BART/refman/BART.html)
62. Sparapani, Spanbauer & McCulloch (2021). "The BART R package." *JSS* 97(1):1–66. [https://doi.org/10.18637/jss.v097.i01](https://doi.org/10.18637/jss.v097.i01)
63. `dbarts` 0.9-34 — [https://cran.r-project.org/package=dbarts](https://cran.r-project.org/package=dbarts) · [GitHub](https://github.com/vdorie/dbarts)
64. `bartMachine` 1.4.2 — [https://cran.r-project.org/package=bartMachine](https://cran.r-project.org/package=bartMachine)
65. Kapelner & Bleich (2016). "bartMachine." *JSS* 70(4):1–40. [https://doi.org/10.18637/jss.v070.i04](https://doi.org/10.18637/jss.v070.i04)
66. `bartisan` — [https://cran.r-project.org/package=bartisan](https://cran.r-project.org/package=bartisan)
67. `bartCause` — [https://cran.r-project.org/package=bartCause](https://cran.r-project.org/package=bartCause)
68. `bartpy` — [https://pypi.org/project/bartpy/](https://pypi.org/project/bartpy/)
69. `bartz` — [https://pypi.org/project/bartz/](https://pypi.org/project/bartz/)
70. `rbartpackages` — [https://pypi.org/project/rbartpackages/](https://pypi.org/project/rbartpackages/)
71. `stochtree` 0.4.5 — [https://cran.r-project.org/package=stochtree](https://cran.r-project.org/package=stochtree)
72. `xbart` 0.1.8 — [https://pypi.org/project/xbart/](https://pypi.org/project/xbart/)
73. `BART::gewekediag` — [https://rdrr.io/cran/BART/man/gewekediag.html](https://rdrr.io/cran/BART/man/gewekediag.html)
74. `tgp` defaults — [https://search.r-project.org/CRAN/refmans/tgp/html/tgp.default.params.html](https://search.r-project.org/CRAN/refmans/tgp/html/tgp.default.params.html)

**Local benchmark artefacts**
75. [`docs/benchmarks/elo_overall.md`](../docs/benchmarks/elo_overall.md) · [`docs/benchmarks/elo_board.md`](../docs/benchmarks/elo_board.md) · [`docs/benchmarks/elo_cla_board.md`](../docs/benchmarks/elo_cla_board.md)

**Unverified, do not assert:**

1. JCGS 2001 "Bayesian backfitting" by Friedman, Hastie, and Tibshirani. It does not exist. Substitute Hastie & Tibshirani 2000, DOI 10.1214/ss/1009212815.
2. JASA 2001 "Additive Bayesian Regression." Not verified; treat as non-existent.
3. Scheibler, Carvalho, Tewari, Boyd spike-and-slab BART, JMLR 2019. Not verified. Substitute Linero & Yang 2018 and Liu, Ročková & Wang 2021.
4. EBART, Kim & Oh, JMLR 2024. Not verified. Substitute Linero 2017 and Goedhart et al. 2025.
5. "Bayesian boosting," Geurst, van der Vaart, van der Velden, AoS 2021. Does not exist.
6. "Safe Bayesian boosting," Simin, Grinsztajn, JRSS-B 2022. Does not exist.
7. Künzel, Sekhon, Bickel, Pollard "Nonparametric survival analysis for causal inference," Biometrics 2021. Not verified; both offered DOIs belong to other papers.
8. "Causal mediation analysis with a multilevel BART model" by that exact title. Not found; substitute Chen 2020, DOI 10.3102/1584027.
9. `stochtree` Python defaults. Unverified; R defaults verified.
10. arXiv:1904.01730 as a BART paper. False; it resolves to a machine-type-communication scheduling paper.
11. Official GPU XBART implementation, and the measured "GPU XBART speedup". Does not exist and is unverified respectively.
