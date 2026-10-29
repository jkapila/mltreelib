# Tree research consolidation + build TODO

**Date:** 2026-10-28 (master note — supersedes nothing; unifies
`tree-vs-foundation-tabular.md`, `model-tree-survey.md`,
`categorical-encoding-research.md`, `jkapila-methods-map.md`,
`dataset-component-design.md`)
**Decision this informs:** what we build first and in what order. Metrics are
done (`Metrics`, commit 30a9ec1) — the yardstick exists; the build starts now.

## The thesis, one line

Tree/ensemble methods are under-studied relative to tabular foundation models,
and the signal in categorical/group/hierarchical structure is what flat
learners lose — measurable via the generator's ground truth and the TabArena
suite (see the earlier notes for the numbers).

## What the studied packages actually teach (mechanisms, not vibes)

### Leaf-local models = the model tree family (our first build)

| Source | Mechanism | Take |
|---|---|---|
| cerlymarco/linear-tree | `Node` + `_LinearTree._split`: split gain = SSE improvement from fitting **linear models on both sides** of each candidate split; leaves hold linear estimators. Plus LinearForest (linear fit + RF on residuals) and LinearBoost (worst-leaf binary features) | The v1 recipe for our model tree; LinearForest = later mixing recipe |
| Shamiri-Institute/modeltrees | Full MOB port of **partykit**: parametric `NodeModel` protocol (`fit` → coefficients/objfun, `predict`, `loglik`, batch `split_objectives`), split chosen by **M-fluctuation instability tests** on estimating functions (supLM/chisq statistics, decorrelated processes), SplitKind numeric/ordered/nominal, cluster-robust sandwich vcov, dqrls port with LAPACK fast path, narwhals dataframes | The reference architecture: protocol-driven node models + a *second* split-criterion family. Its design docs are a blueprint (error taxonomy, warnings, formula API `y ~ x | z`) |
| schufa-innovationlab/model-trees | sklearn estimators, **gradient-based split criterion** | Third criterion family: gradient improvement |
| adc-trust-ai/trust-free (PRICAI 2025) | TRUST: ultra-**sparse** linear leaves, RF-level accuracy (R2 0.67 vs RF 0.62 on 60 datasets), auto `explain()`/`compare()` reports | Sparse leaves are the accuracy+interpretability frontier; report UX bar to match |
| TAO (NeurIPS 2025) | Tree Alternating Optimization for linear-leaf regression trees: monotone convergence, faster than greedy partitioning | Algorithm upgrade path after the greedy version works |
| Cubist (RuleQuest) | Model trees + committees + **instance-based (kNN) corrections** + rules + variable importance | Still the complete spec; the packages above are partial reinventions of it |

### The boosting giant's version

**LightGBM `linear_tree=true`** does put linear models in leaves — but inside
boosting: per iteration, leaf models are fit on the gradient/hessian targets,
regularized by `linear_lambda`, with `linear_tree_output` choosing how leaf
models combine (`share`/`separate`). It is an opt-in flag in a GBDT, not a
standalone model tree: no per-leaf reporting/inference, no stability tests,
categoricals stay histogram-split, and the leaf linearity serves residual
fitting rather than interpretable segments. Verified against the LightGBM
parameters docs and the model-tree literature framing; treat as "the caveat
you remembered is real."

### Ensembles and uncertainty (the R classics that never get highlighted)

- **grf** (Athey/Tibshirani/Wager; grf-labs/grf): generalized random forests —
  and crucially **Local Linear Forests**, the forest version of linear leaves.
- **BART** (Kapelner & Chipman; BayesTree): Bayesian additive regression trees
  — posterior uncertainty, the UQ baseline trees rarely get credit for.
- Held tarballs in `sources/`: partykit (the MOB original), party (conditional
  inference splits), mboost (model-based boosting), glmertree (GLMM leaves!),
  earth (MARS), semtree (SEM leaves), wsrf (weighted RF).

### Rust landscape (references for the build)

`rpart` (cloned), `sklears-tree` (CHAID + oblique), `treeboost`, `affinitree`;
our `mltreelib-core` already carries data layer, features, encoders, metrics.

### arXiv recency (2025-2026)

RO-FIGS (expressive tree ensembles for tabular data), MotherTree (meta-learned
tree training; surveys oblique trees), JMLR 2026 approximation-free
differentiable oblique trees, Vectorized Adaptive Histograms for Sparse
Oblique Forests, Jacobian Aligned Random Forests. Active frontier: oblique
splits, faster non-greedy training, feature interactions.

### Tooling note

Package internals were mapped with structural grep + the mounted `semble`
search and symbol outlines; `archify` / `code-review-graph` are not mounted in
this environment.

## The build TODO (in order)

- [ ] 1. **Simple model tree v1** — CART-style growth, **leaf-local linear
      models** (the user's first target): `ModelTree` core in
      `mltreelib-core/src/tree/`; leaf OLS via a proper linalg crate (faer —
      no wheel reinvention), split gain = SSE reduction from fitting the leaf
      models on both sides (linear-tree recipe)
- [ ] 2. House requirements in the first cut: **native missing values**
      (learned default directions) and **native categorical splits**
- [ ] 3. Leaf diagnostics: per-leaf coefficients + `Metrics.regression_by_group`
      reports (leaves as groups) — day-one interpretability
- [ ] 4. CART + C4.5 with constant leaves (oracle: `legacy/mltreelib/tree/models.py`)
      for the constant-vs-linear comparison on `datasets/`
- [ ] 5. **Cubist**: model trees + committees + kNN instance corrections
      (`references/docs/cubist-unix.md`) — the full spec the surveyed packages only
      approximate
- [ ] 6. C5.0 rulesets + boosting committees (`references/docs/see5-unix.md`);
      GritBot surprising-subgroup mining (`references/docs/gritbot-unix.md`)
- [ ] 7. Second split criterion: MOB-style instability tests (partykit lineage,
      modeltrees as blueprint); third: gradient improvement (schufa/TAO path)
- [ ] 8. Mixing: LinearForest-style (linear + residual ensembles), committees,
      sparse-leaf regularization (TRUST), BART-style uncertainty on top
- [ ] 9. Everything scored with `Metrics` + structure recovery on the
      generator's truth; foundation-model comparisons per
      `tree-vs-foundation-tabular.md`

## Resolved thread: the news-recommendation paper

Found (2026-10-28): **"Personalized News Recommendation with Context Trees"** —
Florent Garcin, Christos Dimitrakakis, Boi Faltings (EPFL), **RecSys 2013**,
pp. 105–112 (arXiv:1303.0665). Context tree (VMM) over browsing steps;
experts per node mixing **standard / popularity / freshness** local models;
every recommendation comes from the stop-weight mixture along the active
path — the popularity + freshness + user-steps hybrid in question.
Pseudocode in `references/algorithms/10-context-trees-tries.md`
("Applied" section).

## Open questions for Jitin

The phase frame (variable pre-selection + expert-mixture thesis, classical
trees only) lives in `preselection-and-expert-thesis.md` — it re-anchors the
build TODO below on its two starting points: a single tree with
linear/logistic leaves, and the pre-selection layer.

1. Model tree v1 leaves: full OLS first, or sparse (TRUST-style) from the start?
2. Leaf reporting: match trust-free's `explain()`/`compare()` UX, or keep v1
   as the per-leaf `Metrics` tables?
3. The "more tree implementations" list to grow — any specific ones beyond the
   surveyed set?
