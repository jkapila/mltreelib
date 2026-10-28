# 09 — The TabArena field (what we actually compete against)

The entrants on the live leaderboard (`references/data/tabarena-leaderboard.csv`,
models pool, 2026-10-28; Elo anchored on RandomForest = 1000). Tree/other
rows only — the foundation models are the *opponents*, summarized in
`references/research/tree-vs-foundation-tabular.md`, not repeated here.

| Entrant | Best (tuned+ens) | tuned | default | Canonical entry |
|---|---|---|---|---|
| LightGBM | 1411 | 1368 | 1184 | #19 |
| CatBoost | 1400 | 1389 | 1361 | #20 |
| ChimeraBoost | 1365 | 1338 | 1302 | below |
| XGBoost | 1359 | 1336 | 1209 | #18 |
| CTBoost | 1326 | 1297 | 1173 | below |
| xRFM | 1336 | 1292 | 1044 | below |
| EBM | 1254 | 1220 | 1191 | #36 |
| PrismBoost | 1216 | 1183 | 1127 | below |
| ExtraTrees | 1204 | 1180 | 1010 | #28 |
| APLR | 1185 | 1144 | 1076 | below |
| RandomForest | 1173 | 1138 | 1000 | #27 |
| PerpetualBooster | 1093 | 1058 | 935 | 08 (`perpetual`) |

Reading: tuned+ensembled trees cluster ~1170–1410; defaults vary wildly
(ChimeraBoost 1302 default vs XGBoost 1209 — **good defaults are a feature**);
PerpetualBooster's defaults lag but it is the only Rust core on the board.

## ChimeraBoost (bbstats/chimeraboost)

**Idea (verified from README):** "lightning-fast gradient boosting, near-CatBoost
quality, all in Python"; **#2 on TabArena GBDT Elo (defaults)**. Internals are
a fast histogram-style GBDT engineered for warm defaults; exact blend not
documented in the README — check its docs before modeling on it.
**Pseudocode:** same skeleton as #17/#19 (gradient boosting with histogram
splits); its contribution is the defaults/speed engineering.

## CTBoost (captnmarkus/ctboost)

**Idea:** GBDT whose trees are **conditional-inference trees**: at each node,
select the feature with a **conditional statistical test** first, then optimize
that feature's split point. C++17 core, optional CUDA, sklearn API.

```text
for m in 1..M:                        # boosting rounds
    r_i = pseudo-residuals
    grow tree on (X, r):
        node: feature z* = conditional test on r vs each feature
              (multiplicity-adjusted, ctree-style)  -> see #6
              threshold = argmax separation of r over z* values
    leaf outputs from r; F += nu * tree
```

**Note:** ctree's unbiased feature choice inside boosting — the statistical-
split idea applied at scale. Directly relevant to our "native categorical"
requirement (test-based choice does not favor high-cardinality features).

## PrismBoost (PrismBoost/PrismBoost)

**Idea:** GBDT with **SEFR oblique (hyperplane) splits** at internal nodes,
optional fast C++ backend.

```text
for m in 1..M:                        # like #25 SEFRBoost
    grow tree where each split is a hyperplane on the residuals
    leaf outputs from residual means; F += nu * tree
```

**Note:** the oblique-GBDT family (#25 SEFRBoost sibling); on TabArena it sits
between EBM and ExtraTrees.

## APLR — Automatic Piecewise Linear Regression (ottenbreit-data-science/aplr)

**Idea:** additive **piecewise-linear** terms learned automatically (both
regression and classification); smooth, interpretable, accuracy "often rivals
tree-based methods".

```text
for each feature j: piecewise-linear basis b_jk(x_j) learned automatically
    (knot placement + segment slopes chosen to reduce loss)
F(x) = base + Σ_j Σ_k beta_jk * b_jk(x_j)     # additive, no interactions
boosting-style refinement of terms until convergence
```

**Knobs:** number of terms/interactions, validation strategy, monotonicity.
**Note:** MARS (#8) / EBM (#36) family — smoothness as the interpretability
selling point; a real alternative when segment edges must not jump.

## xRFM (Beaglehole, Holzmüller, Radhakrishnan, Belkin; ICLR 2026)

**Idea:** eXtended **Random Feature Models** — the kernel/NTK school's tabular
answer: learn nonlinear random features (kernel machine with learned feature
map), then a **linear head** on top; accurate, scalable (GPU), and
interpretable through the linear head.

```text
features: h(x) = phi(W x + b) with learned/expanded random feature map W
head: linear (or generalized linear) model on h(x)
training: fit head + refine features by gradient (RFM-style kernel machinery)
```

**Note:** the most direct "foundation-adjacent classical" competitor on the
board; its interpretability story = linear head over learned features — worth
a structure-recovery test: do its learned features respect our group/hierarchy
signal?

## Field-level takeaways for the build

1. The tree leaderboard's top half is **engineering on old ideas** (defaults,
   speed, oblique splits, statistical splits) — exactly our "under-studied"
   thesis: there is room for a principled rebuild.
2. Defaults matter as much as tuned performance — our model tree needs good
   out-of-the-box behavior.
3. All comparisons must use `Metrics` (03-commit yardstick) + equal budgets
   (see `tree-vs-foundation-tabular.md` for the protocol).
