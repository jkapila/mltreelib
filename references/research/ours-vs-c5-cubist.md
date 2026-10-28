# Ours vs C5.0 vs Cubist — what differs, and how we'll test against them

**Date:** 2026-10-28. **Grounding:** the extracted vendor manuals
(`../docs/see5-unix.md`, `../docs/cubist-unix.md`) and the C sources in
`../sources/extracted/see5-public/`, `../sources/extracted/cubist-public/`.
**Status:** analysis + test plan (the rpart sanity is done — see
`crates/mltreelib-core/tests/rpart_repro.rs`; this extends the same playbook
to RuleQuest).

## 1. The comparison

| aspect | C5.0 | Cubist | ours (v1) |
|---|---|---|---|
| task | classification | regression | regression (numeric cols) |
| split criterion | information/gain-style on class distribution | accuracy/SD-style on target | **fitted-leaf SSE reduction** (≡ rpart deviance in CART mode, verified) |
| leaf | class (votes) or rulesets | **linear model per rule** (model tree) | OLS (`All` or `PathUsed`) or mean (CART mode) |
| missing values | case **splits into parts**, fractional votes aggregated (see5-unix.md:614-617) | unknown attribute → rule conditions may fail → fallback | **learned default direction** + NA-indicator leaf terms |
| pre-selection | **winnowing** (`-w`), before the tree is built (see5-unix.md:1110-1256) | — | preselect layer (see `preselect-preprocessing-catalog.md`) |
| ensembles | **boosting** trials (`-t`), ~25% avg error reduction | **committees** (`-C`, rec. 5): member k corrects members 1..k-1's residuals; predictions averaged (cubist-unix.md:886-923) | not yet (boosting is next per the thesis arc) |
| local correction | — | **composite models** (`-i`/`-a`): n-nearest-neighbor adjustment of the model prediction by neighbors' (actual − predicted) deviations (cubist-unix.md:816-883) | not yet (TrieKNN roadmap) |
| extrapolation | n/a | **clamped** (`-e`, default 5%): predictions limited to rule's training range ±5%·range; never below 0 when all training targets ≥ 0 (cubist-unix.md:992-1024) | **unbounded by design** ("the single tree that extrapolates") — clamping is a policy we can add later |
| pruning | global tree pruning (error-based; `-g` disables) + soft thresholds (`-p`) | simplicity/accuracy trade-off (BREV) | min_rows/min_gain only (cost-complexity = the later "weighing criteria") |
| scale aids | sampling (`-x`), cross-validation (`-X`) | sampling (`-S`), xval (`-X`), eval samples >20k | not yet (catalog has the menu) |

## 2. What this says about our position

- **Split selection** is our one deliberate divergence: we choose the split
  by how well the *fitted leaf models* improve (the model-tree criterion),
  not by node impurity alone. That is exactly the axis Cubist cares about
  (leaf linear models), and it made us match rpart bit-for-bit in CART mode —
  the criterion is sound.
- **Cubist is the closest relative**: rule/model tree + linear leaves. The
  deltas that make Cubist strong are *additive*: committees (boosting-like),
  instance correction (KNN), extrapolation clamping. All three are on our
  roadmap; none requires changing the tree core — that is the validation
  that our `NodeModel` seam is the right one.
- **C5.0's NA mechanism is the most interesting divergence**: fractional
  case-splitting with vote aggregation vs our learned directions. Both are
  "NA as part of the problem"; theirs is distributional, ours is
  learned-per-split + indicator terms. Worth a head-to-head on missingness
  benchmarks later.
- C5.0's **winnowing** is the pre-selection anchor (our `preselect` layer's
  first recipe).

## 3. First executed results (2026-10-28): Cubist vs our model tree

Ground truth apparatus: RuleQuest's official demo tarballs
(`CubistDemo_linux64.tgz`, `C50Demo_linux64.tgz` from
rulequest.com/download.html — vendored in `../sources/`) plus the **full GPL
engines** (`GPL/Cubist.tgz`, `GPL/C50.tgz`; note the older `*-public.tgz`
zips we hold are classify-only samplers, no construction engine).

Executed with `examples/cubist_comparison.py` — Cubist 2.10 demo
(`bin/cubistdemo -f Data/boston`) vs our ModelTree on the identical 200
training cases (the demo's case cap) and Cubist's own 12-predictor set
(Cubist itself dropped `B`):

| model | avg abs err | rel err | corr |
|---|---|---|---|
| Cubist 2.10 (5 rules) | 1.41 | 0.25 | 0.97 |
| ModelTree linear (All) | 1.006 | 0.175 | 0.985 |
| ModelTree linear (Path) | 0.941 | 0.164 | 0.985 |
| ModelTree mean (CART) | 1.764 | 0.307 | 0.947 |

**Reading (honest):** these are *training* numbers (the demo reports training
evaluation; boston ships without a `.test` file). Our single tree fits harder
than Cubist's deliberately simplified 5-rule model — the flexibility is
clearly there (linear leaves matter: mean-leaf CART is far behind, which
validates the model-tree direction). Generalization is the open question:
next run uses `concrete`/`abalone`/`fc2008` which ship `.test` files, with
Cubist's `-e` extrapolation clamp and committees as controlled variants.

Also executable now: C5.0 (`bin/c50demo`, classification datasets
anneal/breast-cancer/housing/soybean) for the missing-value head-to-head
(its fractional case-splitting vs our learned directions) once we add
classification support; `attwinnow.c` in the GPL engine is the winnowing
ground truth for the preselect layer.

## 4. Test plan (the rpart playbook extended)

1. **Ground truth = RuleQuest's official demos** (`bin/cubistdemo`,
   `bin/c50demo` + the shipped Data/) and the GPL engines for source-level
   questions — the `*-public.tgz` samplers cannot construct models.
2. **Cubist base model vs our model tree** (same data, committees and
   instances OFF in Cubist): compare split structure, per-leaf linear
   equations, and predictions. Expected deltas: leaf smoothing? attribute
   selection inside rules (their rules pick attribute subsets ≈ our
   `PathUsed`!), extrapolation clamping (turn `-e` up to isolate it).
3. **Cubist composite/committee deltas**: quantify what the two additive
   mechanisms buy on the same data (the manual already gives numbers: base
   152.6 → composite 99.8 → committee+instances 94.8 avg error) — our
   benchmark harness should reproduce that curve as we add neighbors and
   committees.
4. **C5.0 split-choice comparison** (classification dataset): compare first
   splits vs our regression-tree first splits as a structural sanity, and
   later run the missingness head-to-head (their fractional votes vs our
   learned directions).
5. **Winnowing parity**: implement the preselect utility screen and check it
   drops the same attributes C5.0 `-w` drops on the hypothyroid example
   (the manual shows 14 of 22 winnowed).
