# Research: Can classical tree ensembles beat tabular foundation models — and what Rust core to build on?

**Date:** 2026-10-28
**Decision this informs:** Whether to build a Rust-backed, Python-fronted tree/GBDT package aimed at contextual (non-IID, grouped, temporal, large) settings, and which crate to base it on.
**Confidence overall:** high for leaderboard status (primary CSVs), medium for market direction (fast-moving, vendor-reported).

## TL;DR

On TabArena's IID board (28 Oct 2026) foundation models dominate: best FM Kumo-Tabular 1955 Elo vs best tree LightGBM (tuned+ens) 1411 — a ~540-point gap. But on the beyond-IID slices the gap collapses: on **grouped splits LightGBM (1220) outranks every foundation model** (Kumo 1219, TabPFN-3.5 1159); on **temporal splits it's a statistical tie** (TabPFN-3.5 1179 ≈ CatBoost 1138 with overlapping CIs). Many FMs don't even complete the contextual slices (22–57% of results imputed). Trees also own the cost columns: ~0.14–0.2 s/1K-row CPU inference vs 1.5–7.6 s/1K GPU. Your hypothesis is directionally supported but eroding — TabPFN-3.5 (Sep 2026) explicitly targets non-IID/text/high-cardinality. For the Rust core: **`perpetual` is the right architecture** (Rust core, mature PyO3/maturin Python+R bindings, Arrow/Polars zero-copy, on PyPI, Apache-2.0 — it's even a verified TabArena entrant) but its accuracy is weak; **`forust-ml` is the right algorithm** (XGBoost-parity in pure Rust + Python wrapper). A fork/merge of the two, plus CatBoost-style categoricals and C5.0/Cubist-style rule ensembles, is the credible "beat the giants in contextual settings" play.

## Key findings

### 1. The scoreboard today: FMs own IID, trees own the contextual slices

Elo, models pool, as published in the leaderboard repo's CSVs [1]:

| Regime | Best FM | Best tree | Verdict |
|---|---|---|---|
| IID (51 datasets) | Kumo-Tabular 1955 | LightGBM tuned+ens 1411 | FM +544 |
| Temporal splits | TabPFN-3.5 1179 | CatBoost tuned+ens 1138 | tie (CIs overlap) |
| Grouped splits | Kumo-Tabular 1219 | **LightGBM tuned+ens 1220** | tree wins |

On IID the first tree (LightGBM tuned+ens, rank 37.8) sits below 18 foundation models; defaults are far worse (LightGBM default 1184). Sources: [1], consistent with [2], [4].

### 2. "Contextual" is exactly where FMs get flaky — and where trees hold

BeyondArena temporal: TabFM, LimiX-2, EXAONE, RealTabPFN, TabDPT, TabSwift report **51–57% imputed results** (didn't run/complete on most temporal datasets); grouped: 22% imputed [1]. On grouped data the top 2 are tuned neural nets (RealMLP 1249, TabM 1238) then LightGBM (1220) — every FM is below the best tree. Supporting analysis: TabArena explicitly scopes IID small-to-medium data and leaves >250k rows, temporal/grouped, and shift outside the main board [2], [4]; TabPFN v2 degrades on high-dimensional/many-category/large data [7]. Counter-evidence below in Contradictions.

### 3. Cost/latency asymmetry is structural

Median predict time on temporal: CatBoost tuned 0.138 s/1K rows (CPU) vs TabPFN-3.5 7.643 (GPU), Kumo 1.743 [1]. Trees train and serve on CPU; FMs need a GPU budget conversation. Also NVIDIA's own caveats for Kumo: numerical+categorical columns only (no text/images/timestamps without preprocessing), accuracy degrades under drift from context rows [2].

### 4. Trees win through tuning — budgets must be equal to be fair

LightGBM temporal: 965 default → 1133 tuned+ens (+168). XGBoost grouped: 1000 default → 1096 tuned [1]. This reproduces the 2022 Grinsztajn lesson (most "beat XGBoost" claims used weak baselines) [2], [6]. Any experiment you run MUST give trees the same HPO/ensembling budget as the FM gets for free.

### 5. Tree-side research is live but thin — your hypothesis has merit

2026 board shows new tree entrants beating old giants at defaults: ChimeraBoost 1302, CTBoost 1173, PrismBoost 1127, EBM 1191 vs XGBoost default 1209 / LightGBM default 1184 [1]. Few teams work on tree internals in Rust; the only Rust booster on the board, **PerpetualBooster** (Apache-2.0, the `perpetual` crate), scores 935–1093 — production-quality architecture, algorithmically behind [1], [5]. Gap = opportunity.

### 6. The Rust base: `perpetual` for the package skeleton, `forust-ml` for the algorithm

Grounded from crates.io + project READMEs this session [5]:
- **perpetual** 3.0.0-rc.2 — Rust core, Python & R bindings (PyO3/maturin), zero-copy Polars/Arrow, `pip install perpetual`, conda-forge, documented Python API. Already structured exactly like the package you want to build. Its TabArena scores are the accuracy target to beat.
- **forust-ml** 0.8.0 — pure-Rust implementation of the XGBoost algorithm ("nearly identical results") with a `py-forust` Python wrapper; fastest path to giant-parity accuracy.
- Support cast: `bosk` (pure-Rust LightGBM/CatBoost/ONNX inference — deploy side), `miniboosts`/`sooboost-core`/`gbt-quantile` (boosting algorithm references), `smartcore`/`linfa` (trees/ensembles/ExtraTrees, research reference), `lightgbm3`/`xgb`/`catboost-rs` (FFI baselines to benchmark against).
- Missing everywhere in Rust (the CatBoost/XGBoost feature set you named): ordered target statistics for high-cardinality categoricals, GPU training, monotone constraints, advanced missing-value handling. That's the build list.

### 7. The C5.0/Cubist angle is a real differentiator

Your references folder's See5/C5.0 and Cubist docs cover rule committees, boosting committees, case weighting, and Cubist's model-trees + nearest-neighbor correction — classical tech that is strong exactly where FMs claim their niche (small, contextual, low-latency CPU). No Rust implementation exists in the crates surveyed. A Cubist-style rule-committee regressor is a defensible novelty claim against "no one researches trees" [1], [5], [2].

## Contradictions / uncertainty

- **Your hypothesis vs TabPFN-3.5's claims.** You say FMs fail in contextual settings; TabPFN-3.5 (Sep 2026) claims SOTA on non-IID temporal/grouped splits, strings/text/images, high-cardinality categoricals, wide tables [3]. The live board partially backs them (temporal #1 at 1179, 0% imputed) — but the lead over CatBoost is inside the CI, and on grouped data they lose to LightGBM [1]. Hypothesis: holdingly true for grouped/large, contested for temporal.
- **[2] "tuned MLPs lead BeyondArena grouped/temporal/large slices" vs [1]:** on grouped, the leaders are RealMLP/TabM (neural), not trees or FMs — so "trees win where contextual" is true vs FMs, not vs tuned neural nets.
- **Vendor Elo scores** (TabPFN-3.5 1910 "Thinking", LimiX-2 1935, Kumo 1950 in press) vs verified board numbers (TabPFN-3.5 1858, LimiX-2 1946, Kumo 1955): roughly consistent, but [2] warns these are press-release numbers until independently reproduced.

## Gaps

- Large-data regime (>250k rows) unmeasured by both boards — the single biggest tree-friendly territory. OpenML-CTR or custom 1M+ row datasets needed.
- No per-dataset win/loss breakdown pulled (leaderboard CSVs have it one level down); needed to see which dataset types swing tree-vs-FM.
- The `open`-only pool (no closed API/LLM systems) not extracted — the realistic competitive pool for your product.
- Recency of practitioner sentiment (Reddit/HN/X) not surveyed; would validate the "people only use LightGBM, want speed" premise.
- Cost-per-run numbers are inferred from median times [1], not measured on your hardware.

## Recommended next steps

1. **Prototype experiment (falsifiable):** take `autogluon/tabarena`'s harness; axes = {random, grouped, temporal} splits × {<1k, 1k–10k, 10k–250k, >250k} rows × {low-dim, high-dim, high-cardinality} features (BeyondArena's own slices [1]); contenders = LightGBM/XGB/CatBoost tuned+ensembled (equal budget), `perpetual` and a `forust-ml`-algorithm fork, TabPFN-3.5/Kumo/LimiX-2 defaults, and a tree+FM stacked ensemble. Metrics: accuracy/Elo, total wall-clock, inference latency on CPU, GPU cost, completion rate (non-imputed!). Falsifier: FM beats tuned trees on temporal/grouped/large at lower total cost with full completion.
2. **Build the package skeleton on `perpetual`'s layout** (Rust core + maturin/PyO3 + Polars/Arrow interop + typed Python API), implement the XGBoost-parity histogram GBDT from `forust-ml` into it; land parity with LightGBM on TabArena IID before adding novelty.
3. **Novel tree research:** CatBoost-style ordered target statistics (the #1 requested missing feature in Rust GBDT), then C5.0/Cubist-style rule committees + kNN correction for small contextual data.
4. **Track the imputation/missing-run metric** in any comparison — it's the cleanest empirical handle on "FMs don't work in my setting."

## Sources

[1] TabArena Leaderboard CSVs (models pool; BeyondArena `split_temporal`, `split_grouped`) — https://huggingface.co/spaces/TabArena/leaderboard (data as of 2026-10-28) — high (primary).
[2] "Tabular ML in 2026: Trees Still Rule, Foundation Models Rise", Everyday Data Science — https://everydaydatascience.com/article/tabular-foundation-models-vs-gradient-boosting (2026-10-05) — medium (secondary analysis).
[3] TabPFN-3.5: Technical Report, Jäger et al. — https://arxiv.org/abs/2609.17895 (2026-09-15) — high for self-reported claims, vendor bias.
[4] TabArena: A Living Benchmark for ML on Tabular Data, NeurIPS 2025 — https://arxiv.org/abs/2506.16791 — high.
[5] crates.io API + READMEs (perpetual, forust-ml, bosk, smartcore, linfa, miniboosts; pulled 2026-10-28) — high.
[6] Grinsztajn, Oyallon, Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?" — https://arxiv.org/abs/2207.08815 (2022) — high.
[7] "A Closer Look at TabPFN v2" — https://arxiv.org/abs/2502.17361 (2025) — medium-high.
