# A single-tree comparison with boosting and tabular foundation models

Date 2026-11-21.

## Public Elo

These are default foundation models from the November 20 TabArena data revision
`87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d`. BeyondArena uses the all-sizes,
all-features grouped and temporal subsets. Its LimiX rows include imputed scores.

| Public entrant | TabArena IID | BeyondArena grouped | BeyondArena temporal |
| --- | ---: | ---: | ---: |
| LimiX-2-1016 | 2,045 | 1,169 (22.22% imputed) | 950 (57.14% imputed) |
| NVIDIA Kumo-Tabular | 2,018 | 1,222 | 1,175 |
| TabPFN-3.5 | 1,908 | 1,160 | 1,174 |
| LightGBM, tuned and ensembled | 1,434 | 1,218 | 1,127 |
| CatBoost, tuned and ensembled | 1,424 | 1,184 | 1,130 |

ChimeraBoost has 1,323 Elo at default settings and 1,388 with tuning and
ensembling on the public IID board. The IID CSV excludes missing-result
imputation. Elo depends on the entrant pool and the matches available in each
track. The local board cannot quantify a difference from those Elo numbers.
[IID CSV](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data/entrants_models/imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv) ·
[Grouped CSV](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data_beyondarena/subsets/split_grouped/size_all/features_all/tasks_all/website_leaderboard.csv) ·
[Temporal CSV](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data_beyondarena/subsets/split_temporal/size_all/features_all/tasks_all/website_leaderboard.csv)

## Separate the one-tree ceiling from ensemble gains

Use a single ModelTree with CatBoost and LightGBM as the required local
baselines. Include ChimeraBoost where it runs. Report paired test loss,
train and prediction time, and model size. A comparison with boosted models
measures how competitive one tree can be under a simplicity constraint.
An equal-budget comparison answers a separate question. Keep track of whether
each baseline uses defaults, tuning, or an ensemble of tuned configurations.

LightGBM grows the leaf that offers the largest loss reduction. Uneven depth
lets it spend capacity where it pays off. It can also overfit thin leaves.
CatBoost uses ordered statistics to make target-derived categorical features.
Those statistics can include combinations of categorical features. Its common
symmetric-tree configuration illustrates a different trade-off. The same split
condition applies across a level of that tree.
[LightGBM features](https://lightgbm.readthedocs.io/en/stable/Features.html) ·
[CatBoost categorical transforms](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic) ·
[CatBoost tree policies](https://catboost.ai/docs/en/references/training-parameters/common#grow_policy)

Once a single-tree recipe is pinned, average a small number of independently
seeded, bootstrapped trees with feature subsampling. Hold the base recipe fixed.
An average reduces variance when trees make different errors. It can also
reduce the impact of an unstable split. Shared bias and correlated errors
limit the gain. Averaging increases model size and prediction work. Record
those costs before testing a boosting scheme. This experiment covers the
LinearForest portion of Spec 14. [Spec 14](../../specs/14-cubist-committees-and-model-tree-forests.md)

## Reuse public results on official splits

Run our single tree on the exact public task versions, folds, and metrics. Reuse
published losses for those same folds and fixed model variants where available.
This allows per-task comparisons without rerunning released peers. Record
missing runs and leakage flags. An official-scale Elo needs the benchmark's
aggregation and entrant pool. A pilot on matched tasks can report paired loss
and win rates with fewer assumptions. The local random split for Diamonds
provides a useful first comparison between local entrants.
[TabArena evaluation](https://arxiv.org/abs/2506.16791) ·
[Benchmark artifacts](https://tabarena.ai/code-examples) ·
[Corrected local Diamonds scores](../../benchmarks/runs/elo-v0.4.11-all/diamonds-peers/scores.csv)

## Corrected local results

All 35 historical Diamonds entrants produced finite scores on the corrected
price split. Other peer results remain frozen. The revised 56-dataset arena
places ModelTree(all) ninth at 1,290.1 Elo. The pairwise counts below compare
ModelTree directly with each peer. The aggregate rating depends on the full
field and match order.

| Peer | Corrected Diamonds RMSE | Regression wins–ties–losses | Classification accuracy wins–ties–losses | Classification log loss wins–ties–losses |
| --- | ---: | ---: | ---: | ---: |
| CatBoost(200) | 537.58 | 0–0–18 | 10–1–27 | 2–0–36 |
| LightGBM(200) | 539.08 | 2–0–16 | 13–1–24 | 4–0–34 |
| ChimeraBoost | 520.64 | 0–0–18 | 10–2–26 | 1–0–37 |

ModelTree(all) scored 761.10 RMSE on corrected Diamonds. Its Elo is higher than
LightGBM(200) in the full field. LightGBM wins their direct comparison in
regression, classification accuracy, and classification log loss. Paired
results give a clearer baseline for the single-tree improvement question.
[Corrected overall board](../../docs/benchmarks/elo_overall.md) ·
[Paired results](../../benchmarks/runs/elo-v0.4.11-all/diamonds-peers/focused_comparison.json)

The leaderboard Space publishes aggregate CSVs. The upstream repository
documents Results artifacts and a per-split export script. Check the Results
artifacts for each exact entrant before scheduling its rerun. The availability
of per-fold losses for all three November 20 variants remains unconfirmed.
[Space export documentation](https://huggingface.co/spaces/TabArena/leaderboard/blob/main/AGENTS.md) ·
[Upstream artifact catalog](https://github.com/autogluon/tabarena#-tabarena-artifacts)
