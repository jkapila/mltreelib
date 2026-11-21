# ModelTree(all) Elo rerun with mltreelib v0.4.11

Date 2026-11-21. Scope is the repository's 56-dataset arena.

The v0.4.11 wheel produced finite predictions for all 208,740 test rows. Each
of the 18 regression datasets and 38 classification datasets produced a finite
primary score. The runner used a 180-second wall-clock limit per dataset.
[Manifest](../../benchmarks/runs/elo-v0.4.11-all/manifest.json) ·
[Regression scores](../../benchmarks/runs/elo-v0.4.11-all/reg_scores.csv) ·
[Classification scores](../../benchmarks/runs/elo-v0.4.11-all/cla_scores.csv) ·
[Regression attempts](../../benchmarks/runs/elo-v0.4.11-all/reg_attempts_180s.csv) ·
[Classification attempts](../../benchmarks/runs/elo-v0.4.11-all/cla_attempts_180s.csv)

| Previously missing regression dataset | Test RMSE | Fit and predict |
| --- | ---: | ---: |
| QSAR-TID-11 | 1.0604 | 1.0024 s |
| Superconductivity | 18.1229 | 0.4863 s |

The run used the ModelTree(all) factories from tag `v0.4.7` with v0.4.11
constructor defaults. It retained the classification factory's high-dimensional
leaf safeguards. The historical score files do not pin the model build or split
hashes. Runtime and score changes cannot be attributed to a particular code change
from these records. [Regression recipe](../../benchmarks/runs/elo-v0.4.11-all/recipe_reg.py) ·
[Classification recipe](../../benchmarks/runs/elo-v0.4.11-all/recipe_cla.py)

## Diamonds target correction

The original preparation assigned `price` to a column named `y`. Diamonds
already has a feature named `y`, so the assignment replaced that feature and
left `z` as the last column. The Python scorer treated `z` as the target.
I regenerated this split from the Diamonds parquet with seed 7. The corrected
split retains all nine features and appends `price` as the target. ModelTree(all)
scored **761.1020 RMSE** on 10,788 test cases. The preparation script now
preserves the feature, and the R runners use the last column as the target.
[Preparation](../../examples/benchmarks/elo/elo_prepare.py) ·
[Corrected scores](../../benchmarks/runs/elo-v0.4.11-all/reg_scores.csv)

## Elo against the frozen field

The historical CSVs and board generator at tag `v0.4.7` reproduce every row
of the published v0.4.11 overall board. The comparison drops Diamonds from
both boards because the archived competitors lack valid price results. It uses
regression RMSE and classification accuracy in the original dataset order.

| ModelTree(all) result | Overall Elo | Rank | Scores included |
| --- | ---: | ---: | ---: |
| Published board | 1,333.9 | 7 | 54 of 56 |
| Historical board with Diamonds removed | 1,335.7 | 7 | 53 of 55 |
| v0.4.11 rerun with Diamonds removed | 1,289.7 | 9 | 55 of 55 |

The rerun trails frozen CatBoost(200) by 167.6 Elo and leads frozen
LightGBM(200) by 49.3 Elo. Regression Elo drops from 1,123.8 to 717.9.
Classification accuracy Elo drops from 1,244.3 to 1,204.7. These Elo changes
include the effect of two added regression scores. The round measures local
arena performance. It does not establish a new score on the public TabArena
leaderboard. [Historical published board](https://github.com/jkapila/mltreelib/blob/v0.4.11/docs/benchmarks/elo_overall.md) ·
[Summary](../../benchmarks/runs/elo-v0.4.11-all/summary.json)

## Corrected Diamonds peer round

The follow-up round used the same 43,152 training and 10,788 test cases.
The field contains 35 entrants from the archived Diamonds rows. The driver
reused the v0.4.11 ModelTree(all) score and scored the other 34 entries.
It ran one fit at a time with a 300-second budget. The v0.3.3 entry used a
release build from commit `8ad142b`. External packages use the versions
available in the benchmark environments. The completed board ranks ModelTree(all) ninth at 1,290.1 Elo over 56 datasets. [Peer scores](../../benchmarks/runs/elo-v0.4.11-all/diamonds-peers/scores.csv) ·
[Completed board](../../docs/benchmarks/elo_overall.md)

## What the rerun establishes about foundation models

The same-field result puts ModelTree(all) between frozen CatBoost(200) and
LightGBM(200) on the combined board. The regression component is its weaker
side in this round. The public IID board measures a different set of matches.
Its foundation-model lead cannot quantify the gap to ModelTree. The much tighter
grouped and temporal results show that split design matters for this comparison.
[Local results](../../benchmarks/runs/elo-v0.4.11-all/summary.json) ·
[Pinned public results and method sources](2026-11-21-tree-vs-tabular-foundation-models.md)

Tabular foundation models such as TabPFN learn expectations about tables
before they see the target task. With limited labels, a pretrained model can use
these expectations to select among several explanations that fit the observed
rows. Attention can also condition each query on the relevant labeled rows and
feature combinations. ModelTree learns splits and leaf parameters from the
target table. Its explicit regularization and leaf families provide assumptions
that a user configures. A more powerful leaf solver can improve this fit. It
does not supply experience from other tables. [TabPFN](https://arxiv.org/abs/2207.01848) ·
[TabPFN v2](https://arxiv.org/abs/2502.17361) ·
[ModelTree API](../../docs/reference/model.md)

If a target table differs from the pretraining tasks, the prior can favor the
wrong structure. The cost of context inference depends on the model, data
shape, and inference budget. A fitted ModelTree can serve a new row by
following one path and evaluating its leaf. The tree also exposes the rules and
local coefficients for inspection. Use predictive loss, calibration, memory,
and latency to compare these trade-offs on one fixed split. Use grouped or
temporal holdouts to check whether gains carry over under shifts.
[BeyondArena results](https://huggingface.co/spaces/TabArena/leaderboard/tree/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data_beyondarena) ·
[ModelTree API](../../docs/reference/model.md)
