# TabPFN-4 preview benchmarks and a comparison plan for ModelTree

Date 2026-11-21.

The screenshot contains ten result rows across eight benchmark suites.
The large-table row is a BeyondArena subset. Text and image are separate
MulTaBench tracks. Prior Labs reports TabPFN-4 ahead in nine rows. LimiX-2
leads TabArena by 10 Elo, and the report puts the leading models within
each other's error bars. Each row has its own field and benchmark protocol.
Read Elo within a row. [Prior Labs preview](https://priorlabs.ai/technical-reports/tabpfn-4-preview)

## Scores in the screenshot

| Benchmark | TabPFN-4 | TabPFN-3.5 | Kumo-Tabular | Best other |
| --- | ---: | ---: | ---: | --- |
| BeyondArena | 1,521 | 1,370 | 1,427 | 1,357 Seldon |
| BeyondArena, tables >100k rows | 1,560 | 1,164 | 1,159 | 1,268 RealMLP |
| TabArena | 1,984 | 1,855 | 1,972 | 1,994 LimiX-2 |
| ScoringBench (CRPS) | 2,257 | 1,938 | 2,228 | 2,050 LimiX-2 |
| STRABLE | 1,879 | 1,620 | 1,828 | 1,761 Chakra-Tab M |
| MulTaBench text | 1,843 | 1,637 | 1,813 | 1,790 LimiX-2 |
| MulTaBench image | 1,874 | 1,762 | 1,856 | 1,850 LimiX-2 |
| RelArena-α | 2,055 | 1,914 | — | 1,908 kurversc |
| RamanBench | 1,619 | 1,429 | 1,556 | 1,510 Chakra-Tab |
| TabBench-Bio | 1,270 | 1,173 | 1,210 | 1,157 Seldon-Mini |

The preview uses four TabPFN-4 estimators and one seed by default.
The TabBench-Bio row averages five seeds and reports no confidence interval.
Columns use the best entry from each family. The visual reports a preview
run. The public November 20 IID CSV uses a different field and setup.
It lists LimiX-2-1016 at 2,045, Kumo-Tabular at 2,018, and TabPFN-3.5 at
1,908. [Preview protocol](https://priorlabs.ai/technical-reports/tabpfn-4-preview) ·
[Pinned independent leaderboard](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data/entrants_models/imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv)

## What the benchmarks measure

| Suite | Measurement and screenshot conditions | Use in a ModelTree evaluation |
| --- | --- | --- |
| TabArena | IID classification and regression on 51 curated OpenML datasets. The public metrics are ROC-AUC for binary classification, log loss for multiclass, and RMSE for regression. | First accuracy baseline. Use the official fold IDs and compare one tree with CatBoost, LightGBM, and ChimeraBoost where available. [Paper](https://arxiv.org/abs/2506.16791) · [Artifacts](https://github.com/autogluon/tabarena#-tabarena-artifacts) |
| BeyondArena | Includes IID, grouped, and temporal splits. Grouped splits hold out entities and temporal splits hold out future rows. The screenshot identifies the group ID as a feature for TabPFN-4. Its >100k subset also stresses scale. | Test the same boundary with a shared feature policy. For large tasks record full completion, runtime, peak memory, and inference latency. [Data and splits](https://huggingface.co/datasets/TabArena/BeyondArena) · [Public results](https://huggingface.co/spaces/TabArena/leaderboard/tree/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data_beyondarena) · [Preview](https://priorlabs.ai/technical-reports/tabpfn-4-preview) |
| ScoringBench (CRPS) | Scores the predictive distribution of a continuous outcome. CRPS integrates the squared difference between the forecast cumulative distribution and the step function at the observed outcome. Lower is better. The 3.5 report used 101 datasets and averaged folds before ranking. The 4 preview supplies the Elo row without a separate split manifest. | Evaluate predictive distributions on the repository's splits. Keep CRPS alongside RMSE. [Official benchmark](https://github.com/jonaslandsgesell/ScoringBench) · [3.5 technical report](https://arxiv.org/abs/2609.17895) · [4 preview](https://priorlabs.ai/technical-reports/tabpfn-4-preview) |
| STRABLE | 108 real-world tables with string and numeric columns. The public package includes three folds per table and reports R² for regression and ROC-AUC for classification. The preview supplies TF-IDF text features to all entrants. | Freeze the text vocabulary and preprocessing on training rows. Give each learner the same derived features. [Official benchmark](https://github.com/soda-inria/strable) · [Paper](https://arxiv.org/abs/2605.12292) |
| MulTaBench text and image | The official core set has 20 text and 20 image tables with complementary tabular signal. The preview fixes an e5-small text encoder and a DINOv3-small image encoder. Its image row explicitly reports 40 datasets, broader than the core image set. | Compare tabular-only and shared-embedding runs. Record the exact dataset subset, encoder, and task metric. [Official benchmark](https://github.com/alanarazi7/MulTaBench) · [Paper](https://arxiv.org/abs/2605.10616) · [Preview](https://priorlabs.ai/technical-reports/tabpfn-4-preview) |
| RelArena-α | The alpha benchmark covers 21 RelBench tasks across seven databases. It limits training and evaluation to the data available at each cutoff. The preview uses TabPFN-Rel, a feature synthesis and context-selection pipeline. Kumo has no result in this row. | Use a shared, time-safe feature table for a learner comparison. Measure the full relational pipeline separately. [Protocol](https://star-project.stanford.edu/relarena-alpha/) · [Repo](https://github.com/PriorLabs/relarena) |
| RamanBench | 74 spectroscopy datasets with regression and classification. The public benchmark provides a standardized evaluation and aggregates results across datasets. | Domain-specific stress test for wide, correlated numeric inputs. Follow the dataset-specific splits and preprocessing. [Official benchmark](https://github.com/ml-lab-htw/RamanBench) · [Paper](https://arxiv.org/abs/2605.02003) |
| TabBench-Bio | Wide biomedical tabular tasks evaluated under training-sample and feature budgets. The live site lists five-fold cross-validation and 46 datasets. Metrics include macro-F1 for classification and RMSE for regression. Elo is anchored to random forest at 1,000. | Optional sample-efficiency check on released data. Pin the benchmark version and budget grid. [Official benchmark](https://tabbench-bio.eu/) · [Paper](https://arxiv.org/abs/2609.07441) |

ScoringBench's published default protocol uses five-fold cross-validation
with seed 42 and caps each repeat at 3,000 rows, including 2,400 for training
and 600 for testing on capped datasets. It writes a dataset manifest for each
run. Pin that manifest and the benchmark revision for comparisons.
[ScoringBench experimental protocol](https://arxiv.org/html/2603.29928)

## Comparison order

1. Start with **official TabArena folds**. Keep ModelTree as a single tree
   and compare paired loss, fit time, prediction latency, and model size against
   CatBoost and LightGBM. Add ChimeraBoost where the same results are available.
   Reuse any public per-fold loss that matches task version, split, metric, and
   model variant. Keep public aggregate Elo as a reference. The leaderboard
   Space does not host per-split rows. Check the upstream Results or Raw
   artifacts for the exact entrant before paying to rerun it.
2. Add one **grouped** and one **temporal** BeyondArena task. Keep group
   identifiers under a common feature policy. Compare multiple label budgets
   on fixed test rows to measure sample efficiency. Add the >100k subset
   after the smaller runs establish the protocol.
3. Choose a specialty benchmark for the feature under test. Use ScoringBench
   for predictive distributions, STRABLE for text-derived features, and the
   relational or multimodal tracks once the input pipeline is part of the
   evaluation. Keep the single-tree recipe pinned before testing bagging.

[TabArena result exports](https://huggingface.co/spaces/TabArena/leaderboard/blob/main/AGENTS.md) ·
[Single-tree plan](2026-11-21-single-model-tree-benchmark.md) ·
[Foundation-model analysis](2026-11-21-tree-vs-tabular-foundation-models.md) ·
[Forest spec](../../specs/14-cubist-committees-and-model-tree-forests.md)

## View the corrected local Elo board

Run this from the repository root in a shell.

```bash
uv run --no-cache --no-sync python benchmarks/runs/elo-v0.4.11-all/diamonds-peers/compare.py --table | less -S
```

The command rebuilds the 47-entrant table from saved scores. It includes
the corrected Diamonds peer results and reports ModelTree(all) ninth at
1,290.1 Elo. Omit `--table` for the focused machine-readable summary.
[Corrected full board](../../docs/benchmarks/elo_overall.md) ·
[Comparison script](../../benchmarks/runs/elo-v0.4.11-all/diamonds-peers/compare.py)
