# What tabular foundation models can transfer across tasks

**Date** 2026-11-21

## The gap is sample-efficient adaptation

A model tree learns its splits and leaf coefficients from the data supplied to `fit`. Its split criteria, leaf families, regularization, and stopping rules supply explicit assumptions. With few labeled rows, several rules or combinations of features can explain the same observations.

A tabular foundation model learns a prior across many real or synthetic tasks before it sees the target table. It can condition its predictions on the labeled rows of a new table. If the task resembles those from pretraining, evidence from other tasks helps it choose among relationships that fit the local observations. TabPFN is a documented example of this approach. [TabPFN paper](https://arxiv.org/abs/2207.01848) · [TabPFN v2 analysis](https://arxiv.org/abs/2502.17361)

A model tree can express nonlinear, heterogeneous effects through partitions and local models. The question is how many labels it needs to discover them, especially when useful interactions have weak marginal effects. Ensembles, oblique splits, and feature engineering can reduce this gap. The current ModelTree API does not carry a learned prior from training on other tasks. Its explicit structure is useful when we care which rule survives a group or time shift. [ModelTree reference](../../docs/reference/model.md) · [Planned supervised ensembles](../../specs/14-cubist-committees-and-model-tree-forests.md)

## Official benchmark snapshot

Retrieved November 21 from TabArena's [November 20 data revision](https://huggingface.co/spaces/TabArena/leaderboard/tree/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d). Results below use the models pool, all tasks and datasets, and the IID table without missing-result imputation. The BeyondArena rows use all sizes and feature groups for their respective splits.

| Track | Leader | Best tree-based entry |
| --- | --- | --- |
| [TabArena IID](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data/entrants_models/imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv) | LimiX-2-1016, 2,045 Elo | Tuned and ensembled LightGBM, 1,434 |
| [BeyondArena grouped](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data_beyondarena/subsets/split_grouped/size_all/features_all/tasks_all/website_leaderboard.csv) | Tuned and ensembled RealMLP, 1,248. Kumo-Tabular is the top foundation model at 1,222. | Tuned and ensembled LightGBM, 1,218 |
| [BeyondArena temporal](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data_beyondarena/subsets/split_temporal/size_all/features_all/tasks_all/website_leaderboard.csv) | Kumo-Tabular, 1,175. TabPFN-3.5 follows at 1,174. | Tuned and ensembled CatBoost, 1,130 |

The top foundation and tree confidence intervals overlap in both contextual tracks. In IID, LimiX-2-1016 and LightGBM have a much larger separation. The Elo scale depends on the pool and evaluation protocol. With missing-result imputation enabled, the same IID ranks have scores of 2,006 and 1,411. mltreelib does not appear in these four CSVs. The ranking establishes an advantage on this IID pool and does not quantify a gap for our implementation. The snapshot also does not isolate prior transfer from architecture, compute, or ensembling. [IID with imputation](https://huggingface.co/spaces/TabArena/leaderboard/blob/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data/entrants_models/imputation_yes/splits_all/tasks_all/datasets_all/website_leaderboard.csv) · [TabArena paper](https://arxiv.org/abs/2506.16791)

## TabPFN-4 is an announced preview

Prior Labs' November 19 preview says the model finished training and will be available soon. It reports Elo gains over TabPFN-3.5 of 129 on TabArena, 151 on BeyondArena, and 396 on the BeyondArena subset with at least 100,000 rows. The preview also reports results on additional benchmark families, including multimodal and relational tasks. These scores come from Prior Labs' evaluation. The official November 20 CSVs used above contain no TabPFN-4 result. The preview also describes a sub-quadratic architecture and a run on a 10-million-row table. Its speed comparisons against other models at that scale are extrapolations. [Prior Labs preview](https://priorlabs.ai/technical-reports/tabpfn-4-preview) · [Public TabPFN repository](https://github.com/PriorLabs/TabPFN)

## What to test in mltreelib

- Plot learning curves with the same labeled rows, features, and held-out rows. Include random, grouped, and temporal splits. Compare ModelTree with a tuned GBDT and a released foundation model. Record tuning budget, hardware, calibration, time, and completion.
- Use generated data with planted categorical or hierarchical structure. Measure whether the model recovers that structure alongside predictive accuracy.
- Compare local fitting with a model that can reuse a prior. Report how the gap changes with sample size before attributing the result to a representational limit. [TabArena evaluation](https://arxiv.org/abs/2506.16791) · [Classical tree benchmark](https://arxiv.org/abs/2207.08815)

## Released-model method comparison

- **LimiX-2** uses context-conditional masked modeling on synthetic tables generated from structural causal models. This trains it to model joint feature and target relationships given context. Its paper reports causal skeleton experiments under its evaluation assumptions. [Technical report](https://arxiv.org/abs/2609.17488v2)
- **Kumo-Tabular** uses synthetic-only pretraining. The authors describe predicting on a target table by passing labeled and query rows through the pretrained model without per-table training. [NVIDIA model article](https://huggingface.co/blog/nvidia/kumo-tabular)
- **TabPFN-3.5** conditions predictions on labeled rows through pretrained weights. Its technical report evaluates grouped and temporal data, wide tables, high-cardinality categoricals, and multimodal inputs. The report distinguishes the default model from Plus and Thinking variants. Record the exact variant and its inference budget. [Technical report](https://arxiv.org/abs/2609.17895v3) · [Client and weights documentation](https://github.com/PriorLabs/TabPFN)

ModelTree discovers interactions from the current table. Users prepare text, images, and relational tables as features before fitting. This gives a direct target-table baseline for measuring what reusable representations and prior knowledge add. [ModelTree reference](../../docs/reference/model.md)

Use the papers for method descriptions and the pinned CSVs for comparative scores. Vendor results describe their own evaluation. A larger gap at small label counts would support the sample-efficiency hypothesis. Isolating the effect of pretraining requires an otherwise matched ablation.

## mltreelib coverage and benchmark gap

The overall arena combines **18 regression and 38 classification tasks**. The regression component uses test RMSE and pairwise Elo with `K=32`. The classification component uses test accuracy. [Overall board](../../docs/benchmarks/elo_overall.md) · [Regression board](../../docs/benchmarks/elo_board.md) · [Classification board](../../docs/benchmarks/elo_cla_board.md)

| Historical variant | Overall Elo | Regression scores | Classification scores |
| --- | ---: | ---: | ---: |
| ModelTree(all) | 1,333.9 (rank 7) | 16/18 | 38/38 |
| ModelTree(opt) | 1,036.6 | 18/18 recorded, 17 finite | 38/38 |
| ModelTree(mean) | 1,013.9 | 18/18 finite | 38/38 |

The local attempt logs record all 56 attempts for ModelTree(all). Its two missing regression scores are timeout entries for `ta_qsar_tid11` and `ta_superconductivity`. The ModelTree(opt) regression CSV records infinite RMSE for `fc2008`. The overall board reports coverage by score rows, so the distinction matters when comparing successful runs.

On the same overall board, ModelTree(all) trails CatBoost(200) by **143.6 Elo** and leads LightGBM(200) by **96.2 Elo**. Both comparators have 56/56 recorded scores. On the regression component alone, ModelTree(all) trails LightGBM(200) by **402.6 Elo**. The **1,333.9 / 54-of-56** row is already present in tag v0.4.7, whose package manifest declares 0.4.6. Tags v0.4.8 through v0.4.11 carry the same row. The score files do not identify an exact package build for the historical runs. [Board at v0.4.7](https://github.com/jkapila/mltreelib/blob/v0.4.7/docs/benchmarks/elo_overall.md) · [Archived model variants](../../benchmarks/archive/legacy_models.py)

The canonical local TabArena manifest lists **51 datasets**, with **13 regression**, **38 classification**, and a maximum of **150,000 rows**. The historical 18-dataset regression arena also includes tasks outside that manifest. Some tasks appear under local names. The local board and the public leaderboard use different entrant pools and protocols. **The v0.4.11 gap to the public foundation-model leaders is unmeasured.** [Canonical manifest](../../datasets/MANIFEST.md) · [TabArena protocol](https://arxiv.org/abs/2506.16791)

The repository separately tests survival, competing risks, and causal inference. These tests establish behavior and API coverage. [Survival and competing risks tests](../../tests/test_survival_competing_risks.py) · [Causal tree tests](../../tests/test_causal_tree.py)

### Prioritized external slate

1. **TabArena for the headline comparison.** Run a small pilot on verified task IDs for Concrete, Miami Housing, Credit-G, APSFailure, and Anneal. Use the benchmark's published splits, targets, and task metrics. Record default and tuned results separately. [TabArena guide](https://tabarena.ai/code-examples)
2. **BeyondArena for distribution shift and scale.** Run selected grouped and temporal tasks, then add large tables. Preserve the entity and time boundaries. Report CPU and GPU budgets, prediction latency, and completed runs. [BeyondArena results and task subsets](https://huggingface.co/spaces/TabArena/leaderboard/tree/87bdf9f8c91a21ecfd2a23184aabcf7c75fe612d/data_beyondarena)
3. **TabZilla and OpenML-CTR23 for broader coverage.** Use TabZilla to stress classification and CTR23 to extend regression coverage. Check overlap by dataset and task version before combining scores. [TabZilla benchmark](https://github.com/naszilla/tabzilla/blob/main/docs/DATASHEET.md) · [OpenML-CTR23 task suite](https://www.openml.org/search?id=353&study_type=task&type=study)
4. **SurvSet for censored outcomes.** Compare concordance, calibration, and integrated Brier score under fixed splits and horizons. Use an explicitly defined competing-risk endpoint for cause-specific calibration. [SurvSet](https://github.com/ErikinBC/SurvSet)
5. **Generated structure for the research thesis.** Plant categorical groups, hierarchy, missingness and known interactions or treatment effects. Measure structure and effect recovery separately from prediction error. [Generator implementation](../../crates/mltreelib-core/src/generator.rs)

The first matched experiment uses the pilot tasks, ModelTree v0.4.11, a strong GBDT, and one released foundation model under a fixed split and budget. Keep the test set fixed when thinning the training labels. Save per-task loss, coverage, and timing before calculating any aggregate rank. Record the baseline model variant and any evidence of pretraining overlap.
