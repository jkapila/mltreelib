# A real-data walkthrough: bank churn

This example uses the **Bank Customer Churn** dataset from TabArena-v0.1, distributed through [OpenML dataset 46911](<https://www.openml.org/d/46911>). The repository's [dataset manifest](<https://jkapila.github.io/mltreelib/benchmarks/datasets/index.md>) describes the local benchmark collection. This is a single exploratory split, not a leaderboard comparison.

The dataset has 10,000 records and a churn label. The example uses seven features, including raw `country` and `gender` category labels. It fits classification, clustering and anomaly workflows through the package's public interfaces. The unsupervised models do **not** use the churn labels during fitting.

## Get the data and run

The parquet files are not checked into Git. Download the dataset from OpenML and place it at `datasets/Bank_Customer_Churn.parquet`. See the [dataset manifest](<https://github.com/jkapila/mltreelib/blob/main/datasets/MANIFEST.md>) for the collection and source links. Install pandas and matplotlib as well as the package; pyarrow is a package dependency.

```
uv run python examples/real_data_workflow.py
```

The [complete script](<https://github.com/jkapila/mltreelib/blob/main/examples/real_data_workflow.py>) shuffles rows with seed 2026, then uses 6,000 training, 2,000 validation and 2,000 test rows. It passes category labels and missing entries directly to the models. The selected training columns had no missing values in this run, so this example does not test missing-value routing.

## Classification: compare decisions and probabilities

A binomial `ModelTree` predicts churn. The validation set chooses among four thresholds with an illustrative cost of 3 for a missed event and 1 for a false alert. The test set remains separate from threshold selection. The constant-probability baseline uses the training event rate.

| Held-out result | Value |
| --- | ---: |
| Training churn rate | 0.2017 |
| Validation-selected threshold | 0.2 |
| Test confusion matrix, actual rows and predicted columns | `[[1430, 157], [187, 226]]` |
| Test average precision | 0.5238 |
| Test Brier score, model | 0.1215 |
| Test Brier score, constant training-rate baseline | 0.1639 |

These are results for one random split and configuration. They do not establish generalization to future customers, a new bank or a temporal deployment. Calibration is assessed by the diagram; the package does not fit a post-hoc calibrator in this example.

## Clustering and anomaly review

`ClusterTree(kind="pct")` fits the same training features and assigns the held-out records. This run produced four groups with 444, 530, 561 and 465 test records. `rules()` returned four descriptions. The features have different scales, and the example does not establish whether the resulting groups are stable or useful to the business. Check profiles, scale sensitivity, stability and category labels before acting.

`AnomalyTree(kind="eif")` scored all 2,000 held-out records. The five highest scores were 0.811, 0.811, 0.778, 0.778 and 0.778. A churn label is not an anomaly label. Without reviewed cases or a suitable independent anomaly outcome, these scores cannot establish anomaly precision or recall. Use a defined review budget and collect reviewer outcomes.

## Real-data regression

The [regression pathway](<https://jkapila.github.io/mltreelib/guides/regression/index.md>) first compares four settings on synthetic data with missing values. It then explains the selected setting on the real OpenML diamonds dataset, with 6,000 training and 2,000 test records. Synthetic selection illustrates the workflow and does not establish the best setting for diamonds.

## What this does and does not answer

This real-data example checks that the interfaces run on a mixed customer dataset and produces an independent test report. The synthetic examples remain useful for testing known planted anomalies, treatment effects and censoring. For stronger evidence, repeat across seeds and datasets, use time- or entity-based splits when deployment requires them, and compare with appropriate baselines. Specialized survival and uplift workflows require datasets with censoring and treatment information; a generic churn label does not supply those fields.

[Data preparation](<https://jkapila.github.io/mltreelib/guides/data/index.md>) · [Classification](<https://jkapila.github.io/mltreelib/guides/classification/index.md>) · [Clustering](<https://jkapila.github.io/mltreelib/guides/clustering/index.md>) · [Anomaly detection](<https://jkapila.github.io/mltreelib/guides/anomaly/index.md>) · [Evaluation](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>)
