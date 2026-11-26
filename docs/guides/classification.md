# Binary classification in Python: predict financial distress

Use the Give Me Some Credit dataset to estimate financial distress within two years. The tutorial fits a decision tree classifier, then uses validation data to decide which scores enter a review queue.

## 1. Get the data and define the decision

Download [OpenML 46929](https://www.openml.org/d/46929) to `datasets/GiveMeSomeCredit.parquet`. The target is `FinancialDistressNextTwoYears`. The published table has 150,000 labeled rows. This example uses a seeded 12,000-row subset, split into 8,000 training, 2,000 validation and 2,000 test rows. The test rows are held back for the final report.

Use observed values as supplied. In the sample, monthly income and number of dependents can be missing. Pass them through to the tree. The input schema is numeric and needs no encoder.

**Next:** fit the tree on the 8,000 training rows, use the 2,000 validation rows to choose a threshold, and score the 2,000 test rows for the final report.

## 2. Fit and choose a threshold

This snippet loads a seeded 12,000-row sample of `GiveMeSomeCredit` into a `Dataset`, splits it into 8,000 training, 2,000 validation, and 2,000 test rows, fits a binomial-logit tree, picks a cost-minimizing threshold on validation probabilities, and scores the test set with `MetricsReport`.

<!-- run -->
```python
import pandas as pd
from mltreelib import MetricsReport, ModelTree, dataset_path
from mltreelib.data import Dataset

df = pd.read_parquet(dataset_path("GiveMeSomeCredit"))
target = "FinancialDistressNextTwoYears"
cols = df.to_dict("list")
cols[target] = [1 if v == "Yes" else 0 for v in cols[target]]

ds = Dataset(cols, target=target, name="GiveMeSomeCredit")
sub, _ = ds.split(test_size=ds.n_rows - 12000, seed=2026)
train, val, test = sub.split(val_size=2000, test_size=2000, seed=2026)

tree = ModelTree(
    leaf="mean", leaf_family="binomial", gain="deviance",
    max_depth=3, min_rows=100, prune="none",
)
tree.fit(train)

val_scores = tree.predict_proba(val)
def cost(threshold):
    return sum(
        5 if y and p < threshold else 1 if not y and p >= threshold else 0
        for y, p in zip(val.y, val_scores)
    )

threshold = min([.02, .05, .10, .20, .40, 1.01], key=cost)
scores = tree.predict_proba(test)
alerts = [int(score >= threshold) for score in scores]

rep = MetricsReport("classification", y_true=test.y, y_pred=alerts, proba=scores)
base = MetricsReport(
    "classification",
    y_true=test.y,
    y_pred=[0] * test.n_rows,
    proba=[sum(train.y) / train.n_rows] * test.n_rows,
)
prevalence = sum(test.y) / test.n_rows
print(f"Validation threshold: {threshold:.2f}; flagged test rows: {sum(alerts)} of {len(alerts)}")
print(rep.tables["confusion"])
print(f"Average precision: {rep['average_precision']:.4f} (test prevalence {prevalence:.4f})")
print(f"Brier score: {rep['brier']:.4f} (constant training rate {base['brier']:.4f})")
print(f"Root split: {tree.root_split()['col']}; split columns: {[s['col'] for s in tree.splits()]}")
miss = train.missing()["per_column"]
print(f"Training missing: MonthlyIncome {miss['MonthlyIncome']['n_missing']}, NumberOfDependents {miss['NumberOfDependents']['n_missing']}")
```

<!-- output -->
```text
Validation threshold: 0.10; flagged test rows: 200 of 2000
[{'actual': 0, 'pred 0': 1727, 'pred 1': 138}, {'actual': 1, 'pred 0': 73, 'pred 1': 62}]
Average precision: 0.2647 (test prevalence 0.0675)
Brier score: 0.0542 (constant training rate 0.0629)
Root split: RevolvingUtilizationOfUnsecuredLines; split columns: ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTime30-59DaysPastDueNotWorse', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'RevolvingUtilizationOfUnsecuredLines']
Training missing: MonthlyIncome 1578, NumberOfDependents 205
```

**What to read in the output:** check the validation-chosen threshold, the confusion matrix at that cutoff, average precision against the test prevalence baseline, and Brier score against the constant training-rate baseline. Missing values in `MonthlyIncome` and `NumberOfDependents` stay intact and are routed natively by the tree.

The costs are illustrative. A missed event costs five units and a false alert costs one. A threshold above 1 represents no alerts. The [full script](https://github.com/jkapila/mltreelib/blob/main/examples/guides/real_tree_studies.py) also compares probability error with the training event rate.

```bash
uv run python examples/guides/real_tree_studies.py credit
```

![Credit distress test precision–recall and probability calibration](../assets/guides/real-credit.svg)

**Next:** read the test confusion matrix from step 2 at the validation threshold of 0.10, and compare the tree's average precision and Brier score with the reference values.

## 3. Read the result

The validation threshold was **0.10**. It flagged 200 of 2,000 test rows. Of the 135 observed distress events, 62 were flagged and 73 were missed. The remaining 138 alerts were false positives.

| Test metric | Tree | Reference |
|---|---:|---:|
| Average precision | 0.2647 | 0.0675 event prevalence |
| Brier score | 0.0542 | 0.0629 constant training rate |

The root uses revolving utilization. A later split uses the count of payments 60 to 89 days past due. The training sample contains 1,578 missing income values and 205 missing dependent counts.

Average precision measures how well the scores rank the positive cases. Compare it with test prevalence as a reference. Brier score measures squared probability error. Compare it with the constant training-rate model. The probability plot is a diagnostic. Review bin counts and treat sparsely populated bins with care.

**Next:** review the splits and leaf support behind these alerts, because the metrics in step 3 come from one random historical split.

## 4. Review the model before acting

Use `tree.splits()` to see the original feature names and `tree.leaves()` to check support. Evaluate calibration, decision cost and review volume on realistic future records. This random historical split demonstrates a workflow. An underwriting decision needs time-based validation, group analysis, documented data provenance and governance.

Read the [credit article](../blog/why-credit-scoring-needs-glass-box-trees.md) for a deeper walkthrough of explanation and threshold choice. Use the [evaluation guide](evaluation.md) for the differences between probability, ranking and decision metrics.
