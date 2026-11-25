# Binary classification in Python: predict financial distress

Use the Give Me Some Credit dataset to estimate financial distress within two years. The tutorial fits a decision tree classifier, then uses validation data to decide which scores enter a review queue.

## 1. Get the data and define the decision

Download [OpenML 46929](https://www.openml.org/d/46929) to `datasets/GiveMeSomeCredit.parquet`. The target is `FinancialDistressNextTwoYears`. The published table has 150,000 labeled rows. This example uses a seeded 12,000-row subset, split into 8,000 training, 2,000 validation and 2,000 test rows. The test rows are held back for the final report.

Use observed values as supplied. In the sample, monthly income and number of dependents can be missing. Pass them through to the tree. The input schema is numeric and needs no encoder.

**Next:** fit the tree on the 8,000 training rows, use the 2,000 validation rows to choose a threshold, and score the 2,000 test rows for the final report.

## 2. Fit and choose a threshold

<!-- run -->
```python
import pandas as pd
from mltreelib import Metrics, ModelTree
from mltreelib.data import missing_profile, select_rows, take, train_test_split, train_val_test_split, xy

from mltreelib import dataset_path
df = pd.read_parquet(dataset_path("GiveMeSomeCredit"))
target = "FinancialDistressNextTwoYears"

# Seeded 12,000-row subset, then seeded 8,000 train / 2,000 validation / 2,000 test
sub, _ = train_test_split(len(df), test_size=len(df) - 12000, seed=2026)
sample = take(df.to_dict("list"), sub)
d = xy(sample, target)
train_idx, val_idx, test_idx = train_val_test_split(12000, val_size=2000, test_size=2000, seed=2026)

def part(idx):
    rows = select_rows(d, idx)
    return rows["x"], [1 if v == "Yes" else 0 for v in rows["y"]]

x_train, y_train = part(train_idx)
x_val, y_val = part(val_idx)
x_test, y_test = part(test_idx)

tree = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance",
                 max_depth=3, min_rows=100, prune="none")
tree.fit(x_train, y_train, col_names=d["names"], dtypes=d["dtypes"])

val_scores = tree.predict_proba(x_val)
def cost(threshold):
    return sum(5 if y and p < threshold else 1 if not y and p >= threshold else 0
               for y, p in zip(y_val, val_scores))

threshold = min([.02, .05, .10, .20, .40, 1.01], key=cost)
scores = tree.predict_proba(x_test)
alerts = [int(score >= threshold) for score in scores]

metrics = Metrics()
prevalence = sum(y_test) / len(y_test)
print(f"Validation threshold: {threshold:.2f}; flagged test rows: {sum(alerts)} of {len(alerts)}")
print(metrics.classification(y_test, alerts)["confusion"])
print(f"Average precision: {metrics.pr_curve(y_test, scores)['average_precision']:.4f} (test prevalence {prevalence:.4f})")
train_rate = sum(y_train) / len(y_train)
print(f"Brier score: {metrics.brier_score(y_test, scores):.4f} (constant training rate {metrics.brier_score(y_test, [train_rate] * len(y_test)):.4f})")
print(f"Root split: {tree.root_split()['col']}; split columns: {[s['col'] for s in tree.splits()]}")
miss = missing_profile(take(sample, train_idx))["per_column"]
print(f"Training missing: MonthlyIncome {miss['MonthlyIncome']['n_missing']}, NumberOfDependents {miss['NumberOfDependents']['n_missing']}")
```

<!-- output -->
```text
Validation threshold: 0.10; flagged test rows: 200 of 2000
[[1727.0, 138.0], [73.0, 62.0]]
Average precision: 0.2647 (test prevalence 0.0675)
Brier score: 0.0542 (constant training rate 0.0629)
Root split: RevolvingUtilizationOfUnsecuredLines; split columns: ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTime30-59DaysPastDueNotWorse', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'RevolvingUtilizationOfUnsecuredLines']
Training missing: MonthlyIncome 1578, NumberOfDependents 205
```

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
