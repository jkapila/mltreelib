# Explainable Decision Stumps & Leaf Retraining

A single decision stump (`max_depth=1`) splits feature space into two transparent regimes, then fits independent models inside each leaf. This pattern replicates the high-explainability output of **Quinlan's Cubist and C5.0** — models senior executives, actuaries, and clinical leaders understand in seconds.

With `tree.retrain(...)`, leaf models can be refitted across families (`linear`, `huber`, `quantile`, `binomial`, `poisson`) on frozen partitions in **microseconds**, completely decoupling split finding from leaf estimation.

Script: [`examples/practical_stump_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/practical_stump_showcase.py).

---

## Real-World Case: Healthcare Insurance Expenses

Using the TabArena `healthcare_insurance_expenses` benchmark (1,338 patients with `age`, `bmi`, `children`, `smoker` predicting medical `charges`):

```python
from mltreelib import ModelTree, LeafControl
from mltreelib.datasets import load_dataset

d = load_dataset("healthcare_insurance_expenses")
cols = ["age", "bmi", "children", "smoker"]
x = [[d["columns"][c][i] for c in cols] for i in range(d["n_rows"])]
y = d["columns"]["charges"]

stump = ModelTree(max_depth=1, min_rows=20, leaf="linear", l2=1e-3, leaf_digest=True)
stump.fit(x, y, dtypes=["numeric", "numeric", "numeric", "categorical"])
```

### Cubist / C5.0 Rule Presentation

```text
==============================================================================
ModelTree Explainable Rules (Cubist / C5.0 Style)
==============================================================================

Rule 1: [1,064 cases, mean $8,434.27, range $1,121.87 to $36,910.61, std $5,990.96]
    if
        smoker in {no}
    then
        charges = $-2,809.30 + 265.29 * age + 5.28 * bmi + 580.66 * children

Rule 2: [274 cases, mean $32,050.23, range $12,829.46 to $63,770.43, std $11,520.47]
    if
        smoker in {yes}
    then
        charges = $-22,556.08 + 264.93 * age + 1,438.73 * bmi + 198.88 * children
==============================================================================
```

### Practical Insights Delivered Instantly

1. **Identical Aging Velocity**:
   - Non-smokers: `+$265.29 / year`
   - Smokers: `+$264.93 / year`
   - Both populations accumulate baseline medical risk at identical rates (~$265 annually).

2. **Compounding Metabolic Risk**:
   - Non-smokers: `+$5.28 / BMI unit` (essentially flat).
   - Smokers: **`+$1,438.73 / BMI unit`** (**270× steeper**!).
   - In smokers, high BMI triggers catastrophic cardiovascular and pulmonary claims. A flat linear model would average these slopes (~$330/BMI) and miss the entire interaction.

3. **Base Claim Shift**:
   - Non-smoker average: **$8,434**
   - Smoker average: **$32,050** (nearly 4× higher).

---

## Frozen-Topology Retraining (`tree.retrain`)

Re-partitioning 1,338 rows across feature candidates takes ~38 ms. But once split boundaries are set, leaf models can be swapped and refitted in **< 1 ms**:

```python
# 1. Robust Huber M-estimator (outlier bill resistance)
stump.retrain(x, y, leaf="huber")

# 2. Actuarial Quantile Bands (10th, 50th, 90th percentile reserves)
stump.retrain(x, y, leaf="linear", l2=1e-3, leaf_digest=True)
q10 = stump.quantile(test_cases, 0.10)
q50 = stump.quantile(test_cases, 0.50)
q90 = stump.quantile(test_cases, 0.90)

# 3. Regularized Ridge Leaves (shrinkage under small sample partitions)
stump.retrain(x, y, leaf="linear", l2=10.0)

# 4. Piecewise Constant CART baseline
stump.retrain(x, y, control=LeafControl(leaf="mean"))
```

### Prediction Comparison on a 35yo Patient (BMI 28)

| Model Family | Non-Smoker Prediction | Smoker Prediction | Purpose |
| :--- | :--- | :--- | :--- |
| **OLS Linear** | **$7,204.47** | **$27,199.87** | Best unbiased conditional mean |
| **Robust Huber** | **$7,812.20** | **$31,804.09** | Resists rare $60k+ ICU outliers |
| **10th Percentile** | **$2,138.40** | **$17,294.30** | Best-case health floor |
| **50th Percentile (Median)**| **$7,349.91** | **$34,438.94** | Typical median claim |
| **90th Percentile** | **$14,315.09** | **$46,682.31** | Actuarial tail reserve buffer |
| **Flat Mean** | **$8,434.27** | **$32,050.23** | Standard CART constant |

---

## Architectural Extensibility

By decoupling routing from leaf estimation:
- The tree handles **global discrete partitioning** (discovering regime boundaries).
- The leaves handle **local estimation** (`OLS`, `GLM`, `Huber`, `Quantiles`, and future `kNN`, `SVM`, or `Neural` leaves).
- Leaf models can be updated in production or fine-tuned on recent data without altering approved business rules.
