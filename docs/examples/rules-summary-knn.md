# Rules, Leaf Regression Summaries & k-NN Adjustments

Three classical capabilities modernizing Quinlan's C5.0 and Cubist lineage in Rust:
1. **R / statsmodels Leaf Regression Summaries**: Exact goodness-of-fit tables, degrees of freedom, standard errors, $t$-values, $p$-values, $R^2$, and $F$-statistics computed via closed-form covariance matrix $(X^T X)^{-1}$ inversion.
2. **C5.0 Rule-Set Extraction**: Transforming decision paths into standalone IF-THEN rules with interval simplification and Quinlan's pessimistic binomial error pruning ($CF=0.25$).
3. **Cubist $k$-NN Sample Neighbor Adjustments**: Adjusting terminal leaf predictions at inference time using distance-weighted training residuals.

Script: [`examples/rules_summary_knn_showcase.py`](https://github.com/jkapila/mltreelib/blob/main/examples/rules_summary_knn_showcase.py).

---

## 1. R `summary(lm)` & Python `statsmodels` Leaf Summaries

Terminal leaves with multivariate linear models invert the local covariance matrix $(X^T X)^{-1}$ to provide statistical hypothesis testing for each coefficient:

```python
from mltreelib import ModelTree

tree = ModelTree(leaf="linear", leaf_cols="all", max_depth=2, min_rows=30)
tree.fit(X_train, y_train, col_names=["sqft", "beds", "age", "tax_rate"])

# Print statistical summary for Leaf 0
print(tree.leaf_summary(0))
```

Output matches R's `summary(lm)` and Python's `statsmodels.api.OLS`:

```text
======================================================================
Leaf Node 0 (Leaf ID 0) [n=77]
----------------------------------------------------------------------
Residual standard error: 8.5179 on 72 degrees of freedom
Multiple R-squared:  0.9622,  Adjusted R-squared:  0.9601
F-statistic: 458.03 on 4 and 72 DF,  p-value: < 2.2e-16

Coefficients:
                       Estimate   Std. Error    t value     Pr(>|t|) 
(Intercept)             65.4569       5.6195      11.65      < 2e-16 ***
sqft                     0.1126       0.0032      35.66      < 2e-16 ***
beds                    15.1398       0.6895      21.96      < 2e-16 ***
age                     -0.6264       0.0687      -9.11      < 2e-16 ***
tax_rate                -2.8345       1.2814      -2.21       0.0270 *  
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
======================================================================
```

To view all leaves in the tree at once:
```python
print(tree.summary())
```

---

## 2. C5.0 Rule-Set Extraction with Pessimistic Pruning

A decision tree forces disjoint, hierarchical branching. C5.0 extracts root-to-leaf paths into standalone IF-THEN rules, simplifies multi-condition bounds on the same feature, and greedily prunes preconditions using Quinlan's pessimistic error upper bound ($U_{CF}(E, N)$):

```python
# Extract and prune rules with confidence factor cf=0.25
ruleset = tree.extract_rules(X_train, y_train, prune=True, cf=0.25)
print(ruleset)

# Predict directly using the extracted rule set
predictions = ruleset.predict(X_test)
```

Output:

```text
RuleSet (8 rules, default prediction = 0.0000, default confidence = 0.0%):

Rule 1 (cover: 35, errors: 2, confidence: 91.0%):
  IF credit_score <= 575.3220
     AND income_k > 47.7300
     AND income_k <= 77.9470
  THEN prediction = 28.7111

Rule 2 (cover: 39, errors: 3, confidence: 88.9%):
  IF credit_score > 575.3220
     AND credit_score <= 683.6240
     AND debt_ratio <= 0.5970
  THEN prediction = 62.3743

Rule 3 (cover: 35, errors: 4, confidence: 84.4%):
  IF debt_ratio > 0.5970
     AND income_k > 97.8140
  THEN prediction = 32.0894

Default Rule:
  THEN prediction = 0.0000 (confidence: 0.0%)
```

---

## 3. Cubist $k$-NN Sample Neighbor Adjustments

Cubist's signature technique corrects model tree predictions by finding the $k$ nearest training samples in normalized feature space and adjusting the prediction by their weighted residuals:

$$\hat{y}_{\text{adj}}(x) = \hat{y}(x) + \sum_{i=1}^k w_i (y_i - \hat{y}(x_i)), \quad w_i = \frac{1}{D(x, x_i) + \epsilon}$$

where distance is scaled Manhattan distance:
$$D(x, z) = \frac{1}{p} \sum_{j=1}^p \frac{|x_j - z_j|}{\sigma_j}$$

```python
# Standard ModelTree prediction
base_preds = tree.predict(X_test)

# Cubist k-NN adjusted prediction with 5 neighbors
adj_preds_5 = tree.predict(X_test, neighbors=5)
```

### Benchmark Performance

On datasets with non-linear or localized effects:

| Inference Mode | Test MSE | Test MAE | Improvement |
| :--- | :--- | :--- | :--- |
| **Standard ModelTree** | 1790.45 | 33.24 | Baseline |
| **Cubist Adjustment ($k=3$)** | 1160.44 | 24.31 | **-35.2%** error |
| **Cubist Adjustment ($k=5$)** | 1172.85 | 24.66 | **-34.5%** error |
| **Cubist Adjustment ($k=9$)** | 1083.16 | 24.95 | **-39.5%** error |
