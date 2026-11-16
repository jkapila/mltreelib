# Why We Rebuilt Quinlan's Cubist and C5.0 in Rust

**Author:** Jitin Kapila & mltreelib Team  
**Category:** Core Architecture & Systems ML  

---

## The Forgotten Masterpieces of Machine Learning

Long before gradient boosting and tabular foundation models dominated benchmarks, J. Ross Quinlan introduced a series of machine learning innovations that remain unmatched in rule interpretability and compact leaf modeling:
- **C4.5 / C5.0**: Multi-way discrete branching, rule-set extraction with pruning, and information gain ratio.
- **Cubist**: M5-style model trees combining logical rules with multivariate linear models at each terminal leaf, backed by composite committee boosting and sample neighbor adjustments.
- **GritBot**: Contextual anomaly auditing that uses tree partitions to detect local condition-specific outliers.

In R, packages like `Cubist` and `C50` developed cult followings among chemometricians and domain scientists. However, in Python, these algorithms remained trapped in 1990s C extensions—plagued by segmentation faults, messy memory leaks, and lack of modern DataFrame/Arrow integration.

---

## Enter Rust: Safety, SIMD, and Zero-Copy Frames

`mltreelib` rewrites Quinlan's algorithms from scratch in modern, memory-safe Rust (`crates/mltreelib-core`):

1. **Zero-Copy Arrow Buffering**: Ingests PyArrow and Polars tables directly into contiguous column memory buffers.
2. **Streaming Split Scans**: Implements $O(N)$ numerical linear scans and SIMD aggregations without intermediate heap allocations.
3. **Pure Python Frontends**: Exposed via thin PyO3 bindings with a clean, modern API (`mltreelib._core`).
4. **First-Class Missingness**: Implements Quinlan's original fractional missing value weight distributions without fragile post-hoc imputers.

The result is a production-grade implementation ready for modern tabular machine learning pipelines.

---

## 1. R `summary(lm)` & Python `statsmodels` Leaf Regression Summaries

In classical linear model trees, terminal leaves fit multivariate linear equations. However, standard tree packages throw away the covariance structure of the leaf regressions, leaving practitioners unable to evaluate coefficient significance.

In `mltreelib`, every terminal leaf computes the closed-form inverse $(X^T X)^{-1}$ via `faer` LU decomposition. Calling `tree.leaf_summary(id)` produces exact R `summary(lm)` reports:

```python
from mltreelib import ModelTree

tree = ModelTree(leaf="linear", leaf_cols="all", max_depth=2, min_rows=30)
tree.fit(X_train, y_train, col_names=["sqft", "beds", "age", "tax_rate"])

# Inspect Leaf 0 regression statistics
print(tree.leaf_summary(0))
```

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

---

## 2. C5.0 Rule-Set Extraction with Pessimistic Binomial Pruning

Trees enforce strict hierarchies where conditions near the root apply globally. C5.0 flattens decision paths into modular IF-THEN rules, simplifies multi-condition bounds on the same variable, and prunes preconditions using Quinlan's pessimistic binomial error bound $U_{CF}(E, N)$:

```python
# Extract C5.0 rules with confidence factor 0.25
ruleset = tree.extract_rules(X_train, y_train, prune=True, cf=0.25)
print(ruleset)

# Predict directly using the rule set
rule_predictions = ruleset.predict(X_test)
```

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

Default Rule:
  THEN prediction = 0.0000 (confidence: 0.0%)
```

---

## 3. Cubist $k$-NN Sample Neighbor Adjustments

Cubist's crowning innovation is its post-prediction neighbor correction: adjust leaf predictions at query time using the residuals of $k$ training instances nearest in normalized Manhattan space:

$$\hat{y}_{\text{adj}}(x) = \hat{y}(x) + \sum_{i=1}^k w_i (y_i - \hat{y}(x_i)), \quad w_i = \frac{1}{D(x, x_i) + \epsilon}$$

```python
# Unadjusted tree prediction
base_preds = tree.predict(X_test)

# Cubist 5-neighbor adjusted prediction
adjusted_preds = tree.predict(X_test, neighbors=5)
```

### Empirical Results

On non-linear response surfaces with localized effects, $k$-NN neighbor adjustments reduce error significantly:

```text
Inference Mode                     Test MSE     Test MAE
--------------------------------------------------------
Standard ModelTree                  1790.45        33.24
Cubist Adjustment (k=3)             1160.44        24.31
Cubist Adjustment (k=5)             1172.85        24.66
Cubist Adjustment (k=9)             1083.16        24.95
--------------------------------------------------------
Error reduction: -34.5% test MSE!
```
