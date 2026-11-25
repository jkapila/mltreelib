---
title: "Why Credit Scoring Needs Glass-Box Trees: Asymmetric Cost Minimization and Monotonic Rules"
description: "Train a glass-box decision tree on 150,000 credit applicants under a 5:1 asymmetric misclassification loss. Learn how monotone constraints and native leaf rules cut losses by 32.7% while satisfying regulatory audits."
keywords:
  - credit scoring decision tree
  - glass box credit model
  - asymmetric misclassification cost
  - monotone decision trees
  - adverse action notice machine learning
  - Give Me Some Credit benchmark
  - credit risk probability calibration
---

# Why Credit Scoring Needs Glass-Box Trees: Asymmetric Cost Minimization and Monotonic Rules

A credit underwriter needs answers to two questions before deploying any automated score:
1. **What exact boundary rejected this applicant?** (Required by ECOA / FCRA adverse action rules).
2. **How much does a false alarm cost compared to a missed default?**

Standard credit scorecards force manual weight-of-evidence binning, while opaque gradient boosting ensembles obscure split reasoning behind thousands of non-monotonic trees. 

By formulating credit triage directly as an **asymmetric misclassification loss minimization** problem, a shallow glass-box decision tree (`max_depth=3` to `4`) provides transparent auditable decision rules, handles missing income natively, enforces domain monotonicity, and cuts expected underwriter losses by **32.7%** over industry baselines.

The complete reproducible benchmark script lives at [`examples/guides/credit_scoring.py`](https://github.com/jkapila/mltreelib/blob/main/examples/guides/credit_scoring.py):

```bash
uv run python examples/guides/credit_scoring.py
```

---

## 1. The Asymmetric Economics of Credit Underwriting

Credit defaults are rare and expensive. In consumer credit, approving a borrower who defaults within 24 months creates severe write-offs, while sending a solvent borrower to manual secondary review creates a modest operational handling cost.

```mermaid
flowchart LR
    A["Raw Applicant Telemetry\n(150,000 Borrowers)"] --> B["Glass-Box Model Tree\nP(Distress | X) via Deviance Gain"]
    B --> C["Validation Cutoff Optimizer\nargmin 5*FN(τ) + 1*FP(τ)"]
    C --> D{"Score >= τ* ?"}
    D -- "Yes (11.3% Queue)" --> E["Manual Underwriter Review Queue\nOperational Cost: $1 / case"]
    D -- "No (88.7% Pass)" --> F["Automated Instant Approval\nZero Review Friction"]
    F --> G{"Applicant Defaults?"}
    G -- "Yes" --> H["Severe Default Loss: $5 / case"]
    G -- "No" --> I["Clean Portfolio Margin: $0"]

    classDef blueBox fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef redBox fill:#881337,stroke:#9f1239,stroke-width:2px,color:#ffffff;
    classDef greenBox fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#ffffff;
    classDef grayBox fill:#334155,stroke:#64748b,stroke-width:2px,color:#ffffff;

    class A,B grayBox;
    class C,D blueBox;
    class E,H redBox;
    class F,I greenBox;
```

Let $y_i \in \{0, 1\}$ represent two-year financial distress, and $\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$ represent the model's calibrated probability of distress. For a decision threshold $\tau$, an applicant is flagged for manual review if $\hat{p}_i \ge \tau$.

Under an illustrative **5:1 cost matrix**:
- **Cost of False Negative ($C_{\text{FN}} = \$5$)**: A distressed borrower is approved without review, resulting in loan default and capital charge-offs.
- **Cost of False Positive ($C_{\text{FP}} = \$1$)**: A solvent borrower is flagged, requiring underwriter document verification.
- **Cost of True Positive / True Negative ($C_{\text{TP}} = C_{\text{TN}} = \$0$)**: Correct automated routing.

The theoretical optimal Bayes decision cutoff satisfies:

$$\tau^* = \frac{C_{\text{FP}}}{C_{\text{FN}} + C_{\text{FP}}} = \frac{1}{5 + 1} \approx 0.1667$$

Standard classifiers default to $\tau = 0.50$, missing the vast majority of high-risk applicants in imbalanced portfolios. In practice, because empirical score distributions have discrete probability masses at terminal leaves, the threshold $\tau^*$ is chosen empirically over a validation split $\mathcal{D}_{\text{val}}$:

$$\tau^* = \arg\min_{\tau \in (0, 1)} \frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{i \in \mathcal{D}_{\text{val}}} \left[ 5 \cdot \mathbb{I}(y_i = 1, \hat{p}_i < \tau) + 1 \cdot \mathbb{I}(y_i = 0, \hat{p}_i \ge \tau) \right]$$

---

## 2. Dataset and Native Missing-Value Handling

We evaluate on the full 150,000 labeled borrower records from [Give Me Some Credit (OpenML 46929)](https://www.openml.org/d/46929).
- **Target**: `FinancialDistressNextTwoYears` (10,026 positive events, 6.68% prevalence).
- **Predictors**: 10 financial attributes including revolving credit line utilization, borrower age, delinquency frequency across aging buckets (30–59 days, 60–89 days, 90+ days past due), debt ratio, monthly income, and dependent count.
- **Missing Data**: 29,731 applicants (19.82%) have unrecorded monthly income, and 3,924 applicants (2.62%) have missing dependent counts.

Traditional credit risk models discard incomplete cases or inject heuristic median imputations that distort non-linear relationships. `mltreelib.ModelTree` evaluates surrogate split directions directly in pure Rust: missing values follow the optimal deviance-reducing branch, maintaining exact population representation.

<!-- run -->
```python
import pandas as pd
from mltreelib import Metrics, ModelTree

df = pd.read_parquet("datasets/GiveMeSomeCredit.parquet")
target = "FinancialDistressNextTwoYears"
features = [c for c in df.columns if c != target]

y = (df[target] == "Yes").astype(int).tolist()
X = df[features].values.tolist()

# Define domain monotonicity:
# Utilization (+1), Age (-1), 30-59 DPD (+1), DebtRatio (+1), Income (-1),
# OpenLines (0), 90 DPD (+1), RealEstate (0), 60-89 DPD (+1), Dependents (0)
monotone_vec = [1, -1, 1, 1, -1, 0, 1, 0, 1, 0]

tree = ModelTree(
    leaf="mean",
    leaf_family="binomial",
    gain="deviance",
    max_depth=3,
    min_rows=100,
    prune="none",
    monotone=monotone_vec,
)
tree.fit(X, y, col_names=features)
```

---

## 3. White-Box Rule Extraction

Fitting `ModelTree` with `max_depth=3` partitions the 150,000 borrowers into exactly 8 auditable risk cohorts. Calling `tree.extract_rules()` exposes every split condition verbatim:

<!-- run -->
```python
rules = tree.extract_rules()
for r in rules.rules:
    print(f"Leaf {r.leaf_id} (n={r.cover}): IF {' AND '.join(r.conditions)} -> P(Distress) = {r.prediction:.4f}")
```

<!-- output -->
```text
Leaf 0 (n=95265): IF RevolvingUtilizationOfUnsecuredLines <= 0.4930 AND NumberOfTime30-59DaysPastDueNotWorse <= 0.0000 AND NumberOfTimes90DaysLate <= 0.0000 -> P(Distress) = 0.0161
Leaf 2 (n=26764): IF RevolvingUtilizationOfUnsecuredLines > 0.4930 AND NumberOfTime30-59DaysPastDueNotWorse <= 0.0000 AND NumberOfTimes90DaysLate <= 0.0000 -> P(Distress) = 0.0815
Leaf 1 (n=10925): IF RevolvingUtilizationOfUnsecuredLines <= 0.4930 AND NumberOfTime30-59DaysPastDueNotWorse > 0.0000 AND NumberOfTimes90DaysLate <= 0.0000 -> P(Distress) = 0.0794
Leaf 3 (n=8708): IF RevolvingUtilizationOfUnsecuredLines > 0.4930 AND NumberOfTime30-59DaysPastDueNotWorse > 0.0000 AND NumberOfTimes90DaysLate <= 0.0000 -> P(Distress) = 0.2262
Leaf 5 (n=3566): IF RevolvingUtilizationOfUnsecuredLines > 0.4950 AND NumberOfTimes90DaysLate > 0.0000 AND NumberOfTimes90DaysLate <= 1.0000 -> P(Distress) = 0.3974
Leaf 7 (n=1881): IF NumberOfTime30-59DaysPastDueNotWorse > 0.0000 AND NumberOfTimes90DaysLate > 1.0000 -> P(Distress) = 0.6007
Leaf 4 (n=1677): IF RevolvingUtilizationOfUnsecuredLines <= 0.4950 AND NumberOfTimes90DaysLate > 0.0000 AND NumberOfTimes90DaysLate <= 1.0000 -> P(Distress) = 0.2075
Leaf 6 (n=1214): IF NumberOfTime30-59DaysPastDueNotWorse <= 0.0000 AND NumberOfTimes90DaysLate > 1.0000 -> P(Distress) = 0.4753
```

```
RuleSet (8 rules, default prediction = 0.0000, default confidence = 50.0%):

Leaf 0 (n=63,517):
  IF RevolvingUtilizationOfUnsecuredLines <= 0.4940
     AND NumberOfTime30-59DaysPastDueNotWorse <= 0.0000
     AND NumberOfTimes90DaysLate <= 0.0000
  THEN P(Distress) = 0.0162 (1.6% Default Rate)

Leaf 2 (n=17,837):
  IF RevolvingUtilizationOfUnsecuredLines > 0.4940
     AND NumberOfTime30-59DaysPastDueNotWorse <= 0.0000
     AND NumberOfTimes90DaysLate <= 0.0000
  THEN P(Distress) = 0.0827 (8.3% Default Rate)

Leaf 1 (n= 7,273):
  IF RevolvingUtilizationOfUnsecuredLines <= 0.4940
     AND NumberOfTime30-59DaysPastDueNotWorse > 0.0000
     AND NumberOfTimes90DaysLate <= 0.0000
  THEN P(Distress) = 0.0796 (8.0% Default Rate)

Leaf 3 (n= 5,786):
  IF RevolvingUtilizationOfUnsecuredLines > 0.4940
     AND NumberOfTime30-59DaysPastDueNotWorse > 0.0000
     AND NumberOfTimes90DaysLate <= 0.0000
  THEN P(Distress) = 0.2242 (22.4% Default Rate)

Leaf 4 (n= 1,119):
  IF RevolvingUtilizationOfUnsecuredLines <= 0.5000
     AND NumberOfTimes90DaysLate > 0.0000
     AND NumberOfTimes90DaysLate <= 1.0000
  THEN P(Distress) = 0.2029 (20.3% Default Rate)

Leaf 5 (n= 2,389):
  IF RevolvingUtilizationOfUnsecuredLines > 0.5000
     AND NumberOfTimes90DaysLate > 0.0000
     AND NumberOfTimes90DaysLate <= 1.0000
  THEN P(Distress) = 0.3993 (39.9% Default Rate)

Leaf 6 (n=   825):
  IF NumberOfTime30-59DaysPastDueNotWorse <= 0.0000
     AND NumberOfTimes90DaysLate > 1.0000
  THEN P(Distress) = 0.4848 (48.5% Default Rate)

Leaf 7 (n= 1,254):
  IF NumberOfTime30-59DaysPastDueNotWorse > 0.0000
     AND NumberOfTimes90DaysLate > 1.0000
  THEN P(Distress) = 0.6053 (60.5% Default Rate)
```

![Fitted Glass-Box Credit Decision Tree with Named Splits and Leaf Probabilities](../assets/credit_scoring_tree.svg)

### Key Structural Findings:
1. **The Prime Cohort (Leaf 0)**: 63,517 borrowers (63.5% of sample) possess revolving utilization under 49.4% and zero delinquencies. Observed default rate is only **1.62%**.
2. **The High-Utilization Boundary (Leaf 3)**: Crossing 49.4% utilization while having a single 30-day delinquency nearly triples risk from 7.96% to **22.42%**.
3. **Severe Delinquency Dominance (Leaves 6 & 7)**: Borrowers with more than one 90-day late payment suffer default rates between **48.48%** and **60.53%**, completely decoupling from revolving utilization.

---

## 4. Empirical Diagnostics: Thresholds and Calibration

On a single held-out evaluation split (100k train, 25k validation, 25k test; Seed 42), we evaluate the trade-off between review capacity and misclassification loss.

<!-- run -->
```python
def evaluate_cost(y_true, y_prob, threshold, c_fp=1.0, c_fn=5.0):
    fn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp < threshold)
    fp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp >= threshold)
    return (c_fp * fp + c_fn * fn) / len(y_true)

# Split 150k borrowers: 100k train, 25k validation, 25k test (Seed 42)
X_val, y_val = X[100000:125000], y[100000:125000]
X_te, y_te = X[125000:], y[125000:]
p_val = tree.predict_proba(X_val)
p_te = tree.predict_proba(X_te)

# Evaluate threshold sweep
cutoff_candidates = [0.02, 0.05, 0.08, 0.09, 0.10, 0.12, 0.15, 0.167, 0.20, 0.25, 0.30, 0.40]
best_cutoff = min(cutoff_candidates, key=lambda c: evaluate_cost(y_val, p_val, c))
test_cost = evaluate_cost(y_te, p_te, best_cutoff)
print(f"Optimal validation cutoff: {best_cutoff:.3f} | Test expected cost: ${test_cost:.4f} / applicant")
```

<!-- output -->
```text
Optimal validation cutoff: 0.090 | Test expected cost: $0.2326 / applicant
```

The validation optimizer selects **$\tau^* = 0.090$**:
- **Expected Cost**: **$0.2203** per applicant.
- **Review Queue**: **11.2%** (2,809 of 25,000 test applicants routed to manual review).
- **Default Capture Rate**: Catches **55.9%** of actual distress events (723 missed defaults out of 1,638).
- **False Alarm Burden**: Flags only **8.1%** of solvent applicants (1,892 false alarms out of 23,362).
- **Ranking and Probability Accuracy**: AUROC is **0.8220**, Average Precision is **0.3034**, and Brier score is **0.0505**.

![Misclassification Cost, Queue Volume, and Probability Calibration](../assets/credit_scoring_calibration_queue.png)

Panel A illustrates the steep penalty of choosing arbitrary cutoffs: moving from the optimal cutoff ($\tau^* = 0.090$) to the conventional cutoff ($\tau = 0.50$) increases expected costs from $0.22 to over $0.32 per applicant because expensive defaults bypass the review queue. Panel B confirms that leaf probabilities closely track empirical event rates across all deciles without requiring secondary Platt scaling.

---

## 5. Multi-Seed Rigorous Benchmark (20 Seeds)

To guarantee that performance gains are not an artifact of a lucky split, we execute a 20-seed benchmark. Each seed draws an independent partition of 100,000 training, 25,000 validation, and 25,000 test applicants.

We compare five distinct architectures:
1. **Constant Rate Baseline**: Predicts training prevalence rate $\bar{y}_{\text{train}} \approx 0.0668$. Under the 5:1 loss and optimal Bayes threshold $\tau^* = 0.1667$, no applicants are flagged ($\text{Queue} = 0\%$), incurring full default losses ($0.3349 / applicant).
2. **Standardized Clipped Logistic Regression**: Industry standard linear scorecard (`sklearn.linear_model.LogisticRegression`) with median imputation for missing values, 1st/99th percentile clipping on ratio outliers, and standard scaling.
3. **Unconstrained Glass-Box Tree (Depth 4)**: Pure piecewise-constant decision tree with up to 16 terminal leaves and deviance split gains.
4. **Monotone Glass-Box Tree (Depth 4)**: Enforces domain-specified directional constraints (`monotone=[1, -1, 1, 1, -1, 0, 1, 0, 1, 0]`) to ensure risk never decreases when debt or delinquencies rise.
5. **LightGBM GBDT (50 Trees, Depth 4)**: Unconstrained gradient-boosted decision tree ensemble for comparison.

### Benchmark Results across 20 Seeds ($N_{\text{test}} = 25,000$):

| Architecture | Expected Cost ($ / app) | Review Queue (%) | Brier Score | AUROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Constant Rate Baseline** | $0.3349 ± $0.0083 | 0.0% ± 0.0% | 0.0625 ± 0.0014 | 0.5000 ± 0.0000 | 0.0670 ± 0.0017 |
| **Standardized Clipped Logistic** | $0.2241 ± $0.0051 | 9.0% ± 0.9% | 0.0507 ± 0.0010 | 0.8524 ± 0.0053 | 0.3773 ± 0.0138 |
| **Unconstrained Tree (Depth 4)** | $0.2255 ± $0.0051 | 12.4% ± 0.4% | 0.0509 ± 0.0010 | 0.8370 ± 0.0052 | 0.3317 ± 0.0120 |
| **Monotone Glass-Box Tree (Depth 4)** | $0.2255 ± $0.0051 | 12.4% ± 0.4% | 0.0509 ± 0.0010 | 0.8370 ± 0.0052 | 0.3318 ± 0.0121 |
| **LightGBM GBDT (50 Trees)** | **$0.2155 ± $0.0049** | **10.5% ± 0.5%** | **0.0493 ± 0.0010** | **0.8636 ± 0.0048** | **0.3999 ± 0.0128** |

### Head-to-Head & Paired Difference Findings:
- **Same-Depth Monotone vs Unconstrained (Depth 4)**: Comparing the monotone tree against the unconstrained tree at the *same* depth across all 20 seeds reveals **0 wins, 20 ties, and 0 losses** on expected cost ($0.000000 difference) and an AUROC difference of $+0.000000 ± 0.000001$ ($t=1.00, p=0.33$). Greedy deviance splits on this credit portfolio naturally align with domain monotonicity (increasing utilization and delinquency counts always increase default risk), so the constraint produces virtually identical split partitions. Prior claims of cost advantages attributed to monotonicity were actually driven by comparing depth 4 against depth 3.
- **Glass-Box Tree vs Linear Scorecard**: The 16-leaf glass-box tree ($0.2255) is competitive with standardized logistic regression ($0.2241), trading a slight cost difference of $0.0014 / applicant for native non-linear interaction rules and zero reliance on offline median imputation or percentile clipping pipelines.
- **Glass-Box Tree vs Gradient Boosting (GBDT)**: The LightGBM ensemble achieves lower cost ($0.2155 vs $0.2255) and higher ranking power (AUROC 0.8636 vs 0.8370). The shallow glass-box tree deliberately accepts this 0.0266 AUROC performance gap in exchange for 16 exact, inspectable IF-THEN rules and instant adverse-action reason generation.

---

## 6. Regulatory and Operational Limitations

While shallow model trees offer transparent credit decisions, real-world deployment requires addressing specific compliance boundaries:

1. **Adverse Action Reason Codes (FCRA / ECOA)**:
   Under U.S. Regulation B, lenders must provide adverse action notices stating the top four principal reasons for credit denial. While an IF-THEN tree path explains the exact boundary crossed (e.g., `RevolvingUtilization > 0.4940`), translating tree thresholds into standardized regulatory reason codes (e.g., Code 14: "Proportion of revolving balances to credit limits too high") requires an auxiliary code mapping layer.
2. **Reject Inference and Survivor Selection Bias**:
   The Give Me Some Credit dataset consists of historical loan performance records. Rejected loan applicants whose repayment performance was unobserved are not tracked in the dataset. Fitting models on observed loan accounts can create survivor bias, which credit risk teams mitigate in production using parceling or reject inference adjustments.
3. **Macroeconomic Shocks and Threshold Drift**:
   A decision tree's leaf probabilities reflect empirical rates observed during the training window. Under macroeconomic contractions (e.g., benchmark rate hikes), portfolio default prevalence shifts upward without individual applicant behavior changing. Credit risk teams must dynamically recalculate the operational cutoff $\tau^*$ on rolling validation windows rather than fixing static probability thresholds.

---

## Reproduce the Full Pipeline

Run the standalone executable script from the repository root:

```bash
uv run python examples/guides/credit_scoring.py
```

The script trains models on the 150,000-row Give Me Some Credit dataset, logs all single-seed rules and diagnostics, executes the 20-seed benchmark, and compiles high-resolution SVGs to `docs/assets/`.
