---
title: "Metrics, Diagnostics & Probability Calibration"
description: "Comprehensive model evaluation guide: ROC, Precision-Recall, Calibration curves, Population Stability Index (PSI), Double-Lift charts, and Residual analysis."
keywords:
  - machine learning evaluation metrics
  - probability calibration
  - ROC PR curves
  - PSI drift detection
  - double lift chart
---

# Metrics, Diagnostics & Probability Calibration

A model score without diagnostic verification is an operational liability. A high AUC can disguise severe probability miscalibration, and an acceptable test RMSE can hide heteroscedastic residual explosions or catastrophic drift.

`mltreelib.Metrics` and `mltreelib.plots` provide an evaluation and visualization engine implemented with pure numeric precision, independent of heavy visualization dependencies (matplotlib is an optional extra: `pip install mltreelib[plotting]`).

```mermaid
flowchart LR
    A["Fitted Model Predictions\nScores y_hat vs Truth y"] --> B{"Task Type"}

    B -- "Classification / Risk" --> C["Discrimination\nROC AUC, PR AUC, KS Statistic"]
    B -- "Regression / Amounts" --> D["Loss Surface & Residuals\nRMSE, MAE, MedAE, Pinball Loss"]

    C --> E["Reliability Calibration\nDecile Brier Score & Hosmer-Lemeshow"]
    E --> F["Operational Triage\nCumulative Gain & Lift @ Budget"]
    F --> G["Stability & Drift\nPopulation Stability Index (PSI)"]

    D --> H["Heteroscedasticity\nResidual Variance vs Fitted"]

    classDef blueBox fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef greenBox fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#ffffff;
    classDef grayBox fill:#334155,stroke:#64748b,stroke-width:2px,color:#ffffff;

    class A,B grayBox;
    class C,D blueBox;
    class E,F,G,H greenBox;
```

---

## 1. Multi-Panel Diagnostic Surface

Every chart in `mltreelib.plots` operates on native Matplotlib `Axes`, allowing multi-metric dashboards in a single figure:

```bash
uv run python examples/metrics_showcase.py
```

![Evaluation Metrics Multi-Panel Diagnostic Surface](../assets/metrics_panels.png)

---

## 2. Classification Diagnostics & Probability Calibration

In credit triage, fraud detection, and medical diagnosis, ranking alone is insufficient. Underwriters make decisions based on expected dollar loss, which demands calibrated probabilities $P(y=1 \mid \mathbf{x})$.

| Diagnostic Chart | Core Question Answered | Failure Mode Diagnosed |
| :--- | :--- | :--- |
| **`plot_roc`** / **`plot_pr`** | Can the model rank positive instances ahead of negatives? | High ROC with low PR indicates severe false-positive flooding under class imbalance. |
| **`plot_calibration`** | Does a predicted score of 0.20 correspond to exactly 20% empirical event rate? | S-shaped curve indicates over-confidence; inverted curve indicates mis-specified link function. |
| **`plot_ks`** | What is the maximum separation distance between default and non-default distributions? | Identifies the optimal cutoff threshold $\tau^*$ maximizing operational divergence. |
| **`plot_double_lift`** | When comparing Model A vs Model B, which model captures more true events across deciles? | Reveals regions where a challenger model outperforms a production champion. |
| **`plot_psi`** | Did the score distribution shift between training (baseline) and inference (current)? | PSI $> 0.25$ signals severe population drift requiring immediate model retraining. |

### Visual Examples

| Reliability Calibration Curve | Double-Lift Model Comparison |
| :---: | :---: |
| ![Probability Calibration Curve](../assets/plots/calibration.png) | ![Double Lift Chart Comparison](../assets/plots/double_lift.png) |

---

## 3. Regression & Residual Diagnostics

For continuous pricing, demand forecasting, and risk models:

| Diagnostic Chart | Core Question Answered | Failure Mode Diagnosed |
| :--- | :--- | :--- |
| **`plot_residuals`** | Are residual errors mean-zero and homoscedastic across all response levels? | Funnel-shaped residual spread signals heteroscedasticity, calling for `gain="heteroscedastic"` or Tweedie. |
| **`plot_importance`** | Which features drove the split deviance reduction across the tree? | Over-reliance on single surrogate features or leakage columns. |
| **`plot_breakdown`** | Why did this individual loan applicant receive this exact predicted rate? | DALE-style waterfall attribution decomposed by tree split paths. |

![Regression Residual Error Analysis](../assets/plots/residuals.png)

---

## 4. Code Example: Computing the Full Metrics Suite

```python
from mltreelib import Metrics, plots
import matplotlib.pyplot as plt

metrics = Metrics()

# 1. Classification Evaluation
clf_report = metrics.classification(y_true, y_prob, threshold=0.15)
print(f"ROC-AUC: {clf_report['roc_auc']:.4f} | PR-AUC: {clf_report['pr_auc']:.4f}")
print(f"Brier Score: {clf_report['brier_score']:.4f} | KS Statistic: {clf_report['ks_stat']:.4f}")

# 2. Population Stability Index (PSI) Drift Check
psi_res = metrics.psi(baseline_scores, current_scores, bins=10)
print(f"Population Stability Index: {psi_res['psi']:.4f} (Status: {psi_res['status']})")

# 3. Render Diagnostics Side-by-Side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plots.plot_calibration(y_true, y_prob, ax=ax1)
plots.plot_double_lift(y_true, pred_challenger, pred_champion, ax=ax2)
plt.tight_layout()
plt.show()
```

!!! tip "Related Guides"
    - [Why Credit Scoring Needs Glass-Box Trees](../blog/why-credit-scoring-needs-glass-box-trees.md) for asymmetric cost optimization using KS cutoffs.
    - [Model Inspection & Quinlan Rules](model-inspection-and-rules.md) for white-box split path audits.
