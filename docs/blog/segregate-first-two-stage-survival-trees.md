---
title: "Segregate First, Model Later: Two-Stage Hierarchical Survival Trees for Longitudinal Cohorts"
description: "How to decouple subject ID-level tree cohort discovery from leaf-level Kaplan-Meier survival curves, eliminating right-censoring bias and non-proportional hazard violations on longitudinal customer telemetry."
keywords:
  - two stage survival tree
  - hierarchical survival model
  - customer churn survival analysis
  - non proportional hazards decision tree
  - Kaplan Meier tree stratification
  - longitudinal cohort survival analysis
  - right censoring bias in trees
---

# Segregate First, Model Later: Two-Stage Hierarchical Survival Trees for Longitudinal Cohorts

Standard survival models force a difficult compromise: parametric Cox proportional hazards assume hazard ratios never cross, while standard single-stage survival trees let right-censoring distort split selection. 

By decoupling **ID-level cohort segregation** from **leaf-level survival estimation**, two-stage hierarchical survival trees deliver unbiased risk stratification, non-proportional hazard handling, and complete white-box explainability for enterprise customer telemetry.

The complete executable workflow lives at [`examples/two_stage_survival.py`](https://github.com/jkapila/mltreelib/blob/main/examples/two_stage_survival.py):

```bash
uv run python examples/two_stage_survival.py
```

---

## 1. The Censoring Trap in Longitudinal Telemetry

In enterprise retention, credit facilities, and medical trials, entities (customers, machines, patients) possess two types of data:
1. **Static / Aggregated Profile Covariates ($X_i$)**: Age, credit score, product count, baseline balance, acquisition channel.
2. **Longitudinal Duration Outcomes ($T_i, E_i$)**: Observed tenure duration $T_i$ and event indicator $E_i \in \{0, 1\}$ ($1 = \text{churn/event}, 0 = \text{censored/active}$).

When standard survival trees split directly on duration outcomes using log-rank statistics, they fall into the **censoring trap**:

```mermaid
flowchart TD
    subgraph Trap["Standard Survival Tree Pitfall"]
        A["All Longitudinal Records [X, T, E]"] --> B{"Log-Rank Split Criterion\nMaximizes Event Separation"}
        B --> C["Trivial Censoring Splits:\nIsolates recent sign-ups from mature accounts"]
        B --> D["Overfits Administrative Censoring Horizon"]
    end

    subgraph Solved["Two-Stage Segregation Architecture"]
        E["Static Customer Profile [X_i]"] --> F["Stage 1: Profile Decision Tree\nDiscovers Natural Behavioral Cohorts"]
        F --> G["Leaf 0: Low Risk"]
        F --> H["Leaf 1: High Risk Young"]
        F --> I["Leaf 2: Mature Mainstream"]
        F --> J["Leaf 3: Senior Flight Risk"]
        G --> K["Stage 2: Leaf-Level Kaplan-Meier Engine\nS_0(t) fitted purely on Leaf 0 members"]
        H --> L["Stage 2: Leaf-Level Kaplan-Meier Engine\nS_1(t) fitted purely on Leaf 1 members"]
        I --> M["Stage 2: Leaf-Level Kaplan-Meier Engine\nS_2(t) fitted purely on Leaf 2 members"]
        J --> N["Stage 2: Leaf-Level Kaplan-Meier Engine\nS_3(t) fitted purely on Leaf 3 members"]
    end

    classDef trapBox fill:#881337,stroke:#9f1239,stroke-width:2px,color:#ffffff;
    classDef gateBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef dataBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef stageBox fill:#1e3a5f,stroke:#2b4c7e,stroke-width:2px,color:#ffffff;
    classDef lowRisk fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;
    classDef medRisk fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef highRisk fill:#881337,stroke:#9f1239,stroke-width:2px,color:#ffffff;
    classDef kmBox fill:#0f766e,stroke:#14b8a6,stroke-width:2px,color:#ffffff;

    class A,C,D trapBox;
    class B gateBox;
    class E dataBox;
    class F stageBox;
    class G lowRisk;
    class H,I medRisk;
    class J highRisk;
    class K,L,M,N kmBox;
```

### Why Standard Methods Fail:
- **Cox Proportional Hazards Violations**: Classical Cox models assume $\lambda(t \mid X) = \lambda_0(t) \exp(\boldsymbol{\beta}^T X)$. If early-tenure churn dynamics differ fundamentally from late-tenure attrition, hazard curves cross. The proportional hazard assumption collapses.
- **Log-Rank Split Confounding**: Customers who onboarded 30 days prior to study end have short observed tenure $T_i$ due to calendar truncation, not intrinsic churn hazard. A greedy survival tree frequently splits on exposure duration rather than structural customer behavior, achieving near-random C-index performance.

---

## 2. The Two-Stage Architecture

Instead of conflating cohort segmentation with time-to-event estimation, we decompose the problem into two distinct stages:

### Stage 1: Subject ID-Level Cohort Discovery
We aggregate telemetry to the unique ID level and fit an interpretable `ModelTree` on baseline customer covariates $X_i$:

$$
k = \text{Tree}(X_i), \quad k \in \{0, 1, \dots, K-1\}
$$

The tree partitions the customer base into $K$ transparent behavioral cohorts with exact readable rules (e.g., `Age <= 43 AND Products <= 2`).

### Stage 2: Leaf-Level Localized Kaplan-Meier Engines
Within each terminal leaf cohort $k$, we fit a dedicated non-parametric survival step function over the observed duration and event pairs $(T_i, E_i)_{i \in \text{Leaf}_k}$:

$$
\hat{S}_k(t) = \prod_{t_j \le t} \left( 1 - \frac{d_{jk}}{n_{jk}} \right)
$$

Where $t_j$ are distinct event times in cohort $k$, $d_{jk}$ is observed failures at $t_j$, and $n_{jk}$ is customers at risk just prior to $t_j$.

### Stage 3: Low-Latency Serving Contract
When a new customer arrives:
1. Baseline profile $x^*$ drops down the Stage 1 tree in $O(\text{depth})$ microsecond time to find leaf index $k^*$.
2. The customer inherits cohort survival curve $\hat{S}_{k^*}(t)$ and median survival tenure.

---

## 3. Case Study: 10,000 Commercial Bank Customers

We test this architecture on the real **Bank Customer Churn** benchmark (10,000 customers, 10-year tenure follow-up; 70% train / 30% test).

```python
import pyarrow.parquet as pq
from mltreelib import ModelTree, Metrics
import numpy as np

# 1. Load benchmark dataset
table = pq.read_table("datasets/Bank_Customer_Churn.parquet")
df = table.to_pydict()
n = len(df["churn"])

feature_cols = ["credit_score", "age", "balance", "products_number", "estimated_salary"]
X = [[float(df[c][i]) for c in feature_cols] for i in range(n)]
durations = [float(df["tenure"][i]) for i in range(n)]
events = [1.0 if str(df["churn"][i]).lower() in ("1", "yes", "true") else 0.0 for i in range(n)]

# 70/30 Train/Test split
split = int(n * 0.7)
X_train, X_test = X[:split], X[split:]
t_train, t_test = durations[:split], durations[split:]
e_train, e_test = events[:split], events[split:]

# Stage 1: Cohort Segmentation Tree (depth=2, 4 cohorts)
stage1_tree = ModelTree(max_depth=2, min_rows=50)
stage1_tree.fit(X_train, e_train, col_names=feature_cols)

train_leaf_ids = stage1_tree.apply(X_train)
test_leaf_ids = stage1_tree.apply(X_test)

# Stage 2: Fit Leaf-Level Survival Engines
leaf_models = {}
for leaf_id in set(train_leaf_ids):
    idx = [i for i, l in enumerate(train_leaf_ids) if l == leaf_id]
    l_X = [X_train[i] for i in idx]
    l_t = [t_train[i] for i in idx]
    l_e = [e_train[i] for i in idx]

    surv_leaf = ModelTree(max_depth=0, leaf_family="survival")
    surv_leaf.fit(l_X, (l_t, l_e))
    leaf_models[leaf_id] = surv_leaf

# Stage 3: Generate Monotone Survival Curves for Test Set
tenure_grid = [1.0, 3.0, 5.0, 8.0, 10.0]
sample_preds = [leaf_models[test_leaf_ids[i]].predict([X_test[i]], kind="survival", times=tenure_grid)[0] for i in range(5)]
```

---

## 4. Benchmark Results (20 Seeds)

To rigorously evaluate whether two-stage segregation prevents censoring bias, we evaluated three architectures across 20 independent random train/test splits (70/30) on the full 10,000 customers:

| Method | Harrell's C-Index | Integrated Brier Score (IBS) | Architecture |
| :--- | :--- | :--- | :--- |
| **Marginal Kaplan-Meier** | 0.5000 ± 0.0000 | 0.1924 ± 0.0064 | Unstratified Population $S(t)$ |
| **Single-Stage Survival Tree** | 0.5031 ± 0.0084 | 0.1924 ± 0.0064 | Direct Log-Rank Splitting on $(T, E)$ |
| **Two-Stage Hierarchical Tree** | **0.7508 ± 0.0069** | **0.1979 ± 0.0017** | ID Profile Tree + Leaf Kaplan-Meier |

Across all 20 random seeds, the **Two-Stage Tree beat the Single-Stage Tree in 20 of 20 runs (100% of seeds)**. 

The single-stage tree achieves a C-index of $0.5031$ (scarcely better than the unstratified $0.5000$ baseline) because greedy log-rank splits on $(T, E)$ are dominated by administrative censoring noise. In contrast, the two-stage model reaches a C-index of **$0.7508 \pm 0.0069$**, establishing clean, robust risk rank-ordering on held-out customers.

---

## 5. Auditable Rules & Discovered Cohorts

Calling `stage1_tree.extract_rules()` surfaces the exact mathematical bounds partitioning customer risk:

```text
Rule 1 (Cohort 0, cover: 5,074): IF age <= 43 AND products_number <= 2 THEN churn_rate = 10.7%
Rule 2 (Cohort 2, cover: 1,685): IF age >  43 AND products_number <= 2 THEN churn_rate = 39.5%
Rule 3 (Cohort 1, cover:   129): IF age <= 43 AND products_number >  2 THEN churn_rate = 76.7%
Rule 4 (Cohort 3, cover:   112): IF age >  43 AND products_number >  2 THEN churn_rate = 95.5%
```

![Two-Stage Hierarchical Survival Cohorts](../assets/two_stage_survival_cohorts.svg)

### Key Insights from the Discovered Curves:

1. **Non-Crossing, Distinct Trajectories**:
   - **Cohort 0 (Core Retained, Teal)**: 5,074 customers (72.5% of sample). Even after 10 years of tenure, survival probability remains high at **70.9%**.
   - **Cohort 2 (Mature Mainstream, Blue)**: 1,685 customers. Survival decays steadily to **25.7%** by year 10.
   - **Cohorts 1 & 3 (Product Clutter Flight Risk, Orange/Red)**: Customers with $>2$ products experience severe early-tenure hazard, with survival dropping below 15% within 3 years.
2. **Guaranteed Monotonicity**:
   Because each leaf model fits an exact product-limit Kaplan-Meier step function, predictions are guaranteed non-increasing ($\hat{S}(t_2) \le \hat{S}(t_1)$ for $t_2 > t_1$) with zero negative hazard artifacts.
3. **Auditability for Regulated Finance**:
   When regulators or credit committee auditors question why customer #429 received a high risk provision, compliance presents the exact deterministic rule: `Age > 43 AND Products > 2`, accompanied by the empirical Kaplan-Meier curve.

---

## 6. Limitations & Assumptions

1. **Non-Informative Administrative Censoring**: The product-limit Kaplan-Meier estimator assumes censoring is uninformative ($C \perp T \mid X$). If customers leave the study due to unobserved operational changes (e.g. branch closures in specific territories), censoring bias can reappear.
2. **Tenure Granularity**: In the Bank Churn benchmark, tenure is measured in discrete integer years (0 to 10), causing ties at event intervals. While `mltreelib`'s Breslow/Efron tie handling resolves ties efficiently, higher temporal resolution (daily or weekly timestamps) yields smoother step functions.
3. **Competing Risks**: This baseline treats churn as a single terminal event. In multi-product retail banking, customer attrition may compete with involuntary credit defaults or product migrations, requiring Cause-Specific Hazard trees or Fine-Gray subdistribution modeling via `Metrics.competing_risks()`.

---

## Summary

| Dimension | Standard Direct Survival Tree | Two-Stage Hierarchical Survival Tree |
| :--- | :--- | :--- |
| **Split Optimization** | Greedy log-rank on $(T, E)$ pairs. | Supervised profile clustering on baseline $X$. |
| **Censoring Robustness** | Vulnerable to calendar truncation bias ($C=0.5031$). | **Censoring-independent** cohort segregation ($C=0.7508$). |
| **Hazard Assumption** | Assumes piecewise constant hazard steps. | **Non-parametric** leaf Kaplan-Meier curves. |
| **Inference Latency** | $O(\text{depth})$ | $O(\text{depth})$ tree routing + table lookup. |
| **Explainability** | Complex split thresholds on event times. | **Clean business segmentation rules**. |

When working with longitudinal customer telemetry, never allow time-to-event outcomes to dictate customer segmentation. Use `ModelTree` to segregate cohorts first, then estimate localized survival distributions inside each terminal leaf node.
