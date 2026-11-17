# Why India's Digital Lending Boom Needs Glass-Box Trees Over Black-Box Boosters

**Author:** Jitin Kapila &amp; mltreelib Team  
**Category:** Fintech &amp; Regulatory Compliance

---

## The Regulatory Dilemma in Digital Lending

Digital credit underwriting is exploding globally—particularly in India, where UPI, alternate data, and NBFC lending evaluate millions of borrowers daily. Teams naturally turn to high-capacity gradient boosting algorithms (XGBoost, LightGBM, CatBoost) to squeeze out extra AUC.

However, in credit risk modeling, **predictive accuracy is secondary to regulatory compliance and legal auditability**: - Under Fair Lending regulations, FCRA (US), GDPR Article 22 (EU), and RBI digital lending directives (India), lenders must explain *adverse actions* (e.g., credit denial reasons) deterministically. - Approximating a 1,000-tree gradient booster using post-hoc SHAP or LIME explainers creates instability: neighboring perturbation samples can yield conflicting attribution rankings. - Regulated credit scorecards require **monotonicity**: an applicant's probability of default must strictly decrease as income increases or past delinquencies drop. Black-box ensembles frequently violate monotonicity across local sub-regions.

---

## Why Glass-Box Trees Win

`mltreelib` approaches credit scoring through transparent, verified tree mechanics with strict monotonic shape constraints:

```
from mltreelib import ModelTree

# Features: [cibil_score, monthly_income, delinquencies_24m, debt_to_income]
# Monotone constraints:
# - cibil_score: -1 (higher bureau score strictly decreases default probability)
# - monthly_income: 0 (unconstrained)
# - delinquencies_24m: +1 (more overdue records strictly increases default probability)
# - debt_to_income: +1 (higher DTI strictly increases default probability)
model = ModelTree(
    gain="gini",
    max_depth=3,
    min_rows=40,
    monotone=[-1, 0, 1, 1],
    prune="none",
    seed=123,
)
col_names = ["cibil_score", "monthly_income", "delinquencies_24m", "debt_to_income"]
model.fit(X_train, y_train, col_names=col_names)

# Predict calibrated default probabilities
pd_scores = model.predict(X_test)
```

---

## Data Analysis &amp; Borrower Population

In our underwriting benchmark (\\(N=1,200\\) applicants), applicant profiles reflect standard retail loan portfolios:

| Feature | Type | Range / Distribution | Monotone Constraint | Economic Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **`cibil_score`** | Continuous | 300 to 850 (\\(\\mu=680, \\sigma=75\\)) | **\-1 (Decreasing)** | Higher credit reputation implies lower default risk. |
| **`monthly_income`** | Continuous | ₹2,000 to ₹18,000 | 0 (Free) | General capacity to repay. |
| **`delinquencies_24m`** | Count | 0 to 4 past 30+ DPD events | **\+1 (Increasing)** | Prior delinquency strongly signals credit distress. |
| **`debt_to_income`** | Continuous | 15% to 75% | **\+1 (Increasing)** | High leverage increases sensitivity to income shocks. |

---

## Deterministic Scorecard Tree Structure

Below is the trained glass-box scorecard rendered via `model.to_dot()` and Graphviz:

Every applicant follows an exact boolean logic path. For example: - `IF cibil_score <= 656.3 AND delinquencies_24m >= 2 THEN Decline (PD = 78.4%)` - `IF cibil_score > 714.2 AND debt_to_income <= 0.42 THEN Auto-Approve (PD = 3.2%)`

Adverse action notices (e.g. "Credit denied due to high 24-month delinquency count and debt-to-income ratio exceeding policy thresholds") are guaranteed mathematically by leaf node membership—no post-hoc explainer heuristics required.

---

## Regulatory Scorecard Diagnostics

Regulatory validation requires verifying discrimination, maximum separation, monotonic risk ordering, and population stability across periods:

1. **ROC Discrimination**: Strong separation power (\\(AUC \> 0.75\\)) despite simple tree depth.
2. **Kolmogorov-Smirnov (KS)**: Maximum separation distance between good and bad borrowers peaks above 45% at the scorecard cutoff.
3. **Monotonic Decile Capture**: Event rates increase strictly monotonically across credit score deciles—essential for regulatory rating agency validation.
4. **Population Stability Index (PSI)**: Evaluates distribution shift between training and out-of-time test batches (\\(PSI \< 0.10\\), indicating stable population dynamics).
