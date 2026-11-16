# Why Credit Scoring Needs Glass-Box Trees Over Black-Box Boosters

**Author:** Jitin Kapila & mltreelib Team  
**Category:** Fintech & Regulatory Compliance  

---

## The Regulatory Dilemma in Digital Lending

Digital credit underwriting is exploding globally—particularly in India, where UPI, alternate data, and NBFC lending evaluate millions of borrowers daily. Teams naturally turn to high-capacity gradient boosting algorithms (XGBoost, LightGBM, CatBoost) to squeeze out extra AUC.

However, in credit risk modeling, **predictive accuracy is secondary to regulatory compliance and legal auditability**:
- Under Fair Lending regulations, FCRA (US), GDPR Article 22 (EU), and RBI digital lending directives (India), lenders must explain *adverse actions* (e.g., credit denial reasons) deterministically.
- Approximating a 1,000-tree gradient booster using post-hoc SHAP or LIME explainers creates instability: neighboring perturbation samples can yield conflicting attribution rankings.
- Regulated credit scorecards require **monotonicity**: an applicant's probability of default must strictly decrease as income increases or past delinquencies drop. Black-box ensembles frequently violate monotonicity across local sub-regions.

---

## Why Glass-Box Trees Win

`mltreelib` approaches credit scoring through transparent, verified tree mechanics:

```python
import mltreelib as mt

# Define schema with native categorical levels and continuous features
schema = mt.Schema.from_dict({
    "cibil_score": "float",
    "monthly_income": "float",
    "delinquencies_24m": "int",
    "employment_type": "category"
})

# Ingest dirty applicant data without synthetic imputation
frame = mt.Frame.from_pandas(df, schema=schema, na_policy=mt.NaPolicy.Preserve)

# Fit glass-box tree with monotone constraints
model = mt.ModelTree(
    max_depth=5,
    monotone_constraints={"cibil_score": -1, "delinquencies_24m": 1},
    criterion="gini"
)
model.fit(frame, target="default_flag")
```

### Key Advantages for Fintech Underwriting

1. **Deterministic Adverse Action Codes**: Every applicant follows an exact boolean logic path (e.g., `IF cibil_score < 680 AND delinquencies_24m >= 2 THEN Decline`).
2. **First-Class Missingness (`None`/`NaN`)**: Borrowers with thin credit files or missing employer verification are not filled with artificial medians. Missingness is treated as an explicit branch signal.
3. **No Sklearn Pipeline Overhead**: Ingestion, schema validation, target encoding (`GroupedScaler`), and tree splits live in the same Rust engine.
