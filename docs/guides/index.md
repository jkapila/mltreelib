# Practical modeling pathways

Start with a decision. Each pathway uses a real cited dataset, shows the complete fitting code, reads empirical metrics against baseline benchmarks, and provides transparent decision inspection.

```mermaid
flowchart TD
    Start["What is your core machine learning task?"] --> Target{"What is the target response type?"}
    
    Target -- "Continuous Numeric" --> Dist{"Is the target heavy-tailed, censored, or non-normal?"}
    Dist -- "Standard Gaussian / Linear" --> G_Reg["Regression Pathway\n(Piecewise Linear ModelTree)"]
    Dist -- "Heavy-tailed / Zero-inflated / Censored" --> G_Dist["Distributional Leaves Suite\n(Gamma, Poisson, Tweedie, Tobit)"]
    
    Target -- "Binary / Multi-Class Categorical" --> G_Cla["Classification Pathway\n(Glass-Box Credit Distress & Review Queue)"]
    
    Target -- "Time-to-Event with Right Censoring" --> G_Surv["Survival Pathway\n(Log-Rank Splits & Nelson-Aalen Corridors)"]
    
    Target -- "Randomized Treatment / A/B Intervention" --> G_Uplift["Uplift Pathway\n(Honest Causal Trees & CATE Targeting)"]
    
    Target -- "Unlabeled Observations (Unsupervised)" --> Unsup{"Are you isolating outliers or finding segments?"}
    Unsup -- "Discover Homogeneous Cohorts" --> G_Clust["Clustering Pathway\n(Predictive Clustering Trees with Rules)"]
    Unsup -- "Surface Anomalous Telemetry" --> G_Anom["Anomaly Detection Pathway\n(Extended Isolation Forest without Ghosts)"]

    classDef startBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef gateBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef regBox fill:#1e3a5f,stroke:#2b4c7e,stroke-width:2px,color:#ffffff;
    classDef distBox fill:#4c1d95,stroke:#6d28d9,stroke-width:2px,color:#ffffff;
    classDef claBox fill:#9a3412,stroke:#c2410c,stroke-width:2px,color:#ffffff;
    classDef survBox fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;
    classDef upliftBox fill:#0f766e,stroke:#14b8a6,stroke-width:2px,color:#ffffff;
    classDef clustBox fill:#0369a1,stroke:#0284c7,stroke-width:2px,color:#ffffff;
    classDef anomBox fill:#881337,stroke:#9f1239,stroke-width:2px,color:#ffffff;

    class Start startBox;
    class Target,Dist,Unsup gateBox;
    class G_Reg regBox;
    class G_Dist distBox;
    class G_Cla claBox;
    class G_Surv survBox;
    class G_Uplift upliftBox;
    class G_Clust clustBox;
    class G_Anom anomBox;
```

| Decision | Path | Dataset | What to check |
|---|---|---|---|
| Estimate a price | [Regression](regression.md) | OpenML diamonds | Test errors and residuals |
| Predict financial distress | [Classification](classification.md) | Give Me Some Credit | Ranking, probability and threshold cost |
| Group wine by chemistry | [Clustering](clustering.md) | UCI Wine Quality via OpenML | Leaf profiles and refit stability |
| Prioritize unusual readings | [Anomaly detection](anomaly.md) | Scania APS Failure / Shuttle | Precision at a fixed review budget |
| Estimate survival over time | [Survival](survival.md) | FLCHAIN | Censoring-aware error by horizon |
| Decide who gets an email | [Uplift](uplift.md) | Hillstrom experiment | Policy value against treat-all and treat-none |

## How to use the documentation

1. **Run one pathway.** Get its dataset and follow the analysis from split through evaluation.
2. **Change one part.** The [example library](../examples/index.md) covers missing routing, alternative leaves, calibration diagnostics, retraining and other focused variations.
3. **Review the decision.** Use the [evaluation guide](evaluation.md), [visualization catalog](visualization.md) and [serving contract](serving-contract.md).

The [dataset setup page](real-data.md) lists the downloads. Synthetic checks sit after the practical workflows. They make known effects and missing-data behavior easier to test.
