# Practical modeling pathways

Start with a decision. Each pathway uses a real dataset, shows the fitting code and reads the result against a baseline or diagnostic.

| Decision | Path | Dataset | What to check |
| --- | --- | --- | --- |
| Estimate a price | [Regression](<https://jkapila.github.io/mltreelib/guides/regression/index.md>) | OpenML diamonds | Test errors and residuals |
| Predict financial distress | [Classification](<https://jkapila.github.io/mltreelib/guides/classification/index.md>) | Give Me Some Credit | Ranking, probability and threshold cost |
| Group wine by chemistry | [Clustering](<https://jkapila.github.io/mltreelib/guides/clustering/index.md>) | UCI Wine Quality via OpenML | Leaf profiles and refit stability |
| Prioritize unusual readings | [Anomaly detection](<https://jkapila.github.io/mltreelib/guides/anomaly/index.md>) | UCI Statlog Shuttle | Precision at a fixed review budget |
| Estimate survival over time | [Survival](<https://jkapila.github.io/mltreelib/guides/survival/index.md>) | FLCHAIN | Censoring-aware error by horizon |
| Decide who gets an email | [Uplift](<https://jkapila.github.io/mltreelib/guides/uplift/index.md>) | Hillstrom experiment | Policy value against treat-all and treat-none |

## How to use the documentation

1. **Run one pathway.** Get its dataset and follow the analysis from split through evaluation.
2. **Change one part.** The [example library](<https://jkapila.github.io/mltreelib/examples/index.md>) covers missing routing, alternative leaves, calibration diagnostics, retraining and other focused variations.
3. **Review the decision.** Use the [evaluation guide](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>), [visualization catalog](<https://jkapila.github.io/mltreelib/guides/visualization/index.md>) and [serving contract](<https://jkapila.github.io/mltreelib/guides/serving-contract/index.md>).

The [dataset setup page](<https://jkapila.github.io/mltreelib/guides/real-data/index.md>) lists the downloads. Synthetic checks sit after the practical workflows. They make known effects and missing-data behavior easier to test.
