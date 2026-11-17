# Zero-Imputation Predictive Maintenance for Industrial Sensor Feeds

**Author:** Jitin Kapila &amp; mltreelib Team  
**Category:** Industrie 4.0 &amp; Sensor Analytics

---

## The Flaw of Mean/Median Imputation in IoT

In manufacturing facilities, wind turbine farms, and oil &amp; gas pipelines, sensor dropouts are common. Telemetry connections drop, sensors saturate, or channels get unplugged during maintenance.

Standard machine learning pipelines default to synthetic imputation: filling missing readings with the historical mean, median, or KNN interpolations.

In predictive maintenance, **this destroys signal**: - A missing temperature reading often means the sensor overheated and faulted, not that it was at the average temperature (e.g. 25°C). - Imputing medians distorts rolling volatility, dampens trend acceleration, and triggers false negatives before catastrophic bearing failures.

---

## Native Routing and Piecewise Linear Models

`mltreelib` solves industrial telemetry through two mechanisms:

1. **Native Missing Routing**: Missing sensor values (`None` / `NaN`) are preserved at ingest and routed through primary and surrogate split logic (`maxsurrogate=1`). The tree learns directly whether sensor dropout itself signals imminent failure.
2. **Cubist-Style Linear Leaves**: Instead of predicting constant step-functions, leaves fit localized multivariate linear regressions over active operational telemetry.

```
from mltreelib import ModelTree

# Model tree with linear leaf regressions and surrogate missing routing
model = ModelTree(
    leaf="linear",
    max_depth=3,
    min_rows=30,
    maxsurrogate=1,    # Routes missing sensor channels without imputation
    prune="none",
    seed=999,
)

# Fit over raw telemetry matrix containing None / NaN dropouts
model.fit(X_train, y_train)

# Predict Remaining Useful Life (RUL) in hours
rul_predictions = model.predict(X_test)
```

---

## Data Analysis &amp; Sensor Telemetry Profile

In our bearing degradation benchmark (\\(N=1,000\\) machine monitoring hours), telemetry captures the following dynamics:

| Sensor Channel | Operational Range | Dropout Mechanism | Failure Degradation Impact |
| :--- | :---: | :---: | :--- |
| **`bearing_temp`** | 40°C to 105°C | Continuous (no drops) | Critical friction heat indicator; drives catastrophic wear when \\(\>85^\\circ\\text{C}\\). |
| **`vibration_rms`** | 0.5 to 6.5 mm/s | Continuous (no drops) | Mechanical harmonic resonance; high RMS accelerates bearing degradation. |
| **`oil_pressure`** | 25 to 55 psi | **\~20% Missing (`None`)** | When thermal friction spikes (\\(\>85^\\circ\\text{C}\\)), telemetry transmitters blackout. |
| **`hours_to_failure`** (RUL) | 10 to 350 hours | Target (continuous) | Continuous operational hours until thermal seizure or bearing raceway fracture. |

---

## Piecewise Linear Tree Architecture

Below is the trained piecewise model tree rendered via `model.to_dot()` and Graphviz:

Notice the dual mechanics: - **Decision Partitioning**: The tree isolates regimes by temperature (\\(T \> 83.2^\\circ\\text{C}\\)) and vibration (\\(V \> 3.4\\,\\text{mm/s}\\)). - **Local Linear Regressions**: Rather than a single flat step-function, each leaf evaluates a fitted hyperplane equation: $\\(\\text{RUL} = \\beta\_0 + \\beta\_1 \\cdot \\text{bearing\\\_temp} + \\beta\_2 \\cdot \\text{vibration\\\_rms} + \\beta\_3 \\cdot \\text{oil\\\_pressure}\\)$ capturing high-gradient degradation slopes near the end of machine life.

---

## Model Evaluation &amp; Diagnostics

Evaluating residual errors and split importance validates that native missing-value routing captures the physical failure mechanics:

1. **Actual vs. Predicted RUL**: Residuals cling tightly to the identity diagonal (\\(R^2 \> 0.85\\), RMSE \\(\< 15\\) hours), with zero artifact clustering from imputed medians.
2. **Feature Importance Shares**: Split gain importance confirms that `bearing_temp` and `vibration_rms` drive the macro regime partitions, while `oil_pressure` and its missing-dropout indicators calibrate fine-grained leaf survival curves.
