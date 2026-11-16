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

1. **`NaPolicy` Routing**: Missing sensor values (`None` / `NaN`) are preserved and evaluated as their own partition branch during split selection. The tree learns whether sensor dropout itself signals imminent failure.
2. **Cubist-Style Linear Leaves**: Instead of predicting constant step-functions, leaves fit localized linear models over recent operational telemetry (e.g., vibration vs pressure).

```
import mltreelib as mt

# Create rolling lags and temporal features natively
transformed = mt.timefeat.generate_lags(
    frame,
    timestamp_col="timestamp",
    target_cols=["bearing_temp", "vibration_rms"],
    lags=[1, 5, 15, 60]
)

# Model tree with linear leaf regressions (Cubist-style)
model = mt.ModelTree(
    leaf_type="linear",
    max_depth=6,
    min_leaf_samples=25
)
model.fit(transformed, target="hours_to_failure")
```

By computing streaming splits in pure Rust without heap allocations, `mltreelib` enables low-latency edge deployment for industrial controllers.
