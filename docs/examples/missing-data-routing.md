# Native Missing Data Routing & Surrogate Splits

This guide details `mltreelib`'s first-class handling of missing tabular data (`None` and `NaN`), evaluating learned missing split directions and surrogate splits on industrial telemetry without synthetic imputation.

Script: [`examples/suites/missing_routing_benchmark.py`](https://github.com/jkapila/mltreelib/blob/main/examples/suites/missing_routing_benchmark.py) (`uv run python examples/suites/missing_routing_benchmark.py`).

---

## 1. Domain Context & The Imputation Failure Mode

In mission-critical sensor telemetry (such as heavy truck fleet diagnostics, aerospace health monitoring, and clinical vital tracking), missing values occur regularly due to intermittent sensor disconnections, telemetry dropouts, or power cycles.

Standard tabular workflows apply global mean, median, or MICE imputation before fitting. Imputation introduces three dangerous failure modes:
1. **Covariance Distortion**: Imputing a constant mean collapses the conditional variance of missing sensors and creates artificial point-mass spikes in the distribution.
2. **Destruction of Missingness Signal**: In engineering systems, the *fact* that a sensor stopped reporting is often highly correlated with system failure. Imputation erases this indicator.
3. **Serving-Time Dependency**: In production, streaming scoring pipelines must maintain online imputation buffers across dozens of historical distributions.

`mltreelib` eliminates imputation completely through **Native Rust-Level Missing Routing**:
- **Learned Missing Direction (`missing_left`)**: During node partition search, candidate splits score sending missing records to the left branch versus the right branch, selecting the direction that minimizes node deviance or SSE.
- **Surrogate Splits (`maxsurrogate=N`)**: When a primary split feature is missing at prediction time, the tree inspects pre-computed correlated surrogate features to route the sample along the optimal path.
- **NA-Indicator Linear Leaves**: Linear leaves fit dedicated indicator coefficients $\beta_{\text{NA}, j}$ for absent features, preserving local linear slopes for observed predictors.

```mermaid
flowchart TD
    A["Incoming Feature Vector\n[x1, x2, ..., xk]"] --> B{"Is Primary Split Feature\nx_j Observed?"}
    
    B -- "Observed" --> C{"Predicate Test:\nx_j <= Threshold?"}
    C -- "True" --> Left["Left Child Node"]
    C -- "False" --> Right["Right Child Node"]
    
    B -- "Missing (None / NaN)" --> D{"Are Surrogate Splits\nConfigured & Available?"}
    
    D -- "Surrogate Feature Available" --> E{"Surrogate Predicate Test:\nx_surrogate <= Threshold?"}
    E -- "Agrees Left" --> Left
    E -- "Agrees Right" --> Right
    
    D -- "No Surrogates Available" --> F{"Learned Missing Route\n(missing_left)"}
    F -- "missing_left = True" --> Left
    F -- "missing_left = False" --> Right

    classDef dataBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef gateBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef altGate fill:#4338ca,stroke:#6366f1,stroke-width:2px,color:#ffffff;
    classDef leafBox fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;

    class A dataBox;
    class B,C gateBox;
    class D,E,F altGate;
    class Left,Right leafBox;
```

---

## 2. Benchmark Study: Scania Truck Air Pressure System (APS) Failure

We evaluate native routing against standard mean imputation on the **Scania APS Failure** benchmark (76,000 industrial truck diagnostic records, 171 sensor channels).

```python
from mltreelib import Metrics, ModelTree, load_dataset

# 1. Load Scania truck telemetry benchmark
data = load_dataset("APSFailure")
target = "AirPressureSystemFailure"
features = [
    "aa_000", "ab_000", "ac_000", "ad_000", "ae_000", "af_000",
    "ag_000", "ag_001", "ag_002", "ag_003", "ag_004", "ag_005"
]
n_sample = 12000

y_raw = data["columns"][target][:n_sample]
y = [1 if v in ("pos", "1", 1) else 0 for v in y_raw]
X = [[data["columns"][c][i] for c in features] for i in range(n_sample)]

split = 8000
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:10000], y[split:10000]

# 2. Fit ModelTree with Native Missing Routing and 2 Surrogates per node
tree_native = ModelTree(
    max_depth=3,
    min_rows=50,
    maxsurrogate=2,
    leaf="mean",
    leaf_family="binomial",
    gain="deviance",
)
tree_native.fit(X_train, y_train, col_names=features)

# 3. Evaluate on held-out test telemetry
probs = tree_native.predict_proba(X_test)
metrics = Metrics()
pr = metrics.pr_curve(y_test, probs)
roc = metrics.roc_curve(y_test, probs)

print(f"Native Routing -> Average Precision: {pr['average_precision']:.4f}, ROC-AUC: {roc['auc']:.4f}")
```

---

## 3. Empirical Results: Native Routing vs Mean Imputation

```text
Loaded Scania APS Failure benchmark: 12000 samples, 12 sensor channels.
Train rows: 8000, Val rows: 2000, Test rows: 2000
Sensor 'ab_000' missing rate: 77.4%

=== 1. Native Routing & Surrogate Splits (maxsurrogate=2) ===
Primary split node: aa_000 <= 101748.00 (missing_left=True)
Native Routing   -> Average Precision: 0.5924, ROC-AUC: 0.9848, Brier: 0.0135

=== 2. Comparison against Standard Mean Imputation Baseline ===
Mean Imputation  -> Average Precision: 0.5706, ROC-AUC: 0.9846, Brier: 0.0139
--> Average Precision advantage of Native Missing Routing: +3.8%

Complete Records   (n=394): Brier = 0.0060
Incomplete Records (n=1606): Brier = 0.0153
Native routing preserves calibrated probabilities even when multiple sensor channels are missing.
```

| Evaluation Strategy | Missing Handling | PR Average Precision | ROC-AUC | Brier Score |
|---|---|:---:|:---:|:---:|
| **Mean Imputation Baseline** | Global Column Mean | 0.5706 | 0.9846 | 0.0139 |
| **`mltreelib` Native Routing** | Learned `missing_left` + Surrogates | **0.5924** | **0.9848** | **0.0135** |

![Native Missing Data Routing and Surrogate Performance on Scania APS Telemetry](../assets/missing_routing_aps.png)

### Key Quantitative Findings

1. **High Missingness Tolerance (Panel 1)**: Key diagnostic sensors like `ab_000` exhibit **77.4% missingness** in production telemetry. Standard imputation collapses 77% of this channel to a single scalar value, destroying its variance.
2. **Superior Precision-Recall (Panel 2)**: Native missing routing achieves **0.5924 Average Precision**, outperforming the mean imputation baseline by **+3.8%**.
3. **Calibrated Probabilities on Incomplete Telemetry (Panel 4)**: On 1,606 test records with at least one missing sensor, `ModelTree` achieves a **0.0153 Brier score**, maintaining reliable probability calibration without collapsing to the prior mean.
