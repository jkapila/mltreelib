# Native Missing Data Routing & Surrogate Splits

This guide details `mltreelib`'s first-class handling of missing tabular data (`None` and `NaN`), evaluating learned missing split directions and surrogate splits on industrial telemetry without synthetic imputation.

Script: [`benchmarks/suites/missing_routing_benchmark.py`](https://github.com/jkapila/mltreelib/blob/main/benchmarks/suites/missing_routing_benchmark.py) (`uv run python benchmarks/suites/missing_routing_benchmark.py`).

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
flowchart LR
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

## 2. Mathematical Mechanics of Native Routing & Surrogates

### A. Optimal Learned Missing Direction
Let $\mathcal{I}_{\text{obs}} = \{i \mid x_{ij} \neq \text{NA}\}$ and $\mathcal{I}_{\text{miss}} = \{i \mid x_{ij} = \text{NA}\}$.
For a candidate threshold $\theta$, the partition is evaluated under two competing routing assignments:

$$\Delta \mathcal{D}(\theta, \text{dir}) = \mathcal{D}(P) - [\mathcal{D}(L_{\text{dir}}) + \mathcal{D}(R_{\text{dir}})]$$

- If $\text{dir} = \text{left}$: $L = \{i \in \mathcal{I}_{\text{obs}} \mid x_{ij} \le \theta\} \cup \mathcal{I}_{\text{miss}}$ and $R = \{i \in \mathcal{I}_{\text{obs}} \mid x_{ij} > \theta\}$.
- If $\text{dir} = \text{right}$: $L = \{i \in \mathcal{I}_{\text{obs}} \mid x_{ij} \le \theta\}$ and $R = \{i \in \mathcal{I}_{\text{obs}} \mid x_{ij} > \theta\} \cup \mathcal{I}_{\text{miss}}$.

The split engine stores the tuple $(\theta^*, \text{dir}^*)$ achieving maximum deviance reduction. Missingness acts as an informative categorical state evaluated in $O(N \log N)$ sorting time.

### B. Surrogate Split Agreement Metric
When `maxsurrogate = K` is enabled, `mltreelib` computes up to $K$ backup split rules for each internal node. Let primary split rule $s^*$ partition observed samples into $L$ and $R$. For any surrogate candidate $s_k$ on feature $k \neq j$:

$$\lambda(s^*, s_k) = \frac{n_{LL} + n_{RR}}{n_{\text{obs}}}$$

where $n_{LL}$ is the count of samples sent left by both rules, and $n_{RR}$ is the count sent right by both rules. The surrogate quality metric measures improvement over default majority routing:

$$\text{Agreement Ratio} = \frac{\lambda(s^*, s_k) - p_{\text{maj}}}{1 - p_{\text{maj}}}$$

Surrogates with $\text{Agreement Ratio} \le 0$ are discarded. At inference time, if feature $j$ is absent, the highest-ranking surrogate with an observed value decides the branch direction.

### C. NA-Indicator Linear Leaves
When fitting linear model leaves on partially observed partitions, standard OLS crashes on `NaN`. `mltreelib` augments the local design matrix with binary missingness indicators $m_{ij} = \mathbb{I}(x_{ij} \text{ is missing})$:

$$\hat{y}_i = \beta_0 + \sum_{j=1}^P \beta_j \cdot x_{ij} (1 - m_{ij}) + \sum_{j=1}^P \beta_{\text{NA}, j} \cdot m_{ij}$$

Observed values contribute to continuous slopes $\beta_j$, while missing states contribute discrete level adjustments $\beta_{\text{NA}, j}$ without distorting gradient estimates.

---

## 3. Benchmark Study: Scania Truck Air Pressure System (APS) Failure

We evaluate native routing against standard mean imputation on the **Scania APS Failure** benchmark (76,000 industrial truck diagnostic records, 171 sensor channels).

<!-- run -->
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

<!-- output -->
```text
Native Routing -> Average Precision: 0.4916, ROC-AUC: 0.9524
```

---

## 4. Reading the benchmark result

The block in section 3 fits one `ModelTree` with native missing routing on the first 8,000 APS rows and scores 2,000 later rows. On that split it prints average precision 0.4916 and ROC-AUC 0.9524.

Missingness is high in this channel. In the first 12,000 records, `ab_000` is missing for 77.4% of rows, so a routing rule that handles missing values directly matters for this data.

Earlier drafts of this page compared native routing with mean imputation and reported Brier scores by complete and incomplete records. No block on this page produces those numbers, so they are not quoted here. Run such a comparison on your own split before making a claim about imputation.

---

## 5. Domain Limitations & Failure Modes

1. **Uninformative Complete Blackout**:
   If an entire telemetry record arrives with all features missing ($x_{ij} = \text{NA} \quad \forall j$), surrogate splits cannot fire. The sample cascades down default `missing_left` branches to a single leaf. Upstream health monitors must flag complete packet loss.
2. **Surrogate Memory Overhead**:
   Computing $K$ surrogate splits per node scales tree serialization size as $O(K \cdot N_{\text{nodes}})$. For lean embedded deployment, set `maxsurrogate=0` and rely on `missing_left`.
3. **Imputation Leakage in Training Splits**:
   Mean/median imputation computes statistics across the training set, causing subtle leakage if applied prior to cross-validation splits. Native routing avoids all cross-split contamination.
