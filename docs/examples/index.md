# Example Library

Start with the [quickstart](../quickstart.md), then choose a suite for the modeling problem you want to solve. The guides explain the inputs, the model settings, and how to interpret the outputs.

## Run an example

Follow the [source installation instructions](../installation.md#from-source), then run the script from the repository root.

```bash
uv run python examples/quickstart.py
uv run python examples/suites/core_tree_architecture.py
```

Each guide identifies its datasets and shows the command for its suite. Dataset-backed examples can download data on first use. Plotting examples write figures under `docs/assets/`. Use the release extension for timing comparisons.

## The 9 Consolidated Example Suites

<div class="grid cards" markdown>

-   :lucide-cpu: __Suite 1: Core Tree Architecture & Growth Knobs__

    ---

    Two-Control decoupling (`ModelTree` $\leftrightarrow$ `LeafControl`), 0D leaf stats, and Chow structural break stopping on Yeh Concrete.

    [:octicons-arrow-right-24: Open Suite 1](core-tree-architecture.md)

-   :lucide-shield-check: __Suite 2: Structural Constraints & Honest Inference__

    ---

    Dual monotone split and slope constraints, interaction isolation, and sample-split honesty ($S^{\text{tr}}$ vs $S^{\text{est}}$) on Healthcare data.

    [:octicons-arrow-right-24: Open Suite 2](constraints-and-honesty.md)

-   :lucide-git-branch: __Suite 3: Native Missing Data Routing__

    ---

    Learned `missing_left` direction and surrogate splits (`maxsurrogate=N`) for zero-imputation telemetry triage on Scania APS.

    [:octicons-arrow-right-24: Open Suite 3](missing-data-routing.md)

-   :lucide-chart-scatter: __Suite 4: Distributional & Censored Leaf Families__

    ---

    Gamma, Poisson, Tweedie, and Tobit censored MLE leaves with 60%+ OLS attenuation correction across actuarial risk datasets.

    [:octicons-arrow-right-24: Open Suite 4](distributional-leaves.md)

-   :lucide-zap: __Suite 5: Dynamic Leaf Retraining & Adaptation__

    ---

    Microsecond warm-start leaf retraining, Huber loss breakdown resistance (25.8 vs 54.3 MPa under sensor glitch), and quantile corridors.

    [:octicons-arrow-right-24: Open Suite 5](hierarchical-retrain.md)

-   :lucide-sliders: __Suite 6: Statistical, Metric & Oblique Splitting__

    ---

    Chow structural break gate, oblique ridge splits, Gain Ratio, Twoing, SDR, and Quinlan rule extraction on Housing and Wine.

    [:octicons-arrow-right-24: Open Suite 6](statistical-and-metric-splitting.md)

-   :lucide-calendar-days: __Suite 7: Time Series Feature Engineering & Forecasting__

    ---

    Pure Rust calendar extraction, entity-grouped calendar day lags, additive decomposition, and out-of-time trend extrapolation.

    [:octicons-arrow-right-24: Open Suite 7](timeseries-forecasting.md)

-   :lucide-network: __Suite 8: Unsupervised Trees: Clustering & Anomaly Triage__

    ---

    Predictive Clustering Trees (PCT) with rule extraction and Extended Isolation Forest (EIF) with oblique cuts for multi-target grouping.

    [:octicons-arrow-right-24: Open Suite 8](unsupervised-trees.md)

-   :lucide-heart-pulse: __Suite 9: Survival & Competing Risks Trees__

    ---

    Gray's subdistribution test, weighted log-rank criteria, $O(N)$ streaming martingale scans, and Aalen-Johansen CIF simplex invariant.

    [:octicons-arrow-right-24: Open Suite 9](survival-trees.md)

-   :lucide-gauge: __Leaf Solver Benchmarks: 7 Optimization Algorithms__

    ---

    Pareto frontier of fit latency vs. accuracy across Analytical (`irls`), Quasi-Newton (`lbfgs`), Coordinate (`ccd`), and Evolutionary (`shade`, `cmaes`, `ga`, `nes`) solvers.

    [:octicons-arrow-right-24: Open Solver Benchmarks](solver-benchmarks.md)

</div>

---

### Detailed Mechanics Reference Table

| Suite | Focus Area | Benchmark Dataset | Key Mechanics Covered |
| :--- | :--- | :--- | :--- |
| **[Suite 1: Core Tree Architecture & Growth Knobs](core-tree-architecture.md)** | Core Systems | Yeh Concrete (1,030 rows) | Two-Control decoupling (`ModelTree` <-> `LeafControl`), 0D leaf stats, Chow hypothesis stopping. |
| **[Suite 2: Structural Constraints & Honest Inference](constraints-and-honesty.md)** | Physics & Governance | Healthcare Expenses (1,338 rows) | Dual monotone split and slope constraints, interaction isolation, sample-split honesty ($S^{\text{tr}}$ vs $S^{\text{est}}$). |
| **[Suite 3: Native Missing Data Routing](missing-data-routing.md)** | Industrial IoT | Scania APS Failure (76,000 rows) | Learned `missing_left` direction, surrogate splits (`maxsurrogate=N`), zero-imputation telemetry triage. |
| **[Suite 4: Distributional & Censored Leaf Families](distributional-leaves.md)** | Actuarial & Risk | Credit-G, Healthcare, Affairs, Mroz | Gamma, Poisson, Tweedie, and Tobit censored MLE leaves with 60%+ OLS attenuation correction. |
| **[Suite 5: Dynamic Leaf Retraining & Adaptation](hierarchical-retrain.md)** | Streaming & Systems | Yeh Concrete (1,030 rows) | Microsecond warm-start leaf retraining, Huber breakdown resistance (25.8 MPa vs 54.3 MPa under glitch), quantile corridors. |
| **[Suite 6: Statistical, Metric & Oblique Splitting](statistical-and-metric-splitting.md)** | Split Optimization | Boston/Miami Housing, Wine Quality | Chow structural break gate, oblique ridge splits, Gain Ratio, Twoing, SDR, and Rule extraction. |
| **[Suite 7: Time Series Feature Engineering & Forecasting](timeseries-forecasting.md)** | Temporal Econometrics | Multi-store retail demand panel | Pure Rust calendar extraction, entity-grouped calendar day lags, additive decomposition, and out-of-time trend extrapolation. |
| **[Suite 8: Unsupervised Trees: Clustering & Anomaly Triage](unsupervised-trees.md)** | Discovery & Triage | Wine Quality & 2D Mixture | Predictive Clustering Trees (PCT) with rule extraction, Extended Isolation Forest (EIF) with oblique cuts. |
| **[Suite 9: Survival & Competing Risks Trees](survival-trees.md)** | Clinical & Churn | Mayo PBC, Veterans, FLCHAIN | Gray's test, Tarone-Ware/Peto/Martingale, Aalen-Johansen CIF simplex invariant ($\sum F_c + S = 1$), Uno IPCW C. |
| **[Leaf Solver Benchmarks](solver-benchmarks.md)** | Optimization Systems | Yeh Concrete (1,030 rows) | Analytical Cholesky, Quasi-Newton L-BFGS, Cyclic Coordinate Descent, and Evolutionary Solvers (SHADE, CMA-ES, GA, NES). |

---

## Architectural Workflow Navigator

```mermaid
flowchart TD
    Start["What modeling challenge do you need to solve?"] --> Q1{"What is the target / learning setting?"}
    
    Q1 -- "Continuous Tabular Response" --> A1["Suite 1: Core Tree Architecture\n(Linear Leaves & Growth Knobs)"]
    Q1 -- "Preserve Domain Laws / Fairness" --> A2["Suite 2: Structural Constraints\n(Monotone & Honest Inference)"]
    Q1 -- "Intermittent Telemetry / Missing Inputs" --> A3["Suite 3: Native Missing Data\n(Learned Routes & Surrogates)"]
    Q1 -- "Heavy Tails / Zero Inflation / Censoring" --> A4["Suite 4: Distributional Leaves\n(Tweedie, Gamma, Tobit)"]
    Q1 -- "Sub-Millisecond Adaptation & Online Streams" --> A5["Suite 5: Dynamic Leaf Retraining\n(Warm-Start Hierarchy & Huber)"]
    Q1 -- "Structural Parameter Breaks & Oblique Cuts" --> A6["Suite 6: Statistical & Metric Splitting\n(Chow Test & Oblique Scans)"]
    Q1 -- "Autoregressive Demand & Seasonality" --> A7["Suite 7: Time Series Forecasting\n(Calendar Lags & Additive Decomp)"]
    Q1 -- "Unsupervised Grouping / Outlier Triage" --> A8["Suite 8: Unsupervised Trees\n(PCT Clustering & EIF Anomaly)"]
    Q1 -- "Right-Censored / Competing Multi-Cause" --> A9["Suite 9: Survival & Competing Risks\n(Gray's Test, Aalen-Johansen CIF)"]

    classDef startBox fill:#1e293b,stroke:#475569,stroke-width:2px,color:#ffffff;
    classDef gateBox fill:#b45309,stroke:#d97706,stroke-width:2px,color:#ffffff;
    classDef suiteBox fill:#1e3a5f,stroke:#2b4c7e,stroke-width:2px,color:#ffffff;
    classDef survBox fill:#065f46,stroke:#059669,stroke-width:2px,color:#ffffff;

    class Start startBox;
    class Q1 gateBox;
    class A1,A2,A3,A4,A5,A6,A7,A8 suiteBox;
    class A9 survBox;
```

---

## Reference Guides & End-to-End Studies

- For complete industry case studies on regulatory credit scoring, sensor maintenance, and causal marketing, visit the **[Practical Modeling Pathways](../guides/index.md)**.
- For complete function signatures and parameter specifications, visit the **[API Reference](../reference/model.md)**.
