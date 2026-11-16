# Survival Trees & Kaplan-Meier Estimation

In clinical trials, equipment failure analysis, and credit tenure modeling, outcome variables represent **time-to-event** subject to **right-censoring**: for some units, the study ends before the event occurs.

`ModelTree` provides native non-parametric **Survival Trees** (`leaf_family="survival"`) that estimate the full survival step function $S(t)$ and cumulative hazard function $\Lambda(t)$ directly within tree leaves without proportional hazards assumptions.

Script:
- [`examples/survival_benchmark.py`](https://github.com/jkapila/mltreelib/blob/main/examples/survival_benchmark.py) (`python3 examples/survival_benchmark.py`)

```text
======================================================================
1. VETERANS' LUNG CANCER CLINICAL SURVIVAL BENCHMARK
======================================================================
Total Patients: 137 | Deaths (Events): 128 | Censored: 9 (6.6%)
Follow-up Time: Min=1 days, Median=80 days, Max=999 days

Fitted Survival Tree: 2 Prognostic Cohorts

Predicted Survival Probability S(t):
Cohort                    | t=10d | t=30d | t=60d | t=100d | t=180d | t=365d
---------------------------------------------------------------------------
Poor Prognosis (Karno=30) | 0.737 | 0.316 | 0.132 | 0.079 | 0.039 | 0.039
Good Prognosis (Karno=80) | 0.960 | 0.848 | 0.695 | 0.548 | 0.291 | 0.111

Predicted Cumulative Hazard Lambda(t):
Poor Prognosis (Karno=30) | 0.298 | 1.105 | 1.925 | 2.375 | 2.875 | 2.875
Good Prognosis (Karno=80) | 0.041 | 0.164 | 0.361 | 0.594 | 1.217 | 2.151
  [Pass] Exact Kaplan-Meier Assertion: Tree S(t) matches manual product-limit estimator within 1e-9.
```

---

## 1. The Survival Tuple Target

Survival datasets require both time-to-event (or censoring) $T_i$ and event indicator $E_i \in \{0, 1\}$ (where $1 = \text{event occurred}$, $0 = \text{censored}$). Target inputs take a 2-tuple:

```python
from mltreelib import ModelTree

# X: clinical covariates (e.g. karno, age, diagtime)
# y: 2-tuple (survival_times, event_indicators)
y_tuple = (times, events)

tree = ModelTree(max_depth=2, min_rows=25, leaf_family="survival")
tree.fit(X, y_tuple)
```

---

## 2. Leaf Non-Parametric Estimators

Inside every leaf, the core Rust engine fits:

1. **Nelson-Aalen Cumulative Hazard**:
   $$\hat{\Lambda}(t) = \sum_{t_j \le t} \frac{d_j}{n_j}$$
   where $d_j$ is the number of events at time $t_j$ and $n_j$ is the number of individuals at risk.

2. **Kaplan-Meier Survival Function**:
   $$\hat{S}(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)$$

---

## 3. Querying Survival Probabilities & Hazard Rates

Query step probabilities across arbitrary time horizons:

```python
# Survival probability S(t) at day 30, 90, 180, 365
time_grid = [30.0, 90.0, 180.0, 365.0]
surv_curves = tree.predict(X_new, kind="survival", times=time_grid)

# Cumulative hazard Lambda(t)
hazard_curves = tree.predict(X_new, kind="hazard", times=time_grid)

# Median survival time in days
median_times = tree.predict(X_new, kind="response")
```

---

## 4. Benchmark & Numerical Validation

On the **Veterans' Administration Lung Cancer Trial** (`datasets/external/veteran.csv`):
- Splitting on patient Karnofsky performance scale (`karno`) segments patients into two starkly divergent cohorts:
  - Patients with Karnofsky score $\le 40$ have a 30-day survival rate of **31.6%**.
  - Patients with Karnofsky score $> 40$ have a 30-day survival rate of **84.8%**.
- **Numerical Validation**: The predicted survival probabilities $\hat{S}(t)$ match the empirical product-limit formula step-by-step to machine precision ($< 10^{-9}$).
