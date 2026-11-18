# Check the Kaplan–Meier calculation inside a leaf

This example isolates the survival and cumulative hazard calculations. Use the [FLCHAIN pathway](../guides/survival.md) for a complete held-out evaluation on 7,874 real patients.

![Leaf survival curves and held-out censoring-aware scores on FLCHAIN](../assets/guides/real-survival.svg)

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

## 3. Querying Survival Probabilities & Cumulative Hazard

Query step probabilities and cumulative hazards across arbitrary time horizons:

```python
# Survival probability S(t) at day 30, 90, 180, 365
time_grid = [30.0, 90.0, 180.0, 365.0]
surv_curves = tree.predict(X_new, kind="survival", times=time_grid)

# Cumulative hazard Lambda(t) (Nelson-Aalen integral, not instantaneous hazard rate)
cum_hazard_curves = tree.predict(X_new, kind="hazard", times=time_grid)

# Median survival time in days (or max follow-up bound if S(t) > 0.5)
median_times = tree.predict(X_new, kind="response")
```

---

## 4. Assumptions & Limitations

To interpret survival trees validly in clinical or business workflows, four assumptions must be maintained:

1. **Independent (Non-Informative) Censoring**:
   Conditional on tree split covariates $X$, censoring time $C$ and event time $T$ must be conditionally independent: $T \perp C \mid X$. If patients leave a study or customers cancel because they are in unobserved terminal deterioration, censoring is informative and empirical Kaplan-Meier leaf estimates will overestimate survival.
2. **Synchronized Time Origin**:
   $t = 0$ must represent a clear, homogeneous milestone (e.g. initial diagnosis, medication onset, or account inception) across all units.
3. **Median Survival When $S(t) > 0.5$**:
   When fewer than 50% of units experience the event within a leaf during the observation window, the survival curve $\hat{S}(t)$ never crosses 0.5. In this case, the true median survival time is unobserved (censored beyond the maximum follow-up). The engine sets `median_time` to the maximum observed follow-up time $\max(T_i)$ within that leaf as a conservative empirical lower bound.
4. **Cumulative Hazard vs. Instantaneous Hazard Rate**:
   `predict(kind="hazard")` computes the cumulative hazard function $\hat{\Lambda}(t) = \int_0^t \lambda(u) du$ (dimensionless accumulated expected event count). It does not compute instantaneous hazard rate $\lambda(t) = \frac{d\Lambda(t)}{dt}$, which requires kernel smoothing or parametric baseline hazard assumptions.

---

## 5. Held-Out Censoring-Aware Evaluation

Evaluate predictions on independent test sets using `Metrics.survival`:

```python
from mltreelib import Metrics

# Held-out evaluation
report = Metrics.survival(
    times=test_times,
    events=test_events,
    pred_times=test_median_preds,
    eval_times=[30.0, 90.0, 180.0, 365.0],
    pred_surv=test_surv_curves,
)

print(report)
# Returns:
# - harrell_c: Harrell's Concordance Index (proportion of concordant evaluable pairs)
# - uno_c: Uno's IPCW Concordance Index (robust to heavy or time-dependent censoring)
# - brier_scores: Time-dependent IPCW Brier score BS(t) at each evaluation horizon
# - integrated_brier: Integrated Brier Score (IBS) across the evaluation grid
```

---

## 6. Benchmark & Numerical Validation

On the **Veterans' Administration Lung Cancer Trial** (`datasets/external/veteran.csv`):
- Splitting on patient Karnofsky performance scale (`karno`) segments patients into two starkly divergent cohorts:
  - Patients with Karnofsky score $\le 40$ have a 30-day survival rate of **31.6%**.
  - Patients with Karnofsky score $> 40$ have a 30-day survival rate of **84.8%**.
- **Numerical Validation**: The predicted survival probabilities $\hat{S}(t)$ match the empirical product-limit formula step-by-step to machine precision ($< 10^{-9}$).
