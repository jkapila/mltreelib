# Spec 10: Survival and Competing Risks Trees

## Problem Statement

Time-to-event outcomes (clinical trials, patient progression, equipment mechanical failure, customer subscription churn) are subject to **censoring** (subjects leaving observation without an event) and **competing risks** (an alternative event precluding the primary event of interest, e.g., death from cardiovascular disease precluding cancer recurrence).

Standard regression and classification trees fail because treating censored times as observed leads to severe downward survival bias, while dropping censored observations causes catastrophic selection bias. Existing packages (`scikit-survival`, `randomForestSRC`, `survival`, `cmprsk`) either lack model-tree parametric leaves (offering only flat step-function Kaplan-Meier estimates), run slowly in Python or R, or treat competing risks by naively censoring the competing events (which mathematically overestimates cumulative incidence by 20% to 50%).

## Solution

Build a high-performance survival and competing risks tree system in `mltreelib-core`:
1. **Accelerated Failure Time (AFT) Parametric Leaves**: Fit log-linear survival models $\ln(T_i) = \mu + \beta^T X_i + \sigma \epsilon_i$ with Weibull, Lognormal, and Log-logistic distributions inside terminal leaves, enabling continuous time predictions.
2. **Cox Proportional Hazards Leaves**: Fit Breslow/Efron partial likelihood within leaf partitions to output segment hazard ratios $\exp(\beta^T X)$ relative to leaf baseline hazards.
3. **Competing Risks Trees (Fine-Gray / CIF)**: Implement subdistribution hazard modeling and Gray's test splitting to estimate non-parametric Cumulative Incidence Functions (CIF) for mutually exclusive competing endpoints.
4. **Harrell's Concordance Index (C-Index) Splitting**: Split nodes directly by maximizing rank concordance between predicted mortality risk and true survival order, robust when proportional hazards assumptions fail.
5. **Integrated Brier Score (IBS) Calibrated Splitting**: Split nodes to minimize the time-integrated squared error of survival probability estimates weighted by Inverse Probability of Censoring Weights (IPCW).
6. **Weighted Log-Rank Tests**: Support Tarone-Ware ($w_j = \sqrt{n_j}$) and Peto-Prentice ($w_j = \hat{S}(t_j)$) statistics to flexibly weight early vs late prognostic divergence.

## User Stories

1. As a biostatistician, I want `leaf="aft"` with `aft_dist="weibull"`, so that I can predict parametric survival times and hazard quantiles directly from patient covariates.
2. As a clinical researcher, I want `gain="logrank"` with `weighting="peto_prentice"`, so that the tree prioritizes splits with early treatment divergence when sample sizes are largest.
3. As an oncology data scientist, I want `gain="c_index"`, so that my survival tree optimizes risk ranking concordance without assuming proportional hazards.
4. As an actuarial analyst, I want `gain="brier_score"`, so that tree splits directly minimize calibration and discrimination error on time-dependent survival probabilities.
5. As a medical trial statistician, I want competing risks support (`events=[0, 1, 2]`, `event_of_interest=1`), so that the tree models the Cumulative Incidence Function (CIF) of cancer recurrence without biasing against competing non-cancer mortality.
6. As an epidemiologist, I want `gain="gray"`, so that tree splits maximize divergence in Gray's non-parametric cumulative incidence statistic.
7. As a reliability engineer, I want `leaf="cox"`, so that each failure regime leaf estimates continuous proportional hazard degradation slopes on operating temperature and vibration.
8. As a pharmaceutical researcher, I want `leaf="mixture_cure"`, so that the leaf estimates both the probability of long-term cure $\pi(X)$ and the survival latency of uncured patients.
9. As a software developer, I want `tree.predict_survival(X, time_points=[1.0, 2.0, 5.0])`, so that I can get calibrated survival probabilities at clinically relevant horizons.
10. As an oncologist, I want `tree.predict_cif(X, times)` for competing risks, so that I can provide patients with absolute risk curves for each competing cause of death.
11. As a package user, I want survival tree training and scoring to run in pure Rust without `scipy` or `scikit-survival` dependencies, so that edge medical devices and microservices run without runtime bloat.
12. As a quality engineer, I want tied event times to be handled via Efron's approximation, so that heavily quantized survival times do not destabilize the partial likelihood.

## Implementation Decisions

- **Survival Data Contract**:
  Survival inputs enter `fit(X, y)` where $y$ is a 2-tuple `(times, events)` or distinct arrays `times=T, events=E`. For competing risks, $E_i \in \{0, 1, 2, \dots, K\}$ where $0 = \text{censored}$, $1 = \text{cause 1}$, $2 = \text{cause 2}$.
- **Criteria Additions**:
  - `GainFn::LogRankWeighted { scheme: LogRankWeight }`: Supports Standard, Tarone-Ware ($\sqrt{N}$), and Peto-Prentice ($\hat{S}$).
  - `GainFn::CIndex`: Evaluates pairwise concordance score divergence.
  - `GainFn::BrierScore { time_points }`: IPCW-weighted Brier score reduction.
  - `GainFn::GrayTest { cause_of_interest }`: Gray's subdistribution hazard score for competing risks.
- **Leaf Additions**:
  - `LeafKind::Aft { dist: AftDist }`: Solves Newton-Raphson maximum likelihood for location $\mu$, scale $\sigma$, and slopes $\beta$.
  - `LeafKind::Cox`: Solves Breslow/Efron partial likelihood within leaf rows.
  - `LeafKind::CompetingRisks`: Computes Aalen-Johansen CIF estimates per cause of death.
  - `LeafKind::MixtureCure`: Expectation-Maximization (EM) leaf estimator for cure fraction.
- **High Seam**:
  Extended methods on `ModelTree`:
  - `predict_survival(X, times)` -> survival probability matrix $[N \times T]$.
  - `predict_cumulative_hazard(X, times)` -> $[N \times T]$.
  - `predict_cif(X, times, cause)` -> cumulative incidence probability matrix $[N \times T]$.

## Testing Decisions

- **Behavioral Tests**:
  - Veteran's Administration Lung Cancer benchmark (`veteran.csv`): verify log-rank and AFT trees split on Karnofsky performance score and cell type, matching published survival literature.
  - Competing risks simulation: generate two competing causes with opposing covariate effects; verify Gray's test tree splits to separate cause 1 risk from cause 2 risk.
  - C-Index ranking test: verify tree predictions achieve higher Harrell's C-index than naive linear Cox models on non-linear survival surfaces.
  - IPCW Brier score calibration test: verify predicted survival probabilities closely match empirical Kaplan-Meier curves within each leaf.

## Out of Scope

- Time-varying covariates with dynamic counting process $(T_{\text{start}}, T_{\text{stop}}]$ formats (focus is right/left/interval censored fixed covariates).
- High-dimensional spatial point process models.

## Further Notes

- Complements Ishwaran et al. (2008) *Random Survival Forests* and Fine & Gray (1999) *Proportional Subdistribution Hazards*.
