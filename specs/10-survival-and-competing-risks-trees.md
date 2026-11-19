# Spec 10: Survival and Competing Risks Trees

## Problem Statement

Time-to-event outcomes (clinical trials, patient progression, equipment mechanical failure, customer subscription churn) are subject to **censoring** (subjects leaving observation without an event) and **competing risks** (an alternative event precluding the primary event of interest, e.g., death from cardiovascular disease precluding cancer recurrence).

Standard regression and classification trees fail because treating censored times as observed leads to severe downward survival bias, while dropping censored observations causes catastrophic selection bias. Existing packages (`scikit-survival`, `randomForestSRC`, `survival`, `cmprsk`) either lack model-tree parametric leaves (offering only flat step-function Kaplan-Meier estimates), run slowly in Python or R, or treat competing risks by naively censoring the competing events (which mathematically overestimates cumulative incidence by 20% to 50%).

Recent breakthroughs (AISTATS 2025 *SurvivalBoost*, AAAI 2024 *Optimal Survival Trees*, and 2026 calibration frameworks) prove that proper separable scoring rules with Inverse Probability of Censoring Weights (IPCW) and exact subdistribution incidence models bridge the gap between black-box boosters and interpretable trees.

## Solution

Build a high-performance survival and competing risks tree system in `mltreelib-core`:
1. **Accelerated Failure Time (AFT) Parametric Leaves**: Fit log-linear survival models $\ln(T_i) = \mu + \beta^T X_i + \sigma \epsilon_i$ with Weibull, Lognormal, and Log-logistic distributions inside terminal leaves using L-BFGS, enabling continuous time predictions.
2. **Cox Proportional Hazards Leaves**: Fit Breslow/Efron partial likelihood within leaf partitions to output segment hazard ratios $\exp(\beta^T X)$ relative to leaf baseline hazards.
3. **Competing Risks Trees (Fine-Gray / CIF)**: Implement subdistribution hazard modeling, Gray's test splitting, and Aalen-Johansen Cumulative Incidence Functions (CIF) for mutually exclusive competing endpoints.
4. **Fast $O(N)$ Martingale Residual Streaming Scan**: Transform survival split search into exact scalar least-squares on martingale residuals $M_i = E_i - \hat{\Lambda}_{\text{parent}}(T_i)$, achieving sub-millisecond split scans on large cohorts.
5. **Exact Two-Sample Log-Rank & Weighted Tests**: Support Standard Log-Rank, Tarone-Ware ($w_j = \sqrt{n_j}$), and Peto-Prentice ($w_j = \hat{S}(t_j)$) statistics to flexibly weight early vs late prognostic divergence.
6. **Proper Separable Scoring Rule Splitting (SurvivalBoost)**: Censoring-adjusted IPCW loss splitting to directly optimize calibrated time-dependent probabilities for both single and competing risks.
7. **Harrell's and Uno's Concordance Index (C-Index)**: Evaluate and split on pairwise rank concordance between predicted mortality risk and true survival order.

## User Stories

1. As a biostatistician, I want `leaf="aft"` with `aft_dist="weibull"`, so that I can predict parametric survival times and hazard quantiles directly from patient covariates.
2. As a clinical researcher, I want `gain="logrank"` with `weighting="peto_prentice"`, so that the tree prioritizes splits with early treatment divergence when sample sizes are largest.
3. As a high-performance practitioner, I want `gain="martingale"`, so that split search executes in $O(N)$ linear streaming time via node-level martingale residuals.
4. As an oncology data scientist, I want `gain="c_index"`, so that my survival tree optimizes risk ranking concordance without assuming proportional hazards.
5. As an actuarial analyst, I want `gain="survival_brier"`, so that tree splits directly minimize calibration and discrimination error on time-dependent survival probabilities using IPCW weighting.
6. As a medical trial statistician, I want competing risks support (`events=[0, 1, 2]`, `event_of_interest=1`), so that the tree models the Cumulative Incidence Function (CIF) of cancer recurrence without biasing against competing non-cancer mortality.
7. As an epidemiologist, I want `gain="gray"`, so that tree splits maximize divergence in Gray's non-parametric cumulative incidence statistic.
8. As a reliability engineer, I want `leaf="cox"`, so that each failure regime leaf estimates continuous proportional hazard degradation slopes on operating temperature and vibration.
9. As a pharmaceutical researcher, I want `leaf="mixture_cure"`, so that the leaf estimates both the probability of long-term cure $\pi(X)$ and the survival latency of uncured patients.
10. As a software developer, I want `tree.predict_survival(X, time_points=[1.0, 2.0, 5.0])`, so that I can get calibrated survival probabilities at clinically relevant horizons.
11. As an oncologist, I want `tree.predict_cif(X, times, cause=1)` for competing risks, so that I can provide patients with absolute risk curves for each competing cause of death.
12. As a package user, I want survival tree training and scoring to run in pure Rust without `scipy` or `scikit-survival` dependencies, so that edge medical devices and microservices run without runtime bloat.
13. As a quality engineer, I want tied event times to be handled via Efron's approximation, so that heavily quantized survival times do not destabilize the partial likelihood.

## Implementation Decisions

- **Survival Data Contract**:
  Survival inputs enter `fit(X, y)` where $y$ is a 2-tuple `(times, events)` or distinct arrays `times=T, events=E`. For competing risks, $E_i \in \{0, 1, 2, \dots, K\}$ where $0 = \text{censored}$, $1 = \text{cause 1}$, $2 = \text{cause 2}$.
- **Criteria Additions**:
  - `GainFn::LogRank`: Exact two-sample log-rank $\chi^2$ statistic across ordered failure times.
  - `GainFn::LogRankWeighted { scheme: LogRankWeight }`: Supports Standard, Tarone-Ware ($\sqrt{N}$), and Peto-Prentice ($\hat{S}$).
  - `GainFn::Martingale`: $O(N)$ streaming variance reduction on Nelson-Aalen martingale residuals $M_i = E_i - \hat{\Lambda}_P(T_i)$.
  - `GainFn::GrayTest { cause_of_interest }`: Gray's subdistribution hazard score for competing risks.
  - `GainFn::CIndex`: Evaluates pairwise concordance score divergence via Fenwick tree.
  - `GainFn::SurvivalBrier { time_points }`: IPCW-weighted proper separable Brier score reduction.
- **Leaf Additions**:
  - `LeafKind::KaplanMeier`: Empirical non-parametric survival step function $\hat{S}(t)$.
  - `LeafKind::AalenJohansen`: Non-parametric multi-state Cumulative Incidence Functions $\hat{F}_j(t)$ for competing risks.
  - `LeafKind::Aft { dist: AftDist }`: Solves maximum likelihood for location $\mu$, scale $\sigma$, and slopes $\beta$ via `crates/mltreelib-core/src/solver/lbfgs.rs`. Supports Weibull, Lognormal, and Log-logistic.
  - `LeafKind::Cox`: Solves Breslow/Efron partial likelihood within leaf rows.
  - `LeafKind::MixtureCure`: Expectation-Maximization (EM) leaf estimator for cure fraction.
- **High Seam**:
  Extended methods on `ModelTree`:
  - `predict_survival(X, times)` -> survival probability matrix $[N \times T]$.
  - `predict_cumulative_hazard(X, times)` -> $[N \times T]$.
  - `predict_cif(X, times, cause)` -> cumulative incidence probability matrix $[N \times T]$.
  - `predict_risk(X)` -> continuous hazard score or median survival time $[N]$.

## Testing Decisions

- **Behavioral Tests**:
  - Veteran's Administration Lung Cancer benchmark (`veteran`): verify log-rank and AFT trees split on Karnofsky performance score and cell type, matching published survival literature.
  - Primary Biliary Cirrhosis benchmark (`pbc`): verify competing risks CIF separates liver transplantation from death without cumulative risk overestimation.
  - Monoclonal Gammopathy benchmark (`mgus2`): verify Gray's test isolates progression to plasma cell malignancy against competing mortality.
  - C-Index ranking test: verify tree predictions achieve higher Harrell's and Uno's C-index than naive linear Cox models on non-linear survival surfaces.
  - IPCW Brier score calibration test: verify predicted survival probabilities closely match empirical Kaplan-Meier curves within each leaf.

## Out of Scope

- Time-varying covariates with dynamic counting process $(T_{\text{start}}, T_{\text{stop}}]$ formats (focus is right/left/interval censored fixed covariates).
- High-dimensional spatial point process models.

## Further Notes

- Aligns directly with Ishwaran et al. (2008, 2014) *Random Survival Forests for Competing Risks*, LeBlanc & Crowley (1992, 1993) *Relative Risk Trees*, Fine & Gray (1999) *Proportional Subdistribution Hazards*, and Alberge et al. (AISTATS 2025) *Survival Models: Proper Scoring Rule and Stochastic Optimization with Competing Risks*.

## Implementation Status (Completed)

- **Pure Rust Core (`crates/mltreelib-core`)**:
  - `GainFn::LogRank`, `GainFn::LogRankWeighted { scheme: LogRankWeight }` (Standard, TaroneWare, PetoPrentice), `GainFn::Martingale` ($O(N)$ streaming scan), `GainFn::GrayTest { cause }`, `GainFn::SurvivalBrier`.
  - Leaf estimation in `leaf.rs` for `LeafFamily::Survival`: Kaplan-Meier $S(t)$, Nelson-Aalen cumulative hazard $\Lambda(t)$, and Aalen-Johansen multi-state Cumulative Incidence Functions $\hat{F}_c(t)$ with exact simplex invariance $\sum_{c=1}^J \hat{F}_c(t) + \hat{S}(t) = 1.0$.
  - Evaluation metrics in `metrics.rs`: `competing_risks_report` computing cause-specific Harrell's C-index, Uno's IPCW C-index, cause-specific Integrated Brier Score (IBS), and time-dependent Brier score matrices across evaluation horizons.
- **Python / PyO3 Shell (`crates/mltreelib-py`, `python/mltreelib`)**:
  - Unified prediction dispatcher: `predict(X, kind="survival", times=...)`, `predict(X, kind="hazard", times=...)`, `predict(X, kind="cif", times=..., cause=...)`, `predict(X, kind="risk")`, `predict(X, kind="response")`.
  - Metrics binding: `Metrics().competing_risks(...)` and top-level `competing_risks_report(...)` accepting dicts `{cause: cif_matrix}` or ndarray-like structures.
  - Integration script: `examples/competing_risks_pbc.py` generating `docs/assets/competing_risks_cif_curves.png`.
- **Verification**:
  - All 92 Rust tests passing (`cargo test`).
  - All 205 Python tests passing (`uv run pytest tests/ -q`).
  - 0-warning documentation build (`uv run zensical build`).
