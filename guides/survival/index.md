# Survival: a practical pathway

Estimate time to an event when some records are right-censored. Define the start time, event and prediction horizon before modeling.

**Path:** Define the decision → Prepare data → Fit a baseline → Evaluate → Inspect → Act and monitor.

## 1\. Define the data and split

The synthetic example supplies `(times, events)`, with 1 for an observed event and 0 for right-censoring. It fits 2,000 rows. The remaining 1,000 rows are held out from this introductory fit. Pruning is disabled here to illustrate the fitted leaf curves. The curve is a leaf estimate, not a guaranteed individual outcome.

## 2\. Fit and inspect a small example

This complete example uses the package and Python standard library. It uses synthetic data and a fixed seed. Run it from a Python environment with `mltreelib` installed.

```
from random import Random
from mltreelib import ModelTree

rng = Random(45)
x = [[rng.uniform(0, 1)] for _ in range(3000)]
event_time = [rng.expovariate(.03 + .07 * r[0]) for r in x]
censor_time = [rng.uniform(5, 40) for _ in x]
times = [min(t, c) for t, c in zip(event_time, censor_time)]
events = [float(t <= c) for t, c in zip(event_time, censor_time)]
model = ModelTree(leaf_family="survival", max_depth=1, min_rows=20, prune="none")
model.fit(x[:2000], (times[:2000], events[:2000]), col_names=["risk_signal"])
grid = [float(i) for i in range(0, 41)]
curves = model.predict([[.1], [.9]], kind="survival", times=grid)
print("survival at day 20:", [round(curve[20], 3) for curve in curves])
```

## 3\. See the result

The figure comes from [`examples/modeling_pathways.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/modeling_pathways.py>). Rendering the figures requires matplotlib. The example is a teaching illustration, not a benchmark.

## 4\. Evaluate before deciding

The figure shows predicted survival curves, not held-out validation. Evaluate on independent data with censoring-aware discrimination and calibration at relevant horizons. Use a split that respects calendar time and repeated entities when appropriate.

## 5\. Interpret and monitor

Kaplan–Meier leaf estimates rely on assumptions about censoring within the relevant population. Curves become less reliable when few subjects remain at risk. The hazard output is cumulative hazard, not an instantaneous event probability.

## 6\. Evaluate with censoring on a held-out set

The [survival and uplift evaluation script](<https://github.com/jkapila/mltreelib/blob/main/examples/survival_uplift_evaluation.py>) simulates 5,000 people. It fits 3,000 and evaluates 2,000 independent records, including 1,391 observed events and 609 censored observations. A global survival model provides the baseline.

```
uv run python examples/survival_uplift_evaluation.py
```

| Horizon | Tree IPCW Brier | Global baseline IPCW Brier |
| ---: | ---: | ---: |
| 10 | 0.2417 | 0.2509 |
| 20 | 0.2000 | 0.2102 |
| 30 | 0.1226 | 0.1316 |

The Brier score compares predicted survival probabilities with event outcomes at a fixed horizon. Inverse probability of censoring weights (IPCW) account for records censored before that horizon. Lower is better. This simulation uses the **known** censoring distribution. Real data requires a justified censoring model estimated without using the test outcomes for model selection, adequate follow-up, and positive censoring survival at each evaluated horizon. Conditional censoring requires appropriate conditional weights; a single global curve can be wrong.

The script also compares predictions with the known synthetic survival probability. Real individual survival probabilities are not observed. Check discrimination and horizon-specific calibration on real held-out data, with uncertainty and risk-set sizes. This example implements the evaluation calculation locally; it does not claim an IPCW metric API exists in the package. See [Graf et al. (1999)](<https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18%3C2529::AID-SIM274%3E3.0.CO;2-5>).

## 7\. Real data: FLCHAIN survival

The `survival` package's [FLCHAIN cohort](<https://cran.r-project.org/web/packages/survival/survival.pdf>) contains 7,874 people and right-censored follow-up. `futime` is follow-up in days and `death=1` records a death. We use age, sex, kappa, lambda and creatinine as predictors. We exclude `chapter`, which describes cause of death and is unavailable at prediction time. Creatinine is missing for 952 of the 5,500 training records and 398 of the 2,374 test records; values enter the tree without imputation.

Download [FLCHAIN CSV from Rdatasets](<https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv>) to `datasets/external/real_eval/flchain.csv`. The checked file SHA-256 is `a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563`. Run the shared [real-data evaluation script](<https://github.com/jkapila/mltreelib/blob/main/examples/real_survival_uplift.py>) after also downloading the Hillstrom file described in the [uplift guide](<https://jkapila.github.io/mltreelib/guides/uplift/index.md>).

```
uv run python examples/real_survival_uplift.py
```

| Horizon, days | Test at risk | Censoring survival, training estimate | Tree IPCW Brier | Global baseline |
| ---: | ---: | ---: | ---: | ---: |
| 365 | 2,270 | 0.9930 | 0.0329 | 0.0343 |
| 1,095 | 2,144 | 0.9834 | 0.0671 | 0.0731 |
| 1,825 | 2,043 | 0.9737 | 0.0884 | 0.1026 |

The example uses a seeded random split and estimates the marginal censoring distribution from training data only. A marginal censoring model assumes censoring is independent of the outcome in the relevant population. This observational cohort and one split do not establish clinical usefulness. Check conditional censoring, calibration, discrimination and uncertainty with the intended deployment population before interpreting the comparison.

## Next steps and references

[Survival trees](<https://jkapila.github.io/mltreelib/examples/survival-trees/index.md>) · [Kaplan and Meier (1958)](<https://doi.org/10.1080/01621459.1958.10501452>)

Continue with [data preparation](<https://jkapila.github.io/mltreelib/guides/data/index.md>), [evaluation and monitoring](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>), or [model extensions](<https://jkapila.github.io/mltreelib/guides/advanced-modeling/index.md>). The [practical modeling pathways](<https://jkapila.github.io/mltreelib/guides/index.md>) show the other tasks. For broader coverage, see the [documentation audit](<https://github.com/jkapila/mltreelib/tree/main/references/research>).
