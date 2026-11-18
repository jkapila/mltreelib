# Uplift modeling: a practical pathway

Estimate how an intervention changes outcomes, rather than who is likely to have an outcome. The example uses randomly assigned synthetic treatment.

**Path:** Define the decision → Prepare data → Fit a baseline → Evaluate → Inspect → Act and monitor.

## 1\. Define the data and split

Each row has a pre-treatment feature and a tuple `(outcome, treatment)`, where treatment is 0 or 1. The model fits 2,000 rows and scores 1,000 held-out rows. A causal interpretation requires consistency, overlap, appropriate treatment assignment and no relevant interference.

## 2\. Fit and inspect a small example

This complete example uses the package and Python standard library. It uses synthetic data and a fixed seed. Run it from a Python environment with `mltreelib` installed.

```
from random import Random
from mltreelib import ModelTree

rng = Random(46)
x = [[rng.uniform(-1, 1)] for _ in range(3000)]
treatment = [float(rng.random() < .5) for _ in x]
outcome = [5 + (2 if r[0] >= 0 else -1) * w + rng.gauss(0, 1) for r, w in zip(x, treatment)]
model = ModelTree(gain="causal", leaf="mean", max_depth=1, min_rows=40)
model.fit(x[:2000], (outcome[:2000], treatment[:2000]), col_names=["segment_signal"])
scores = model.predict(x[2000:], kind="uplift")
print("uplift held-out score range:", round(min(scores), 3), round(max(scores), 3))
```

## 3\. See the result

The figure comes from [`examples/modeling_pathways.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/modeling_pathways.py>). Rendering the figures requires matplotlib. The example is a teaching illustration, not a benchmark.

## 4\. Evaluate before deciding

The plot shows predictions, not proof of policy value. On an independent randomized holdout, compare treatment and control outcomes by score band, report arm sizes and uncertainty, and account for treatment cost. Individual causal effects are not observed in real data.

## 5\. Interpret and monitor

A leaf comparison alone does not remove confounding. Honesty does not fix non-random treatment assignment. Check treatment and control support before applying a rule to a business decision.

## 6\. Evaluate a targeting policy

The [survival and uplift evaluation script](<https://github.com/jkapila/mltreelib/blob/main/examples/survival_uplift_evaluation.py>) simulates a randomized trial. It uses 3,000 training, 1,000 validation and 2,000 independent test observations. It selects a threshold on validation data and evaluates the fixed policy on test data. The example assigns a treatment cost of 0.5 outcome units per treated person.

```
uv run python examples/survival_uplift_evaluation.py
```

| Test policy | Selected | Treated / control among selected | Estimated net outcome per person |
| --- | ---: | ---: | ---: |
| Selected policy | 967 | 471 / 496 | 0.7072 |
| Treat all | 2,000 | 972 / 1,028 | \-0.0478 |
| Treat none | 0 | 0 / 0 | 0 |

Within the selected subgroup, the script compares observed treated and control outcomes. It subtracts the cost and multiplies by the selected fraction to report value per person in the whole test population. The selected policy's approximate 95% interval is \[0.6438, 0.7706\]. It uses a normal approximation conditional on the selected fraction; it does not include model-selection uncertainty. Randomized assignment and independent test data support this comparison. Check arm counts and use a suitable uncertainty method for the actual design.

Synthetic truth is available here: the selected policy's true net value is 0.7252 per person and PEHE is 0.2347. Real records do not reveal both potential outcomes, so individual-effect error and PEHE cannot be calculated directly from ordinary real-world outcomes. These calculations are example code, not a claim that the package already exposes uplift evaluation metrics.

## 7\. Real randomized trial: Hillstrom email

The [Hillstrom email experiment](<https://www.tensorflow.org/datasets/catalog/hillstrom>) randomly assigned 64,000 customers to a men's email, women's email or no email. This example compares **women's email versus no email**, using a website visit within two weeks as the outcome. It excludes the men's arm and uses only pre-treatment features. A visit is not a purchase or profit.

The publisher's original download endpoint failed HTTPS certificate validation in our environment. We used this [public mirror](<https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/main/data/raw/hillstrom.csv>), SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`. The mirror matches published row and arm counts; independent byte equivalence with the original is not established. Save the CSV as `datasets/external/real_eval/hillstrom.csv`. It is excluded from Git. The shared [script](<https://github.com/jkapila/mltreelib/blob/main/examples/real_survival_uplift.py>) also requires the FLCHAIN CSV described in the [survival guide](<https://jkapila.github.io/mltreelib/guides/survival/index.md>).

The seeded split contains 25,615 training, 8,539 validation and 8,539 test rows. An honest causal tree ranks customers. Validation selects the policy threshold, and the randomized test set estimates incremental visits. This example assumes zero treatment cost because campaign costs are not in the data.

| Test policy | Selected | Treated / control | Incremental visits per eligible customer | Approximate 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Validation-selected policy | 8,539 | 4,270 / 4,269 | 0.04517 | 0.03108 to 0.05927 |
| Treat all | 8,539 | 4,270 / 4,269 | 0.04517 | 0.03108 to 0.05927 |
| Treat none | 0 | — | 0 | — |

The chosen policy treats everyone. This run therefore supplies **no evidence that targeting improves on treat-all**. The interval is a normal approximation conditional on the selected population and excludes model-selection uncertainty. The data is from a historical campaign; repeat evaluation in a new experiment, include costs and use the outcome the business actually values before applying a policy.

## Next steps and references

[Causal uplift guide](<https://jkapila.github.io/mltreelib/examples/causal-uplift/index.md>) · [Athey and Imbens (2016)](<https://www.pnas.org/doi/10.1073/pnas.1510489113>)

Continue with [data preparation](<https://jkapila.github.io/mltreelib/guides/data/index.md>), [evaluation and monitoring](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>), or [model extensions](<https://jkapila.github.io/mltreelib/guides/advanced-modeling/index.md>). The [practical modeling pathways](<https://jkapila.github.io/mltreelib/guides/index.md>) show the other tasks. For broader coverage, see the [documentation audit](<https://github.com/jkapila/mltreelib/tree/main/references/research>).
