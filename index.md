# mltreelib

**Tree-based machine learning with a Rust core.** A data &amp; feature layer that treats missing data as first-class, an adjustable benchmark generator with known ground truth, and a tree learner that handles categoricals and missing values natively — with rpart-faithful printing, SVG diagrams, and LightGBM-style monotone &amp; interaction constraints.

- Rust core (`mltreelib-core`), thin Python shell (PyO3)
- Scikit-learn-shaped API — without the scikit-learn dependency
- NA **and NaN** handled end-to-end: missing in → missing out; imputation opt-in
- 51 TabArena-v0.1 benchmark datasets on board, plus a generator whose knobs control how much signal lives in group and hierarchy structure

```
uv sync                                          # venv + Rust extension (maturin)
uv run python examples/quickstart.py             # fit, print, visualize
uv run python examples/real_tree_studies.py wine  # after downloading Wine Quality
```

| I want to… | go to |
| --- | --- |
| install it | [Installation](<https://jkapila.github.io/mltreelib/installation/index.md>) |
| see it work in 30 seconds | [Quickstart](<https://jkapila.github.io/mltreelib/quickstart/index.md>) |
| choose a modeling task | [Practical modeling pathways](<https://jkapila.github.io/mltreelib/guides/index.md>) |
| solve a real problem | [Real datasets and worked studies](<https://jkapila.github.io/mltreelib/guides/real-data/index.md>) |
| change one part of a model | [Example library](<https://jkapila.github.io/mltreelib/examples/index.md>) |
| wrangle data &amp; encodings | [Examples — Data Processing](<https://jkapila.github.io/mltreelib/examples/data-processing/index.md>) |
| know every knob | [Examples — Model building &amp; predictions](<https://jkapila.github.io/mltreelib/examples/model-building/index.md>) |
| read the numbers | [Examples — Inspection](<https://jkapila.github.io/mltreelib/examples/inspection/index.md>) |
| draw the tree | [Examples — Visualization](<https://jkapila.github.io/mltreelib/examples/visualization/index.md>) |
| see how it compares | [Benchmarks — ELO](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>) |
| read technical deep dives &amp; blogs | [Blog &amp; Technical Insights](<https://jkapila.github.io/mltreelib/blog/index.md>) |
| look up every parameter | [API reference](<https://jkapila.github.io/mltreelib/reference/model/index.md>) |

---

## Featured Technical Articles

Explore our in-depth architectural and commercial case studies:

1. **[Decision Trees for Credit Scoring](<https://jkapila.github.io/mltreelib/blog/why-credit-scoring-needs-glass-box-trees/index.md>)** *Regulatory Compliance &amp; Fintech*: How tree paths, constraints and calibration support model review.
2. **[Zero-Imputation Predictive Maintenance for Industrial Sensor Feeds](<https://jkapila.github.io/mltreelib/blog/zero-imputation-predictive-maintenance/index.md>)** *Industrie 4.0 &amp; Sensor Telemetry*: A synthetic example of missing sensor values and tree routing.
3. **[Uplift Modeling with Causal Trees](<https://jkapila.github.io/mltreelib/blog/uplift-modeling-with-causal-trees/index.md>)** *Commercial Decision Optimization*: How an honest causal tree compares targeting policies on a randomized held-out trial.
4. **[Cubist Models and C5.0 Rules](<https://jkapila.github.io/mltreelib/blog/modernizing-quinlan-classics-in-rust/index.md>)** *Core Systems ML*: Linear leaves, extracted rules and the contracts still to check.
5. **[Beyond Mean Squared Error: Distributional Decision Trees for Heavy Tails, Insurance, and Censored Data](<https://jkapila.github.io/mltreelib/blog/beyond-mse-distributional-trees/index.md>)** *Actuarial &amp; Risk Modeling*: Tweedie compound Poisson-Gamma claims, Tobit censored log-likelihood, and heteroscedastic volatility regimes.
6. **[Breaking the Step Function: Model Trees and Step Functions for Continuous Regression](<https://jkapila.github.io/mltreelib/blog/breaking-step-functions-model-trees-vs-gbdt/index.md>)** *Systems ML &amp; Physics-Informed ML*: How linear leaves represent local trends and why fair comparisons need held-out data.

> [!TIP]
>
> **Try it right now**
>
> `uv run python examples/quickstart.py` fits a tree, prints it in rpart format and writes an SVG diagram. The end-to-end walkthrough then does the same on 10,000 real bank customers.

The research program, roadmap and contribution rules live in the repository (`README.md`, `AGENTS.md`).
