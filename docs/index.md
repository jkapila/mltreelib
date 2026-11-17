# mltreelib

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mltreelib-logo-dark.svg">
    <img src="assets/mltreelib-logo.svg" alt="mltreelib logo" width="320">
  </picture>
</p>

**Tree-based machine learning with a Rust core.** A data & feature layer that
treats missing data as first-class, an adjustable benchmark generator with known
ground truth, and a tree learner that handles categoricals and missing values
natively — with rpart-faithful printing, SVG diagrams, and LightGBM-style
monotone & interaction constraints.

- Rust core (`mltreelib-core`), thin Python shell (PyO3)
- Scikit-learn-shaped API — without the scikit-learn dependency
- NA **and NaN** handled end-to-end: missing in → missing out; imputation opt-in
- 51 TabArena-v0.1 benchmark datasets on board, plus a generator whose knobs
  control how much signal lives in group and hierarchy structure

```bash
uv sync                                          # venv + Rust extension (maturin)
uv run python examples/quickstart.py             # fit, print, visualize
uv run python examples/churn_demo.py             # end-to-end on real data
```

| I want to… | go to |
|---|---|
| install it | [Installation](installation.md) |
| see it work in 30 seconds | [Quickstart](quickstart.md) |
| solve a real problem | [Examples — end-to-end walkthrough](examples/walkthrough.md) |
| wrangle data & encodings | [Examples — Data Processing](examples/data-processing.md) |
| know every knob | [Examples — Model building & predictions](examples/model-building.md) |
| read the numbers | [Examples — Inspection](examples/inspection.md) |
| draw the tree | [Examples — Visualization](examples/visualization.md) |
| see how it compares | [Benchmarks — ELO](benchmarks/elo.md) |
| read technical deep dives & blogs | [Blog & Technical Insights](blog/index.md) |
| look up every parameter | [API reference](reference/model.md) |

---

## Featured Technical Articles

Explore our in-depth architectural and commercial case studies:

1. **[Why India's Digital Lending Boom Needs Glass-Box Trees Over Black-Box Boosters](blog/why-credit-scoring-needs-glass-box-trees.md)**  
   *Regulatory Compliance & Fintech*: Monotonic shape constraints, WOE binning, and auditable credit scorecards.
2. **[Zero-Imputation Predictive Maintenance for Industrial Sensor Feeds](blog/zero-imputation-predictive-maintenance.md)**  
   *Industrie 4.0 & Sensor Telemetry*: Why synthetic imputation corrupts failure dynamics, and how native surrogate routing and linear leaves solve sensor blackouts.
3. **[Uplift Modeling from Scratch: Maximizing ROI with Honest Causal Trees](blog/uplift-modeling-with-causal-trees.md)**  
   *Commercial Decision Optimization*: Why standard response modeling fails, and how sample-split honest trees isolate incremental treatment uplift.
4. **[Why We Rebuilt Quinlan's Cubist and C5.0 in Rust](blog/modernizing-quinlan-classics-in-rust.md)**  
   *Core Systems ML*: Modernizing 1990s C rule engines with memory-safe SIMD Cholesky factorization in Rust.
5. **[Beyond Mean Squared Error: Distributional Decision Trees for Heavy Tails, Insurance, and Censored Data](blog/beyond-mse-distributional-trees.md)**  
   *Actuarial & Risk Modeling*: Tweedie compound Poisson-Gamma claims, Tobit censored log-likelihood, and heteroscedastic volatility regimes.
6. **[Breaking the Step Function: Why Model Trees Beat Gradient Boosters on Continuous Physics](blog/breaking-step-functions-model-trees-vs-gbdt.md)**  
   *Systems ML & Physics-Informed ML*: Overcoming orthogonal step staircases on continuous physical manifolds with piecewise linear leaves and hierarchical smoothing.

!!! tip "Try it right now"
    `uv run python examples/quickstart.py` fits a tree, prints it in rpart
    format and writes an SVG diagram. The end-to-end walkthrough then does the same on
    10,000 real bank customers.

The research program, roadmap and contribution rules live in the repository
(`README.md`, `AGENTS.md`).
