# mltreelib

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
| solve a real problem | [Examples — Churn](examples/churn.md) |
| wrangle data & encodings | [Examples — Data Processing](examples/data-processing.md) |
| read & draw the tree | [Examples — Model build & Tree Visualization](examples/model-build-viz.md) |
| see how it compares | [Benchmarks — ELO](benchmarks/elo.md) |
| look up every parameter | [API reference — Data & Model](reference/data-model.md) |

!!! tip "Try it right now"
    `uv run python examples/quickstart.py` fits a tree, prints it in rpart
    format and writes an SVG diagram. The churn example then does the same on
    10,000 real bank customers.

The research program, roadmap and contribution rules live in the repository
(`README.md`, `AGENTS.md`).
