# mltreelib

**Tree-based machine learning with a Rust core.** A data &amp; feature layer that treats missing data as first-class, an adjustable benchmark generator with known ground truth, and a tree learner that handles categoricals and missing values natively — with rpart-faithful printing, SVG diagrams, and LightGBM-style monotone &amp; interaction constraints.

- Rust core (`mltreelib-core`), thin Python shell (PyO3)
- Scikit-learn-shaped API — without the scikit-learn dependency
- NA **and NaN** handled end-to-end: missing in → missing out; imputation opt-in
- 51 TabArena-v0.1 benchmark datasets on board, plus a generator whose knobs control how much signal lives in group and hierarchy structure

```
uv sync                                          # venv + Rust extension (maturin)
uv run python examples/quickstart.py             # fit, print, visualize
uv run python examples/churn_demo.py             # end-to-end on real data
```

| I want to… | go to |
| --- | --- |
| install it | [Installation](<https://jkapila.github.io/mltreelib/installation/index.md>) |
| see it work in 30 seconds | [Quickstart](<https://jkapila.github.io/mltreelib/quickstart/index.md>) |
| solve a real problem | [Examples — end-to-end walkthrough](<https://jkapila.github.io/mltreelib/examples/walkthrough/index.md>) |
| wrangle data &amp; encodings | [Examples — Data Processing](<https://jkapila.github.io/mltreelib/examples/data-processing/index.md>) |
| know every knob | [Examples — Model building &amp; predictions](<https://jkapila.github.io/mltreelib/examples/model-building/index.md>) |
| read the numbers | [Examples — Inspection](<https://jkapila.github.io/mltreelib/examples/inspection/index.md>) |
| draw the tree | [Examples — Visualization](<https://jkapila.github.io/mltreelib/examples/visualization/index.md>) |
| see how it compares | [Benchmarks — ELO](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>) |
| look up every parameter | [API reference](<https://jkapila.github.io/mltreelib/reference/model/index.md>) |

> [!TIP]
>
> **Try it right now**
>
> `uv run python examples/quickstart.py` fits a tree, prints it in rpart format and writes an SVG diagram. The end-to-end walkthrough then does the same on 10,000 real bank customers.

The research program, roadmap and contribution rules live in the repository (`README.md`, `AGENTS.md`).
