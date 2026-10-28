# mltreelib

**Tree-based machine learning with a Rust core.** A data and feature layer that treats missing data as first-class, an adjustable benchmark generator with known ground truth, and — next — tree learners that handle categoricals and missing values natively. The research bet: the signal hidden in categorical, grouped and hierarchical structure is what flat learners lose, and it can be *measured*.

- Rust core (`mltreelib-core`), thin Python shell (PyO3); R planned
- Scikit-learn-shaped API — without the scikit-learn dependency
- NA **and NaN** handled end-to-end: missing in → missing out; imputation opt-in
- 51 TabArena-v0.1 benchmark datasets on board, plus a generator whose knobs control how much signal lives in group and hierarchy structure

```
uv sync                                # venv + Rust extension (maturin)
uv run pytest tests/ -q                # behavior tests
uv run python examples/data_layer_demo.py   # see everything on real data
```

The research program, roadmap and contribution rules live in the repository (`README.md`, `AGENTS.md`).

> [!TIP]
>
> **Try it right now**
>
> `uv run python examples/data_layer_demo.py churn` prints ingest, missingness, encodings, grouped/trended/windowed/decomposed features and the generator's ground truth — all on a real dataset.
