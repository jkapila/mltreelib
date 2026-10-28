# mltreelib — Machine Learning with Tree Based Library

Tree-based machine learning with a **Rust core** and thin **Python** (and, next, **R**) frontends. Evolved from the attempt to make the *right* kind of **Decision Trees**, after Hastie, Tibshirani, Friedman, Quinlan, Loh, Chaudhari.

## Layout

```
crates/mltreelib-core/   Rust core: dataset component (data layer + adjustable
                         benchmark generator); next: tree learners
crates/mltreelib-py/    PyO3 bindings, exposed as mltreelib._core
python/mltreelib/       Python package — scikit-learn-shaped API, no sklearn dependency
tests/                  Python behavior tests
references/             research notes, benchmarks, source material
legacy/                 previous Python/nbdev implementation (kept as reference oracle)
```

## Setup

Requires Rust and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # creates .venv and builds the Rust extension (maturin)
uv run pytest tests/ -q  # behavior tests
```

## Dataset component

The dataset component is the heart of the library: an adjustable benchmark
generator whose knobs control how much signal lives in categorical, grouped and
hierarchical structure, plus a data layer that takes any data in.

```python
from mltreelib import generate, IndexMapper, grouped_transform, summarize

g = generate(
    n_rows=10_000, n_numeric=5, cardinalities=[3, 8, 50, 200],
    hierarchy_levels=3, branching=[4, 10, 25],
    group_effect_strength=0.6, hierarchy_effect_strength=0.3,
    interaction_strength=0.2, noise=0.1, seed=42,
)
g["columns"]                 # num_*, cat_*, hier_* columns (None = NA)
g["target"]                  # regression target (labels for task="classification")
g["truth"]["variance"]       # realized variance share per component
g["truth"]["categorical_effects"]  # ground-truth level effects
```

Because the ground truth is known, every learner can be scored on *structure
recovery*, not just accuracy — including tail behavior (`tail_behavior=
"bathtub"` reproduces the DAR bathtub edges) and missingness.

## Status

- [x] Dataset component v0: `n_category` schema rule, `IndexMapper`, grouped transforms, adjustable generator (Rust + Python)
- [ ] Tree learners in Rust: CART, C4.5, then C5.0, Cubist, GritBot replications
- [ ] Comparison harness vs LightGBM / XGBoost / CatBoost / tabular foundation models
- [ ] R bindings (extendr over `mltreelib-core`)

## License

Apache-2.0
