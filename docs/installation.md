# Installation

mltreelib is a Python package with a Rust core — one wheel, no transitive
dependencies.

## From PyPI

```bash
pip install mltreelib
```

## With uv

```bash
uv add mltreelib
```

## From source

Building from source needs a Rust toolchain ([rustup](https://rustup.rs)) and
[maturin](https://www.maturin.rs):

```bash
git clone https://github.com/jkapila/mltreelib
cd mltreelib
uv run maturin develop        # builds the Rust core into your environment
```

## Verify the install

```bash
uv run python -c "from mltreelib import ModelTree; print(ModelTree())"
```

An unfitted tree refuses to print its structure (by design); the call should
raise `ValueError: ModelTree is not fitted`. Fit anything and `print(model)`
shows the rpart-style tree.

## What you get

| | |
|---|---|
| `ModelTree` | the tree: fit / predict, rpart-style text, SVG diagram, constraints |
| 18 encoders | OneHot, Hashing, Target, CatBoost, WoE, James-Stein, Count, Leave-One-Out, M-Estimator, Base-N, Sum, Quantile, TF-IDF, Embedding, … |
| scalers & builders | `VariableScaler`, `GroupedScaler`, `DatasetBuilder`, `IndexMapper` |
| `Metrics` | regression / classification / regression-by-group reports |
| dataset helpers | `list_datasets`, `load_dataset`, `summarize`, `generate`, calendar / lag / window transforms |

Numerics run in the Rust core; the data layer is the same code the
[benchmarks](benchmarks/elo.md) run on.
