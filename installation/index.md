# Installation

mltreelib is a Python package with a Rust core — one wheel, no transitive dependencies.

## From PyPI

```
pip install mltreelib
```

## With uv

```
uv add mltreelib
```

## From source

Building from source needs a Rust toolchain ([rustup](<https://rustup.rs>)) and [maturin](<https://www.maturin.rs>):

```
git clone https://github.com/jkapila/mltreelib
cd mltreelib
uv run maturin develop        # builds the Rust core into your environment
```

## Verify the install

```
uv run python -c "from mltreelib import ModelTree; print(ModelTree())"
```

An unfitted tree refuses to print its structure (by design); the call should raise `ValueError: ModelTree is not fitted`. Fit anything and `print(model)` shows the rpart-style tree.

## Plotting (optional extra)

Charts and diagrams ride on the standard rendering wheels — install them only when you build figures:

```
pip install mltreelib[plotting]     # matplotlib + pygraphviz + seaborn
# or
uv sync --extra plotting
```

`mltreelib.plots` (metric charts) and `ModelTree.to_dot()` + pygraphviz (tree diagrams) are the two consumers. The core library never imports them.

## What you get

|  |  |
| --- | --- |
| `ModelTree` | the tree: fit / predict, rpart-style text, SVG diagram, constraints |
| 18 encoders | OneHot, Hashing, Target, CatBoost, WoE, James-Stein, Count, Leave-One-Out, M-Estimator, Base-N, Sum, Quantile, TF-IDF, Embedding, … |
| scalers &amp; builders | `VariableScaler`, `GroupedScaler`, `DatasetBuilder`, `IndexMapper` |
| `Metrics` | regression / classification / regression-by-group reports |
| dataset helpers | `list_datasets`, `load_dataset`, `summarize`, `generate`, calendar / lag / window transforms |

Numerics run in the Rust core; the data layer is the same code the [benchmarks](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>) run on.
