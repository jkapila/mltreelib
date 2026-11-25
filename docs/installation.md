# Installation

Install mltreelib as a Python wheel with a Rust core. Model fitting and inference require no additional Python packages.

!!! warning "Early development"
    mltreelib is at an early stage. APIs may change or break between releases.
    For any new feature, please [raise an issue first](https://github.com/jkapila/mltreelib/issues)
    so we can discuss scope before code is written.

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
uv sync --group dev
uv run maturin develop --release
```

## Verify the install

```bash
python -c "import mltreelib; print(mltreelib.__version__)"
```

If you use uv, run the command with `uv run python`. Continue to the [quickstart](quickstart.md) to fit and inspect your first tree.

## Dataset loading (optional extra)

Install the dataset extra to read the package's Parquet datasets.

```bash
pip install "mltreelib[dataset]"
# or
uv add "mltreelib[dataset]"
```

## Plotting (optional extra)

Charts and diagrams ride on the standard rendering wheels — install them
only when you build figures:

```bash
pip install "mltreelib[plotting]"   # matplotlib + pygraphviz + seaborn
# or
uv sync --extra plotting
```

`mltreelib.plots` (metric charts) and `ModelTree.to_dot()` + pygraphviz
(tree diagrams) are the two consumers. The core library never imports them.

## What you get

| | |
|---|---|
| `ModelTree` | the tree: fit / predict, rpart-style text, SVG diagram, constraints |
| 14 encoders | OneHot, Hashing, Target, CatBoost, WoE, James-Stein, Count, Leave-One-Out, M-Estimator, Base-N, Sum, Quantile, TF-IDF, Embedding |
| scalers & builders | `VariableScaler`, `GroupedScaler`, `DatasetBuilder`, `IndexMapper` |
| `Metrics` | regression / classification / regression-by-group reports |
| dataset helpers | `list_datasets`, `load_dataset`, `summarize`, `generate`, calendar / lag / window transforms |

Numerics run in the Rust core; the data layer is the same code the
[benchmarks](benchmarks/elo.md) run on.
