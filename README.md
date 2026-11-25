# mltreelib — Machine Learning Tree-Based Library

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mltreelib-logo-dark.svg">
    <img src="docs/assets/mltreelib-logo.svg" alt="mltreelib logo" width="320">
  </picture>
</p>

[![PyPI](https://img.shields.io/pypi/v/mltreelib.svg)](https://pypi.org/project/mltreelib/)
[![CI](https://github.com/jkapila/mltreelib/actions/workflows/test.yaml/badge.svg)](https://github.com/jkapila/mltreelib/actions/workflows/test.yaml)
[![Docs](https://github.com/jkapila/mltreelib/actions/workflows/deploy.yaml/badge.svg)](https://jkapila.github.io/mltreelib/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)](pyproject.toml)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg?logo=rust)](crates/mltreelib-core/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Zero Sklearn](https://img.shields.io/badge/dependencies-zero--sklearn-brightgreen)](#)
[![Missing Data](https://img.shields.io/badge/missing--data-first--class-blueviolet)](#)

**mltreelib is a tree-based machine learning library with a Rust core and a Python API**, covering regression, classification, survival, uplift, clustering, anomaly detection, components and forecasting through one consistent surface.

> [!WARNING]
> **Early development:** mltreelib is at an early stage. APIs may change or break between releases. For any new feature, please [raise an issue first](https://github.com/jkapila/mltreelib/issues) so we can discuss scope before code is written.

## Why mltreelib

- **One API across tasks.** `ModelTree`, `ClusterModel`, `AnomalyModel`, `ComponentModel` and `RuleModel` share the same `fit` / `predict` shape.
- **One explainable tree.** Build a single tree, grow and prune it, and choose the leaf model (linear, mean, distributional, survival). The result is easy to explain to a business audience.
- **Missing data is handled natively.** Missing values (`None`, `NaN`) stay in your data: the library does not impute them, and the tree routes them with surrogate splits and a learned missing direction.
- **Statistical summaries for troubleshooting.** Each linear leaf reports coefficients, standard errors, t and p values, and R² in the style of `summary(lm)`.
- **No scikit-learn dependency.** The runtime is a Rust core with a thin Python layer and a small dependency footprint.

## Installation

Install pre-compiled native binary wheels directly from PyPI:

```sh
pip install mltreelib
```

For development or compiling from source (requires Rust toolchain and [uv](https://docs.astral.sh/uv/)):

```sh
git clone https://github.com/jkapila/mltreelib.git
cd mltreelib
uv sync
uv run maturin develop --release
```

See the [changelog](CHANGELOG.md) for release history and the [release guide](RELEASING.md) for publishing steps.

### R ModelTree

From a v0.4.15 checkout, use R, a C compiler, and Cargo to build and install the source package. The source package includes the Rust core. Cargo needs access to its dependencies.

```sh
r/build-r-package.sh dist
R CMD INSTALL dist/mltreelib_0.4.15.tar.gz
Rscript --vanilla r/airquality_model_tree.R
```

The package exposes ModelTree fitting and prediction. The source tarball was checked on Ubuntu 24.04 with R 4.3.3. [Holdout results and source checks](references/research/2026-11-21-r-airquality-results.md)

### Example suites and agent skill

Follow the [example suites](https://jkapila.github.io/mltreelib/examples/) for runnable walkthroughs. Start with the [quickstart](https://jkapila.github.io/mltreelib/quickstart/) to fit and inspect your first model tree.

Install the mltreelib skill for an agent that supports the Skills CLI. The skill includes recipes for selecting, fitting, evaluating, and saving models. Install the Python package separately to run the recipes. [Skill source](https://github.com/jkapila/mltreelib/tree/main/skills/mltreelib)

```sh
npx skills add jkapila/mltreelib --skill mltreelib
```

## Quickstart

Fit a model tree with piecewise linear leaves on the Concrete Compressive Strength benchmark (1,030 mixtures, 8 ingredient features):

```python
from mltreelib import Metrics, ModelTree, load_dataset

# 1. Load benchmark dataset (1,030 mixtures, 8 features)
data = load_dataset("concrete_compressive_strength")
target = "ConcreteCompressiveStrength"
feature_names = [col for col in data["columns"].keys() if col != target]
n_rows = data["n_rows"]

X = [[data["columns"][c][i] for c in feature_names] for i in range(n_rows)]
y = data["columns"][target]

# 80/20 train/test split
split_idx = int(n_rows * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Fit mltreelib ModelTree (piecewise linear ridge leaves, depth=3)
model_tree = ModelTree(leaf="linear", max_depth=3, min_rows=20, l2=1e-3)
model_tree.fit(X_train, y_train, col_names=feature_names)
pred_mt = model_tree.predict(X_test)

# Measure holdout metrics
metrics = Metrics()
m_mt = metrics.regression(y_test, pred_mt)
print(f"ModelTree (Linear): RMSE = {m_mt['rmse']:.2f} MPa, R2 = {m_mt['r2']:.3f}, MAE = {m_mt['mae']:.2f} MPa")
```

Full walkthrough, tree printout and comparison with a constant-leaf tree: [Quickstart](https://jkapila.github.io/mltreelib/quickstart/).

## What is in the box

| Task | Entry point | Docs |
|---|---|---|
| Regression | `ModelTree(leaf="linear")` | [Regression](https://jkapila.github.io/mltreelib/guides/regression/) |
| Classification | `ModelTree` | [Classification](https://jkapila.github.io/mltreelib/guides/classification/) |
| Survival and competing risks | `ModelTree` | [Survival](https://jkapila.github.io/mltreelib/guides/survival/) |
| Uplift and causal effects | `ModelTree(gain="causal")` | [Uplift](https://jkapila.github.io/mltreelib/guides/uplift/) |
| Clustering | `ClusterModel(kind=...)` | [Clustering](https://jkapila.github.io/mltreelib/guides/clustering/) |
| Anomaly detection | `AnomalyModel(kind=...)`, `GritBot` | [Anomaly detection](https://jkapila.github.io/mltreelib/guides/anomaly/) |
| Components and rotations | `ComponentModel` | [Components](https://jkapila.github.io/mltreelib/concepts/components-and-rotations/) |
| Rules and inspection | `RuleModel`, `ModelTree.extract_rules()` | [Rules and stumps](https://jkapila.github.io/mltreelib/guides/model-inspection-and-rules/) |
| Forecasting features | `lag_by_days`, `windowed_transform` | [Time series](https://jkapila.github.io/mltreelib/concepts/timeseries-forecasting/) |
| Solvers and optimizers | `mltreelib.solvers.fit`, `mltreelib.solvers.optimize` | [Solvers](https://jkapila.github.io/mltreelib/concepts/solvers-and-optimizers/) |
| Data preparation and synthetic data | `load_dataset`, `generate`, encoders | [Prepare data](https://jkapila.github.io/mltreelib/guides/data/) |
| Metrics and diagnostics | `Metrics` | [Evaluate](https://jkapila.github.io/mltreelib/guides/evaluation/) |

Notes:
- Model save and restore (`save`, `load`, JSON, bytes) is documented for `ModelTree` in the [serving contract](https://jkapila.github.io/mltreelib/guides/serving-contract/).
- Custom Python split criteria (user callables) are not yet documented on the docs site.
- `TrieKNN` is listed in the [examples index](https://jkapila.github.io/mltreelib/examples/) only.

## Documentation

- [Concepts](https://jkapila.github.io/mltreelib/concepts/core-tree-architecture/): how the tree, splits, leaves and missing-data routing work.
- [Blog](https://jkapila.github.io/mltreelib/blog/): technical case studies.
- [Guides](https://jkapila.github.io/mltreelib/guides/): task-by-task workflows on real data.
- [Examples](https://jkapila.github.io/mltreelib/examples/): runnable scripts, one per topic.
- [Benchmarks](https://jkapila.github.io/mltreelib/benchmarks/elo/): arena and method studies, with the datasets used.
- [API reference](https://jkapila.github.io/mltreelib/reference/model/): every parameter and method.

Early development: APIs may change between releases. Raise an issue before starting a new feature.

## Roadmap

A short outlook, not a commitment. Details and status are in [specs/README.md](specs/README.md).

- **Data tools:** splits and data profiling.
- **Metrics and drift monitoring:** planned as part of a split into data, core and metrics areas.
- **Pruning and defaults:** scale-free pruning and better defaults.
- **Benchmarks:** broader suites with confidence intervals.
- **Survival and uplift:** a real-data suite for both.
- **Ensembles:** forest, boosting and meta estimators, after the single tree is settled.

## Contributing

- **Raise an issue first** for any new feature, so scope is agreed before code is written: [issues](https://github.com/jkapila/mltreelib/issues).
- **Tests** live in `tests/` (Python behavior tests, pytest) and `crates/mltreelib-core/tests/` (Rust).
- **Build and test:**

```sh
uv sync                              # venv + Rust extension (maturin)
uv run maturin develop --release     # rebuild the extension with release optimizations
cargo test                           # Rust core tests
uv run pytest tests/ -q              # Python behavior tests
```

For contribution rules, see [contributing/](contributing/) and [AGENTS.md](AGENTS.md).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
