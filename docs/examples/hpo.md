# Hyperparameter search (Optuna)

The full `ModelTree` growth surface is a search space — and Optuna is the
wheel we borrow for it. `examples/hpo_optuna.py` tunes depth, leaf size,
ridge, leaf kind, criterion, the split grid, categorical search, the
imbalance knobs and pruning on a seeded TPE study:

```bash
uv run --with pandas --with optuna python examples/hpo_optuna.py 40
```

The objective is plain — fit, score, return RMSE. Every suggestion maps
1:1 to a `ModelTree` keyword:

```python
params = dict(
    max_depth=trial.suggest_int("max_depth", 1, 8),
    min_rows=trial.suggest_int("min_rows", 5, 200, log=True),
    l2=trial.suggest_float("l2", 1e-3, 10.0, log=True),
    leaf=trial.suggest_categorical("leaf", ["linear", "mean"]),
    leaf_cols=trial.suggest_categorical("leaf_cols", ["all", "path"]),
    gain=trial.suggest_categorical("gain", ["sse", "sdr"]),
    split_precision=trial.suggest_categorical("split_precision", [None, 0.1, 0.01]),
    cat_search=trial.suggest_categorical("cat_search", ["cumulative", "single_vs_rest", "both"]),
    alpha=trial.suggest_float("alpha", 0.0, 0.2),
    imbalance_penalty=trial.suggest_float("imbalance_penalty", 1e-3, 100.0, log=True),
    prune=trial.suggest_categorical("prune", ["none", "bic", "mdl"]),
)
m = ModelTree(**params)
m.fit(xtr, ytr, dtypes=dtypes)
return Metrics().regression(yte, m.predict(xte))["rmse"]
```

## A real study (airfoil_self_noise, 40 seeded TPE trials)

```text
trials: 40   best rmse: 1.848
best settings (the elo_round SETTINGS shape):
    max_depth=8,
    min_rows=6,
    l2=5.8545958517512275,
    leaf='linear',
    leaf_cols='all',
    gain='sse',
    split_precision=None,
    cat_search='both',
    alpha=0.030185289276873507,
    imbalance_penalty=0.05567092885866246,
    prune='bic',
trial #38  (seeded TPE, airfoil 80/20)
```

Two findings worth keeping: TPE independently rejected the split grid
(`split_precision=None`) for this fit, and it picked `prune='bic'` with a
small imbalance penalty — cost pruning and balance tuning, chosen by the
search rather than by hand.

**The winner prints in the exact `SETTINGS` dict shape the Elo rounds
record** — paste it into `examples/elo_round.py` to score a tuned flavor on
the full field. That is the intended loop: search per-dataset, then let the
round measure what the settings are actually worth across 18 datasets
([ELO](../benchmarks/elo.md) — *Running a round*).

**Examples — next: [Metrics & plots](metrics-plots.md)**
