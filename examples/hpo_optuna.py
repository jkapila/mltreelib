"""Hyperparameter search over the full ModelTree surface with Optuna.

Tunes the growth knobs (depth, leaf size, ridge, criterion, split grid,
cat search, imbalance, pruning) on airfoil_self_noise with a deterministic
seeded TPE study. The winning settings print as the exact `SETTINGS` dict
shape the Elo rounds record (examples/elo_round.py) — paste them there to
score a tuned flavor.

    uv run --with pandas --with optuna python examples/hpo_optuna.py [n_trials]
"""

import sys

import optuna
import pandas as pd

from mltreelib import Metrics, ModelTree

df = pd.read_parquet("datasets/airfoil_self_noise.parquet")
y = df["scaled-sound-pressure"].astype(float).tolist()
cols = [c for c in df.columns if c != "scaled-sound-pressure"]
X, dtypes = [], []
for col in cols:
    raw = [None if pd.isna(v) else v for v in df[col]]
    if str(df[col].dtype) == "category":
        dtypes.append("categorical")
        X.append([None if v is None else float(str(v)) for v in raw])
    else:
        dtypes.append("numeric")
        X.append([None if v is None else float(v) for v in raw])
X = [list(r) for r in zip(*X)]

n_tr = int(0.8 * len(X))
xtr, ytr, xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def objective(trial: optuna.Trial) -> float:
    params = dict(
        max_depth=trial.suggest_int("max_depth", 1, 8),
        min_rows=trial.suggest_int("min_rows", 5, 200, log=True),
        l2=trial.suggest_float("l2", 1e-3, 10.0, log=True),
        leaf=trial.suggest_categorical("leaf", ["linear", "mean"]),
        leaf_cols=trial.suggest_categorical("leaf_cols", ["all", "path"]),
        gain=trial.suggest_categorical("gain", ["sse", "sdr"]),
        split_precision=trial.suggest_categorical(
            "split_precision", [None, 0.1, 0.01]
        ),
        cat_search=trial.suggest_categorical(
            "cat_search", ["cumulative", "single_vs_rest", "both"]
        ),
        alpha=trial.suggest_float("alpha", 0.0, 0.2),
        imbalance_penalty=trial.suggest_float(
            "imbalance_penalty", 1e-3, 100.0, log=True
        ),
        prune=trial.suggest_categorical("prune", ["none", "bic", "mdl"]),
    )
    m = ModelTree(**params)
    m.fit(xtr, ytr, dtypes=dtypes)
    return Metrics().regression(yte, m.predict(xte))["rmse"]


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"trials: {len(study.trials)}   best rmse: {study.best_value:.3f}")
    print("best settings (the elo_round SETTINGS shape):")
    for k, v in study.best_params.items():
        print(f"    {k}={v!r},")
    t = study.best_trial
    print(f"trial #{t.number}  (seeded TPE, airfoil 80/20)")


if __name__ == "__main__":
    main()
