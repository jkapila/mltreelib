"""ModelTree vs LightGBM vs CatBoost — how far are we?

Three regimes:
  1. piecewise_linear — the model tree's home turf (segmented linear signal);
  2. nonlinear — interactions and curves (the ensembles' home turf);
  3. na_signal — 20% missing x0 whose missingness itself carries the signal.

All three include NAs, so every model exercises its native missing handling.
Run:
    uv run --with lightgbm --with catboost python examples/model_tree_benchmark.py
"""

import time

import numpy as np

from mltreelib import ModelTree

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None

N, D, SEED = 2000, 8, 7


def make_regime(name, rng):
    x = rng.uniform(-1.0, 1.0, size=(N, D))
    if name == "piecewise_linear":
        z = x[:, 0]
        y = np.where(
            z < -0.3,
            3.0 * x[:, 1] + 1.0,
            np.where(z < 0.3, -2.0 * x[:, 2] + 0.5, 4.0 * x[:, 3] - 1.0),
        )
    elif name == "nonlinear":
        y = np.sin(3 * x[:, 0]) * x[:, 1] + 2.0 * x[:, 2] ** 2 + x[:, 3] * x[:, 4]
    else:  # na_signal
        y = 2.0 * x[:, 1] - x[:, 2]
    # NAs: 20% of column 0 missing in every regime.
    na = rng.uniform(size=N) < 0.2
    x[na, 0] = np.nan
    if name == "na_signal":
        y = y + 5.0 * na  # missingness is the signal
    y = y + rng.normal(scale=0.1, size=N)
    return x, y


def to_rows(x):
    """Row-major Option-style lists for our ModelTree (None = missing)."""
    return [[None if np.isnan(v) else float(v) for v in row] for row in x]


def run():
    rng = np.random.default_rng(SEED)
    models = [("ModelTree(linear)", None), ("ModelTree(path)", None)]
    if LGBMRegressor is not None:
        models.append(("LightGBM", LGBMRegressor))
    if CatBoostRegressor is not None:
        models.append(("CatBoost", CatBoostRegressor))

    print(f"{'regime':<18}{'model':<20}{'RMSE':>8}{'MAE':>8}{'fit s':>8}")
    print("-" * 62)
    for regime in ("piecewise_linear", "nonlinear", "na_signal"):
        x, y = make_regime(regime, rng)
        n_train = int(0.8 * N)
        xtr, xte = x[:n_train], x[n_train:]
        ytr, yte = y[:n_train], y[n_train:]
        rows_tr, rows_te = to_rows(xtr), to_rows(xte)

        for name, cls in models:
            t0 = time.perf_counter()
            if name == "ModelTree(linear)":
                m = ModelTree(max_depth=6, min_rows=40, leaf="linear")
                m.fit(rows_tr, [float(v) for v in ytr])
                pred = m.predict(rows_te)
            elif name == "ModelTree(path)":
                m = ModelTree(max_depth=6, min_rows=40, leaf_cols="path", leaf="linear")
                m.fit(rows_tr, [float(v) for v in ytr])
                pred = m.predict(rows_te)
            else:
                m = cls(
                    **(
                        {"iterations": 200, "verbose": 0, "allow_writing_files": False}
                        if cls is CatBoostRegressor
                        else {"n_estimators": 200}
                    )
                )
                m.fit(xtr, ytr)
                pred = [float(v) for v in m.predict(xte)]
            dt = time.perf_counter() - t0
            err = np.array(pred) - yte
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            print(f"{regime:<18}{name:<20}{rmse:>8.3f}{mae:>8.3f}{dt:>8.2f}")


if __name__ == "__main__":
    run()
