"""ModelTree vs LightGBM vs CatBoost — how far are we?

Three regimes:
  1. piecewise_linear — the model tree's home turf (segmented linear signal);
  2. nonlinear — interactions and curves (the ensembles' home turf);
  3. na_signal — 20% missing x0 whose missingness itself carries the signal.

All three include NAs, so every model exercises its native missing handling.
LightGBM and CatBoost run through their **native APIs** (`lgb.train` Booster,
`catboost.train` Pool) — no scikit-learn wrappers. Models whose packages
cannot be installed (slow/blocked internet) are reported as skipped, never
silently dropped. The ModelTree per-level timing profile (group/sort, scan,
leaf fit) prints at the end.

Run:
    uv run --with lightgbm --with catboost python examples/model_tree_benchmark.py
"""

import time

import numpy as np

from mltreelib import ModelTree

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import catboost as cb
except ImportError:  # pragma: no cover
    cb = None

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


def lightgbm_predict(xtr, ytr, xte):
    dtrain = lgb.Dataset(xtr, label=ytr)
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1, "num_leaves": 31},
        dtrain,
        num_boost_round=200,
    )
    return booster.predict(xte)


def catboost_predict(xtr, ytr, xte):
    model = cb.train(
        {"iterations": 200, "loss_function": "RMSE", "verbose": 0},
        cb.Pool(xtr, label=ytr),
    )
    return model.predict(xte)


def run():
    rng = np.random.default_rng(SEED)
    models = [
        ("ModelTree(linear)", "tree_all"),
        ("ModelTree(path)", "tree_path"),
        ("LightGBM (booster)", lightgbm_predict if lgb is not None else None),
        ("CatBoost (native)", catboost_predict if cb is not None else None),
    ]

    print(f"{'regime':<18}{'model':<20}{'RMSE':>8}{'MAE':>8}{'fit s':>8}")
    print("-" * 62)
    last_profile = None
    for regime in ("piecewise_linear", "nonlinear", "na_signal"):
        x, y = make_regime(regime, rng)
        n_train = int(0.8 * N)
        xtr, xte = x[:n_train], x[n_train:]
        ytr, yte = y[:n_train], y[n_train:]
        rows_tr, rows_te = to_rows(xtr), to_rows(xte)

        for name, kind in models:
            if kind is None:
                print(f"{regime:<18}{name:<20}{'skipped':>8}{'':>8}{'':>8}  (deps unavailable)")
                continue
            t0 = time.perf_counter()
            try:
                if kind == "tree_all":
                    m = ModelTree(max_depth=6, min_rows=40, leaf="linear")
                    m.fit(rows_tr, [float(v) for v in ytr])
                    pred = m.predict(rows_te)
                    last_profile = m.profile()
                elif kind == "tree_path":
                    m = ModelTree(max_depth=6, min_rows=40, leaf_cols="path", leaf="linear")
                    m.fit(rows_tr, [float(v) for v in ytr])
                    pred = m.predict(rows_te)
                else:
                    pred = [float(v) for v in kind(xtr, ytr, xte)]
            except Exception as exc:  # test mode: never die on a missing dep
                print(f"{regime:<18}{name:<20}{'skipped':>8}{'':>8}{'':>8}  ({type(exc).__name__})")
                continue
            dt = time.perf_counter() - t0
            err = np.array(pred) - yte
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            print(f"{regime:<18}{name:<20}{rmse:>8.3f}{mae:>8.3f}{dt:>8.2f}")

    if last_profile is not None:
        print("\nModelTree per-level timings (last fit):")
        print(
            f"{'depth':>5}{'nodes':>7}{'rows':>8}{'cands':>10}"
            f"{'group ms':>10}{'scan ms':>10}{'leaf ms':>10}"
        )
        for p in last_profile:
            print(
                f"{p['depth']:>5}{p['nodes']:>7}{p['rows']:>8}{p['candidates']:>10}"
                f"{p['group_ns'] / 1e6:>10.1f}{p['scan_ns'] / 1e6:>10.1f}"
                f"{p['leaf_ns'] / 1e6:>10.1f}"
            )


if __name__ == "__main__":
    run()
