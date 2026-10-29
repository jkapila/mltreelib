"""HPO by data shape — one parameter at a time, raw data, real TabArena.

Run:  uv run --with optuna --with pandas --with pyarrow python examples/hpo_shapes.py

Three real datasets, each a different shape challenge:
  Food_Delivery_Time  45,451 x 10    large categories (Delivery_person_ID, 1,320 levels)
  QSAR-TID-11          5,742 x 1,025 very wide features
  diamonds            53,940 x 10    very long data (cut/color/clarity levels)

The knob surface is heavy now, so each parameter is tuned ALONE around the
defaults (coordinate-wise): 10 Optuna trials per (shape, parameter). Every
trial records rmse and fit wall time; each study reports the best model,
the fastest model, and the sweet spot (best rmse within 1.5x of the
fastest fit). Raw cells go straight to the model: ints, floats, level
strings, NaN — the three data handlers, no preprocessing.
"""

import logging
import random
import signal
import time

import optuna
import pandas as pd

from mltreelib import ModelTree

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("benchmarks/hpo_shapes.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("hpo.shapes")

SHAPES = {
    "large-categories": ("Food_Delivery_Time", "Time_taken(min)"),
    "very-wide": ("QSAR-TID-11", "MEDIAN_PXC50"),
    "very-long": ("diamonds", "price"),
}

TRIAL_CAP_S = 180  # per-fit wall cap; over-cap trials lose on time
N_TRIALS = 10

SPACE = {
    "max_depth": lambda t: {"max_depth": t.suggest_int("max_depth", 1, 12)},
    "min_rows": lambda t: {"min_rows": t.suggest_categorical("min_rows", [2, 5, 10, 25, 50, 100, 250, 500])},
    "split_precision": lambda t: {"split_precision": t.suggest_categorical("split_precision", [None, 0.1, 0.01, 0.001, 0.0001])},
    "leaf": lambda t: {"leaf": t.suggest_categorical("leaf", ["mean", "linear", "median", "huber"])},
    "l2": lambda t: {"l2": t.suggest_float("l2", 1e-4, 10.0, log=True)},
    "prune/cp_alpha": lambda t: {"prune": "cp", "cp_alpha": t.suggest_float("cp_alpha", 1e-4, 10.0, log=True)},
    "col_sample": lambda t: {"col_sample": t.suggest_categorical("col_sample", [None, 0.25, 0.5, 0.75])},
}


class TrialTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise TrialTimeout()


def load(name, ycol):
    df = pd.read_parquet(f"datasets/{name}.parquet")
    xcols = [c for c in df.columns if c != ycol]
    dtypes = [
        "categorical" if str(df[c].dtype) in ("object", "category") else "numeric"
        for c in xcols
    ]
    x = df[xcols].values.tolist()  # raw cells as-is
    y = df[ycol].astype(float).tolist()
    rng = random.Random(42)
    idx = list(range(len(y)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    return ([x[i] for i in tr], [y[i] for i in tr],
            [x[i] for i in te], [y[i] for i in te], dtypes)


def run_study(label, xtr, ytr, xte, yte, dtypes, param):
    def objective(trial):
        kw = SPACE[param](trial)
        m = ModelTree(**kw)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(TRIAL_CAP_S)
        t0 = time.perf_counter()
        try:
            m.fit(xtr, ytr, dtypes=dtypes)
            dt = time.perf_counter() - t0
        except TrialTimeout:
            signal.alarm(0)
            trial.set_user_attr("fit_s", float("inf"))
            return float("inf")
        signal.alarm(0)
        pred = m.predict(xte)
        rmse = (sum((p - y) ** 2 for p, y in zip(pred, yte)) / len(yte)) ** 0.5
        trial.set_user_attr("fit_s", dt)
        trial.set_user_attr("params_used", kw)
        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))
    trials = [t for t in study.trials if t.value is not None and t.value < float("inf")]
    if not trials:
        log.info(f"  {param:<16} all trials over {TRIAL_CAP_S}s cap")
        return
    best = min(trials, key=lambda t: t.value)
    fastest = min(trials, key=lambda t: t.user_attrs.get("fit_s", float("inf")))
    fmin = fastest.user_attrs["fit_s"]
    sweet = min(
        (t for t in trials if t.user_attrs.get("fit_s", float("inf")) <= 1.5 * fmin),
        key=lambda t: t.value,
        default=best,
    )
    log.info(
        f"  {param:<16} best rmse={best.value:>10.4f} ({best.user_attrs['fit_s']:.2f}s)  "
        f"fastest={fastest.user_attrs['fit_s']:>6.2f}s (rmse={fastest.value:.4f})  "
        f"sweet rmse={sweet.value:.4f} ({sweet.user_attrs['fit_s']:.2f}s)"
    )
    log.info(f"  {'':16} best params: {best.user_attrs['params_used']}")


def main():
    for label, (name, ycol) in SHAPES.items():
        xtr, ytr, xte, yte, dtypes = load(name, ycol)
        log.info(f"== {label}: {name} ({len(ytr)} train rows, {len(dtypes)} cols) ==")
        for param in SPACE:
            log.info(f"  {param}: trial 1/{N_TRIALS}", extra={})
            run_study(label, xtr, ytr, xte, yte, dtypes, param)


if __name__ == "__main__":
    main()
