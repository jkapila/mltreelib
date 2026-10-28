"""Python-environment scorer for the Elo arena. Run SECOND:
    uv run --with pandas --with lightgbm --with catboost --with ctboost \
           python examples/elo_score_python.py
Reads benchmarks/elo_data/<ds>_{train,test}.csv, appends
(dataset,model,rmse,mae,fit_s) rows to benchmarks/elo_scores.csv.

Models: our unoptimized ModelTree (all three leaf flavors), LightGBM,
CatBoost, CTBoost — stock configs, no HPO (Optuna is a later phase).
"""

import csv
import math
import time
from pathlib import Path

from mltreelib import ModelTree

DATA_DIR = Path("benchmarks/elo_data")
OUT = Path("benchmarks/elo_scores.csv")
# Datasets are discovered from the prepared files - add data without
# touching the scorers.
DATASETS = sorted(p.name[: -len("_train.csv")] for p in DATA_DIR.glob("*_train.csv"))


def read_xy(path):
    x, y = [], []
    width = [0]
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)  # first row is the header, always
        width[0] = len(header) if header else 0
        for row in reader:
            if not row:
                continue
            if len(row) != width[0]:
                raise ValueError(f"{path}: row width {len(row)} != {width[0]}")
            x.append([None if v == "" else float(v) for v in row[:-1]])
            y.append(float(row[-1]))
    return x, y


def scores(pred, yte):
    err = [p - t for p, t in zip(pred, yte)]
    return (
        math.sqrt(sum(e * e for e in err) / len(err)),
        sum(abs(e) for e in err) / len(err),
    )


def append_row(ds, model, rmse, mae, fit):
    """Upsert: one row per (dataset, model); rerunning is incremental."""
    rows = []
    if OUT.exists():
        with open(OUT) as fh:
            rdr = csv.reader(fh)
            next(rdr, None)
            rows = [r for r in rdr if not (r[0] == ds and r[1] == model)]
    with open(OUT, "w") as fh:
        fh.write("dataset,model,rmse,mae,fit_s\n")
        for r in rows:
            fh.write(",".join(r) + "\n")
        fh.write(f"{ds},{model},{rmse:.4f},{mae:.4f},{fit:.3f}\n")


def make_models():
    import numpy as np
    import pandas as pd

    ms = {
        "ModelTree(all)": lambda xtr, ytr, xte: run_tree(xtr, ytr, xte, "linear", "all"),
        "ModelTree(path)": lambda xtr, ytr, xte: run_tree(xtr, ytr, xte, "linear", "path"),
        "ModelTree(mean)": lambda xtr, ytr, xte: run_tree(xtr, ytr, xte, "mean", "all"),
    }

    def attempt(name, builder):
        try:
            ms[name] = builder()
        except Exception as e:
            ms[name] = _raise(f"{type(e).__name__}: {e}")

    def _raise(msg):
        def f(*_a, **_k):
            raise RuntimeError(msg)
        return f

    def lgbm():
        import lightgbm as lgb

        def run(xtr, ytr, xte):
            t0 = time.perf_counter()
            b = lgb.train(
                {"objective": "regression", "verbosity": -1, "num_leaves": 31},
                lgb.Dataset(np.asarray(xtr, dtype=np.float64),
                            label=np.asarray(ytr, dtype=np.float64)),
                num_boost_round=200,
            )
            return b.predict(np.asarray(xte, dtype=np.float64)), time.perf_counter() - t0

        return run

    def catb():
        from catboost import CatBoostRegressor

        def run(xtr, ytr, xte):
            # The native Pool path trips a numpy-2 `Pool.copy` bug in this
            # build; the sklearn-style regressor is the working surface.
            m = CatBoostRegressor(iterations=200, verbose=0, allow_writing_files=False)
            t0 = time.perf_counter()
            m.fit(np.asarray(xtr, dtype=np.float64), np.asarray(ytr, dtype=np.float64))
            return m.predict(np.asarray(xte, dtype=np.float64)), time.perf_counter() - t0

        return run

    def ctb():
        from ctboost import CTBoostRegressor

        def run(xtr, ytr, xte):
            m = CTBoostRegressor()
            t0 = time.perf_counter()
            m.fit(np.asarray(xtr), np.asarray(ytr))
            return m.predict(np.asarray(xte)), time.perf_counter() - t0

        return run

    def skl_tree():
        from sklearn.tree import DecisionTreeRegressor

        def run(xtr, ytr, xte):
            # sklearn's CART cannot consume NAs — mean-impute (documented).
            cols = list(zip(*[[float("nan") if v is None else v for v in row] for row in xtr]))
            means = [sum(c) / len(c) if any(v == v for v in c) else 0.0 for c in cols]
            fill = lambda rows: [
                [means[j] if v is None else v for j, v in enumerate(row)] for row in rows
            ]
            m = DecisionTreeRegressor(random_state=0)  # stock, unpruned
            t0 = time.perf_counter()
            m.fit(np.asarray(fill(xtr)), np.asarray(ytr))
            return m.predict(np.asarray(fill(xte))), time.perf_counter() - t0

        return run

    def mtrees():
        import sys

        sys.path.insert(0, "references/repos/modeltrees/src")
        from modeltrees import lmtree as mt_lmtree

        def run(xtr, ytr, xte):
            cols = [f"x{i}" for i in range(len(xtr[0]))]
            f = "y ~ " + " + ".join(cols) + " | " + " + ".join(cols)
            df = pd.DataFrame(np.asarray(xtr, dtype=float), columns=cols)
            df["y"] = np.asarray(ytr, dtype=float)
            dte = pd.DataFrame(np.asarray(xte, dtype=float), columns=cols)
            t0 = time.perf_counter()
            m = mt_lmtree(f, data=df)
            return np.asarray(m.predict(newdata=dte), dtype=float), time.perf_counter() - t0

        return run

    def mars():
        import pymars as earth  # the mars-earth wheel ships module `pymars`

        def run(xtr, ytr, xte):
            # Like R's earth, MARS gets raw rows: NA handling is part of the
            # comparison (R earth failed fc2008's NA cells too).
            m = earth.Earth()
            t0 = time.perf_counter()
            m.fit(np.asarray(xtr, dtype=float), np.asarray(ytr, dtype=float))
            return m.predict(np.asarray(xte, dtype=float)), time.perf_counter() - t0

        return run

    attempt("LightGBM(200)", lgbm)
    attempt("CatBoost(200)", catb)
    attempt("CTBoost", ctb)
    attempt("sklearn tree", skl_tree)
    attempt("modeltrees lmtree", mtrees)
    attempt("MARS (mars-earth)", mars)
    return ms


def run_tree(xtr, ytr, xte, leaf, cols):
    kw = {"leaf": leaf}
    if cols == "path":
        kw["leaf_cols"] = "path"
    t0 = time.perf_counter()
    m = ModelTree(**kw)  # unoptimized: plain defaults
    m.fit(xtr, [float(v) for v in ytr])
    return m.predict(xte), time.perf_counter() - t0


def main():
    models = make_models()
    for ds in DATASETS:
        xtr, ytr = read_xy(DATA_DIR / f"{ds}_train.csv")
        xte, yte = read_xy(DATA_DIR / f"{ds}_test.csv")
        for name, fn in models.items():
            try:
                pred, fit = fn(xtr, ytr, xte)
                pairs = [(float(p), t) for p, t in zip(pred, yte) if p == p]
                if not pairs:
                    raise RuntimeError("no finite predictions")
                rmse, mae = scores([p for p, _ in pairs], [t for _, t in pairs])
                append_row(ds, name, rmse, mae, fit)
                print(f"{ds:<9} {name:<16} rmse={rmse:8.3f} mae={mae:8.3f} fit={fit:6.2f}")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"{ds:<9} {name:<16} FAILED ({msg[:70]})")
    print("Python scorer done ->", OUT)


if __name__ == "__main__":
    main()
