"""Holistic real-data comparison: ModelTree vs rpart vs Cubist vs LightGBM
vs CatBoost on two real regression datasets.

Datasets (RuleQuest demo suite, `references/sources/cubist-demo/CubistR2/Data/`):
  - concrete: 1,030 rows x 8 numeric predictors (mid-size);
  - abalone:  4,177 rows x 8 predictors incl. one categorical (large).

All models see the identical 80/20 split (seed 7) and the identical
one-hot encoding of the single categorical. Notes:
  - rpart runs live via R (`examples/rpart_realdata.R`), default control;
  - Cubist runs through its official demo binary (`bin/cubistdemo`) — the
    demo caps train/test at 200 cases each, so it sees a seeded 200/200
    subset (footnoted in the output);
  - LightGBM/CatBoost use their native boosters; missing packages print as
    skipped, never silently dropped.

Run:
    uv run --with lightgbm --with catboost python examples/realdata_benchmark.py
"""

import math
import subprocess
import time
from pathlib import Path

from mltreelib import ModelTree

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import catboost as cb
except ImportError:  # pragma: no cover
    cb = None

DATA = Path("references/sources/cubist-demo/CubistR2/Data")
CUBIST = Path("references/sources/cubist-demo/CubistR2/bin/cubistdemo")
SEED = 7


def load(name, target_idx, cat_idx=None):
    rows, y = [], []
    for line in (DATA / f"{name}.data").read_text().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        y.append(float(parts[target_idx]))
        feats = []
        for i, p in enumerate(parts):
            if i == target_idx:
                continue
            if cat_idx is not None and i == cat_idx:
                for lev in ("I", "M", "F"):
                    feats.append(1.0 if p == lev else 0.0)
            else:
                feats.append(float(p))
        rows.append(feats)
    return rows, y


def split(rows, y):
    idx = list(range(len(y)))
    rng = __import__("random").Random(SEED)
    rng.shuffle(idx)
    n_tr = int(0.8 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    f = lambda a, i: [a[k] for k in i]
    return f(rows, tr), f(y, tr), f(rows, te), f(y, te)


def scores(pred, yte):
    err = [p - t for p, t in zip(pred, yte)]
    rmse = math.sqrt(sum(e * e for e in err) / len(err))
    mae = sum(abs(e) for e in err) / len(err)
    return rmse, mae


def fmt(name, rmse, mae, secs, note=""):
    return f"{name:<22}{rmse:>9.3f}{mae:>9.3f}{secs:>9.2f}  {note}"


def run_rpart(xtr, ytr, xte, yte, tmp):
    def write_csv(path, x, y):
        with open(path, "w") as fh:
            fh.write(",".join([f"x{i}" for i in range(len(x[0]))]) + ",y\n")
            for row, t in zip(x, y):
                fh.write(",".join(str(v) for v in row) + f",{t}\n")

    write_csv(tmp / "train.csv", xtr, ytr)
    write_csv(tmp / "test.csv", xte, yte)
    out = subprocess.run(
        ["Rscript", "examples/rpart_realdata.R", str(tmp / "train.csv"), str(tmp / "test.csv")],
        capture_output=True, text=True,
    )
    line = out.stdout.strip().splitlines()[-1]
    _, rmse, mae, fit = line.split()
    return float(rmse.split("=")[1]), float(mae.split("=")[1]), float(fit.split("=")[1])


def run_cubist(name, xtr, ytr, xte, yte, tmp):
    # Cubist's demo caps train/test at 200 cases; seeded subset, footnoted.
    n = len(xtr[0])
    lines = ["y."]
    lines += [f"a{i}: continuous." for i in range(n)]
    lines += ["y: continuous."]
    (tmp / "c.names").write_text("\n".join(lines) + "\n")
    seed_rng = __import__("random").Random(SEED)
    tr = seed_rng.sample(range(len(ytr)), min(200, len(ytr)))
    te = seed_rng.sample(range(len(yte)), min(200, len(yte)))
    (tmp / "c.data").write_text(
        "".join(",".join(str(v) for v in xtr[k]) + f",{ytr[k]}\n" for k in tr)
    )
    (tmp / "c.test").write_text(
        "".join(",".join(str(v) for v in xte[k]) + f",{yte[k]}\n" for k in te)
    )
    t0 = time.perf_counter()
    out = subprocess.run([str(CUBIST), "-f", str(tmp / "c")], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    mae = float(next(l for l in out.stdout.splitlines() if "Average" in l and "error" in l).split()[-1])
    # Cubist's |error| is the mean absolute error on its 200-case test sample.
    # Recover RMSE the same way our table does: rerun predictions is not
    # exposed by the demo, so report MAE + its own correlation line.
    return mae, wall


def main():
    import tempfile

    for ds, t_idx, cat_idx in [("concrete", 8, None), ("abalone", 8, 0)]:
        rows, y = load(ds, t_idx, cat_idx)
        xtr, ytr, xte, yte = split(rows, y)
        print(f"\n== {ds}: {len(ytr)} train / {len(yte)} test, {len(xtr[0])} cols ==")
        print(f"{'model':<22}{'RMSE':>9}{'MAE':>9}{'secs':>9}")
        print("-" * 55)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            for name, kw in [
                ("ModelTree linear(all)", {"leaf": "linear"}),
                ("ModelTree linear(path)", {"leaf": "linear", "leaf_cols": "path"}),
                ("ModelTree mean(CART)", {"leaf": "mean"}),
            ]:
                t0 = time.perf_counter()
                m = ModelTree(max_depth=8, min_rows=20, **kw)
                m.fit(xtr, [float(v) for v in ytr])
                pred = m.predict(xte)
                dt = time.perf_counter() - t0
                rmse, mae = scores(pred, yte)
                print(fmt(name, rmse, mae, dt))

            try:
                rmse, mae, fit = run_rpart(xtr, ytr, xte, yte, tmp)
                print(fmt("rpart (live R, default)", rmse, mae, fit))
            except Exception as exc:
                print(fmt("rpart (live R, default)", 0, 0, 0, f"skipped ({type(exc).__name__})"))

            if CUBIST.exists():
                try:
                    mae, wall = run_cubist(ds, xtr, ytr, xte, yte, tmp)
                    print(fmt("Cubist 2.10 (demo)", float("nan"), mae, wall, "[200-case demo cap; MAE only]"))
                except Exception as exc:
                    print(fmt("Cubist 2.10 (demo)", 0, 0, 0, f"skipped ({type(exc).__name__})"))

            if lgb is not None:
                import numpy as np

                t0 = time.perf_counter()
                booster = lgb.train(
                    {"objective": "regression", "verbosity": -1, "num_leaves": 31},
                    lgb.Dataset(np.asarray(xtr, dtype=np.float64),
                                label=np.asarray(ytr, dtype=np.float64)),
                    num_boost_round=200,
                )
                pred = [float(v) for v in booster.predict(np.asarray(xte, dtype=np.float64))]
                dt = time.perf_counter() - t0
                rmse, mae = scores(pred, yte)
                print(fmt("LightGBM (200 rounds)", rmse, mae, dt))
            else:
                print(fmt("LightGBM (200 rounds)", 0, 0, 0, "skipped (deps unavailable)"))

            if cb is not None:
                import numpy as np

                t0 = time.perf_counter()
                model = cb.train(
                    {"iterations": 200, "loss_function": "RMSE", "verbose": 0},
                    cb.Pool(np.asarray(xtr, dtype=np.float64),
                            label=np.asarray(ytr, dtype=np.float64)),
                )
                pred = [float(v) for v in model.predict(np.asarray(xte, dtype=np.float64))]
                dt = time.perf_counter() - t0
                rmse, mae = scores(pred, yte)
                print(fmt("CatBoost (native)", rmse, mae, dt))
            else:
                print(fmt("CatBoost (native)", 0, 0, 0, "skipped (deps unavailable)"))


if __name__ == "__main__":
    main()
