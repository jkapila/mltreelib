"""Rerun the archived Diamonds field on the corrected price target."""
import csv
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(RUN / "examples"))
from elo_budget import run_with_budget
from elo_common import AttemptsLedger, ScoreSheet, read_split, setup_logging

DS = "ta_diamonds"
WORK = OUT / "work"
SPLITS = WORK / "benchmarks" / "elo_data"
COLUMNS = ["dataset", "model", "lang", "cat_handling", "rmse", "mae", "fit_s"]


def worker(name):
    xtr, ytr, xte, yte, meta = read_split(SPLITS, DS)
    if name == "ModelTree(v0.3.3-default)":
        sys.path.append(os.environ["ELO_312_DEPS"])
        from mltreelib import ModelTree
        assert importlib.metadata.version("mltreelib") == "0.3.3"
        model = ModelTree()
        start = time.perf_counter()
        model.fit(xtr, ytr, dtypes=list(meta["columns"].values()))
        pred = model.predict(xte)
        fit = time.perf_counter() - start
    else:
        if name.startswith(("ModelTree(", "mltreelib MARS(")):
            sys.path.append(os.environ["ELO_312_DEPS"])
            assert importlib.metadata.version("mltreelib") == "0.4.11"
            from elo_models_reg import make_models
        else:
            from external_registry import make_models
        pred, fit = make_models(meta)[name](xtr, ytr, xte)
    assert len(pred) == len(yte) and all(math.isfinite(p) for p in pred), "Incomplete or non-finite predictions"
    errors = [p - y for p, y in zip(pred, yte)]
    print(f"RESULT {DS},{name},{math.sqrt(sum(e * e for e in errors) / len(errors))},"
          f"{sum(map(abs, errors)) / len(errors)},{fit}", flush=True)


def main(phase, budget):
    log = setup_logging("elo.diamonds.peers", str(OUT / "run.log"))
    with (RUN / "frozen_reg.csv").open() as f:
        field = {row["model"]: row for row in csv.DictReader(f) if row["dataset"] == DS}
    own = "ModelTree(all)"
    assert own in field and len(field) == 35
    for split in ("train", "test"):
        with (SPLITS / f"{DS}_{split}.csv").open() as f:
            assert next(csv.reader(f)) == ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"]
    sheet = ScoreSheet(OUT / "scores.csv", COLUMNS)
    if (DS, own) not in sheet.done_pairs():
        with (RUN / "reg_scores.csv").open() as f:
            row = next(r for r in csv.DictReader(f) if r["dataset"] == DS)
        sheet.upsert(DS, own, rmse=float(row["rmse"]), mae=float(row["mae"]), fit_s=float(row["fit_s"]))
    ledger = AttemptsLedger(OUT / f"attempts_{phase}_{budget:g}s.csv")
    done = sheet.done_pairs()
    for name, peer in field.items():
        if phase == "r" and peer["lang"] != "R" or phase == "py" and peer["lang"] == "R":
            continue
        if phase == "legacy" and name != "ModelTree(v0.3.3-default)":
            continue
        if (DS, name) in done:
            continue
        if peer["lang"] == "R":
            argv = ["Rscript", "--vanilla", str(ROOT / "examples/benchmarks/elo/elo_r_worker.R"), "reg", DS, name]
        else:
            if name == "ModelTree(v0.3.3-default)":
                python = os.environ.get("ELO_PY033", "")
            elif name.startswith(("ModelTree(", "mltreelib MARS(")):
                python = sys.executable
            else:
                python = os.environ["ELO_EXTERNAL_PY"]
            if not python:
                ledger.record(DS, name, "missing-v0.3.3", 0)
                log.warning("%s requires the historical wheel", name)
                continue
            argv = [python, __file__, "worker", name]
        log.info("START %s", name)
        start = time.monotonic()
        out, code, timed_out = run_with_budget(argv, budget_s=budget, cwd=WORK)
        status = "timeout" if timed_out else "error" if code else "invalid"
        for line in out.splitlines():
            if code != 0 or not line.startswith(f"RESULT {DS},{name},"):
                continue
            values = list(map(float, line.rsplit(",", 3)[-3:]))
            if all(math.isfinite(v) and v >= 0 for v in values):
                rmse, mae, fit_s = values
                sheet.upsert(DS, name, rmse=rmse, mae=mae, fit_s=fit_s,
                             lang=peer["lang"], cat_handling=peer["cat_handling"])
                status = "ok"
        elapsed = time.monotonic() - start
        ledger.record(DS, name, status, elapsed)
        log.info("%s %s %.1fs", name, status, elapsed)
    covered = {name for ds, name in sheet.done_pairs() if ds == DS}
    (OUT / "coverage.json").write_text(json.dumps(dict(expected=len(field), scored=len(covered),
        missing=sorted(field.keys() - covered)), indent=2) + "\n")
    log.info("Coverage %d/%d", len(covered), len(field))


if __name__ == "__main__":
    phase, value = sys.argv[1:]
    if phase == "worker":
        worker(value)
    else:
        with (OUT / "phase.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            main(phase, float(value))
