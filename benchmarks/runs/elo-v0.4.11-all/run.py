"""Rerun the v0.4.7 benchmark recipes against the v0.4.11 wheel."""
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT / "examples"))

from cla_scores import score_cla
from elo_budget import run_with_budget
from elo_common import AttemptsLedger, ScoreSheet, read_split, regression_scores, setup_logging

NAME = "ModelTree(all)"
TASKS = {"reg": ("elo_data", 18), "cla": ("elo_cla_data", 38)}
METRICS = {"reg": ["rmse", "mae"], "cla": ["logloss", "accuracy", "bal_acc", "auc", "macro_f1"]}


def data_dir(task, ds=None):
    if task == "reg" and ds == "ta_diamonds":
        return OUT / "corrected" / "elo_data"
    return ROOT / "benchmarks" / TASKS[task][0]


def worker(task, ds):
    if task == "reg":
        from recipe_reg import make_models
    else:
        from recipe_cla import make_models

    xtr, ytr, xte, yte, meta = read_split(data_dir(task, ds), ds, is_classification=task == "cla")
    assert xtr and xte and meta["columns"]
    predictions, fit_s = make_models(meta)[NAME](xtr, ytr, xte)
    assert len(predictions) == len(yte), "Incomplete predictions"
    if task == "reg":
        assert all(map(math.isfinite, predictions)), "Non-finite regression predictions"
        assert all(map(math.isfinite, yte)), "Non-finite regression targets"
        rmse, mae = regression_scores(predictions, yte)
        metrics = dict(rmse=rmse, mae=mae)
    else:
        classes = meta["classes"]
        assert all(
            len(row) == len(classes) and all(math.isfinite(p) and 0 <= p <= 1 for p in row)
            and math.isclose(sum(row), 1.0, abs_tol=1e-7)
            for row in predictions
        ), "Invalid class probabilities"
        metrics = score_cla(predictions, yte, classes)
    assert all(math.isfinite(val) for val in metrics.values() if val is not None)
    print(json.dumps(dict(metrics, fit_s=fit_s), allow_nan=False))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(budget):
    import mltreelib._core as core

    assert metadata.version("mltreelib") == "0.4.11"
    log = setup_logging("elo.v0.4.11", str(OUT / "run.log"))
    cases = {}
    hashes = {}
    for task, (directory, count) in TASKS.items():
        base = data_dir(task)
        cases[task] = sorted(p.name.removesuffix(".meta.json") for p in base.glob("*.meta.json"))
        assert len(cases[task]) == count, (task, cases[task])
        for ds in cases[task]:
            dataset_dir = data_dir(task, ds)
            meta = json.loads((dataset_dir / f"{ds}.meta.json").read_text())
            for split in ("train", "test"):
                with (dataset_dir / f"{ds}_{split}.csv").open(newline="") as stream:
                    reader = csv.reader(stream)
                    header = next(reader)
                    assert header[:-1] == list(meta["columns"]), (ds, split, "feature roles")
                    assert header[-1] in (meta["target"], "y"), (ds, split, "target")
                    assert header[-1] not in meta["columns"], (ds, split, "target collision")
            for suffix in (".meta.json", "_train.csv", "_test.csv"):
                path = data_dir(task, ds) / (ds + suffix)
                assert path.stat().st_size > 0
                hashes[f"{directory}/{path.name}"] = sha(path)

        history = subprocess.check_output(
            ["git", "show", f"v0.4.7:benchmarks/{'elo' if task == 'reg' else 'elo_cla'}_scores.csv"],
            cwd=ROOT, text=True,
        )
        historical_tasks = {row["dataset"] for row in csv.DictReader(history.splitlines())}
        assert set(cases[task]) <= historical_tasks, set(cases[task]) - historical_tasks

    manifest = dict(
        version=metadata.version("mltreelib"),
        release_commit=subprocess.check_output(["git", "rev-parse", "v0.4.11^{}"], cwd=ROOT, text=True).strip(),
        recipe_commit=subprocess.check_output(["git", "rev-parse", "v0.4.7^{}"], cwd=ROOT, text=True).strip(),
        extension=core.__file__,
        extension_sha256=sha(Path(core.__file__)),
        python=sys.version,
        recipe_sha256={p.name: sha(p) for p in sorted(OUT.glob("recipe_*.py"))},
        cases=cases,
        inputs_sha256=hashes,
        corrected_split="reg/ta_diamonds",
        corrected_split_source_sha256=sha(ROOT / "datasets" / "diamonds.parquet"),
    )
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Run inputs changed"
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Checked %s prepared splits and v%s; %.0fs budget", {k: len(v) for k, v in cases.items()}, manifest["version"], budget)

    failed = []
    for task, datasets in cases.items():
        scores = ScoreSheet(OUT / f"{task}_scores.csv", ["dataset", "model", *METRICS[task], "fit_s", "budget_s"])
        ledger = AttemptsLedger(OUT / f"{task}_attempts_{budget:g}s.csv")
        done = scores.done_pairs()
        for ds in datasets:
            if (ds, NAME) in done:
                continue
            log.info("START %s %s", task, ds)
            start = time.monotonic()
            out, code, timed_out = run_with_budget([sys.executable, __file__, task, ds], budget_s=budget)
            elapsed = time.monotonic() - start
            status = "timeout" if timed_out else "error" if code else "ok"
            if status == "ok":
                metrics = json.loads(out.splitlines()[-1])
                scores.upsert(ds, NAME, **metrics, budget_s=budget)
            else:
                failed.append((task, ds, status))
            ledger.record(ds, NAME, status, elapsed)
            log.info("%s %s %s %.1fs", task, ds, status, elapsed)
        log.info("%s coverage %d/%d", task, len(scores.done_pairs()), len(datasets))
    log.info("Remaining: %s", failed)
    return bool(failed)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        worker(*sys.argv[1:])
    else:
        sys.exit(run(float(sys.argv[1])))
