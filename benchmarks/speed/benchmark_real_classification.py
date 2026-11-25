"""Paired real-data classification benchmark for split evaluation and calibration.

The script writes a JSON payload from local Parquet files, then runs that payload
under either the current checkout or a separately installed wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path


def _loss(y: list[int], probabilities: list[list[float]]) -> float:
    return -sum(math.log(max(row[label], 1e-15)) for label, row in zip(y, probabilities)) / len(y)


def _brier(y, probabilities):
    return sum(sum((float(c == label) - p) ** 2 for c, p in enumerate(row)) for label, row in zip(y, probabilities)) / len(y)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(probabilities, classes: int) -> list[list[float]]:
    if classes == 2 and probabilities and isinstance(probabilities[0], (int, float)):
        return [[1.0 - p, p] for p in probabilities]
    return probabilities


def make_payload(path: Path, out: Path, split: Path | None = None) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    target = "ObjectType" if "ObjectType" in data else "TestedPositiveForDiabetes"
    labels = {name: i for i, name in enumerate(sorted(set(data[target])))}
    y = [labels[value] for value in data.pop(target)]
    names = list(data)
    # SDSS categorical identifiers are excluded. Diabetes has numeric columns only.
    names = [name for name in names if not table.schema.field(name).type.__class__.__name__.startswith("Dictionary")]
    x = [[data[name][i] for name in names] for i in range(len(y))]
    if target == "ObjectType":
        order = sorted(range(len(y)), key=lambda i: ((i * 1103515245 + 12345) & 0x7FFFFFFF))
        cut = int(len(order) * 0.8)
        train, test = order[:cut], order[cut:]
        calibration = train[int(len(train) * 0.75) :]
        train = train[: int(len(train) * 0.75)]
        kind = "sdss17"
    else:
        # OpenML task 363629 fold 0, with its train rows split into fit/calibration.
        if split is None:
            raise ValueError("--split-file is required for Diabetes payloads")
        train, test = [], []
        for line in split.read_text().splitlines():
            if not line or line.startswith("@") or line.startswith("%"):
                continue
            row = line.split(",")
            if len(row) == 4 and row[2:] == ["0", "0"]:
                (train if row[0] == "TRAIN" else test).append(int(float(row[1])))
        calibration = train[len(train) // 2 :]
        train = train[: len(train) // 2]
        kind = "diabetes"
    out.write_text(json.dumps({"kind": kind, "x": x, "y": y, "train": train, "calibration": calibration, "test": test}, allow_nan=False))


def run(payload_path: Path, mode: str, output: Path) -> None:
    from mltreelib import ModelTree

    payload = json.loads(payload_path.read_text())
    x, y = payload["x"], payload["y"]
    train, calibration, test = payload["train"], payload["calibration"], payload["test"]
    assert not (set(train) & set(calibration) or set(train) & set(test) or set(calibration) & set(test))
    assert len(x) == len(y) and x and all(len(row) == len(x[0]) for row in x)
    classes = len(set(y))
    kwargs = dict(max_depth=6, min_rows=30, gain="gini", leaf="mean", prune="none", seed=42, n_classes=classes, split_eval=mode, leaf_family="softmax" if classes > 2 else "binomial")
    trials = []
    for _ in range(3):
        model = ModelTree(**kwargs)
        started = time.perf_counter()
        model.fit([x[i] for i in train], [y[i] for i in train])
        fit = time.perf_counter() - started
        started = time.perf_counter()
        probs = model.predict_proba([x[i] for i in test])
        predict = time.perf_counter() - started
        trials.append({"fit_seconds": fit, "predict_seconds": predict})
    probs = _rows(probs, classes)
    assert len(probs) == len(test) and all(all(math.isfinite(p) and 0 <= p <= 1 for p in row) and abs(sum(row) - 1.0) < 1e-8 for row in probs)
    stable = _rows(model.predict_proba([x[i] for i in test]), classes)
    assert json.dumps(probs, separators=(",", ":")) == json.dumps(stable, separators=(",", ":"))
    raw_loss = _loss([y[i] for i in test], probs)
    raw_accuracy = sum(max(range(classes), key=lambda c: row[c]) == y[i] for i, row in zip(test, probs)) / len(test)
    package_file = Path(sys.modules["mltreelib"].__file__)
    binaries = sorted(package_file.parent.glob("_core*.so"))
    result = {"package": str(package_file), "version": getattr(sys.modules["mltreelib"], "__version__", None), "binary_sha256": _sha256(binaries[0]) if binaries else None, "payload_sha256": _sha256(payload_path), "kind": payload["kind"], "mode": mode, "n_train": len(train), "n_calibration": len(calibration), "n_test": len(test), "n_features": len(x[0]), "classes": classes, "raw_log_loss": raw_loss, "raw_brier": _brier([y[i] for i in test], probs), "raw_accuracy": raw_accuracy, "probability_sha256": hashlib.sha256(json.dumps(probs, separators=(",", ":")).encode()).hexdigest(), "trials": trials, "fit_median_seconds": statistics.median(t["fit_seconds"] for t in trials), "predict_median_seconds": statistics.median(t["predict_seconds"] for t in trials)}
    if payload["kind"] == "diabetes":
        cal_model = ModelTree(**kwargs)
        cal_model.fit([x[i] for i in train], [y[i] for i in train])
        cal_probs = _rows(cal_model.predict_proba([x[i] for i in calibration]), classes)
        test_probs = _rows(cal_model.predict_proba([x[i] for i in test]), classes)
        try:
            from mltreelib.calibration import IsotonicCalibrator

            started = time.perf_counter()
            calibrator = IsotonicCalibrator.fit([y[i] for i in calibration], [row[1] for row in cal_probs])
            calibrated_positive = calibrator.predict([row[1] for row in test_probs])
            iso_seconds = time.perf_counter() - started
            calibrated = [[1.0 - p, p] for p in calibrated_positive]
            result["isotonic_log_loss"] = _loss([y[i] for i in test], calibrated)
            result["isotonic_brier"] = _brier([y[i] for i in test], calibrated)
            result["isotonic_fit_predict_seconds"] = iso_seconds
        except (ImportError, AttributeError):
            result["isotonic_log_loss"] = None
    output.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-payload", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--mode", choices=("fast", "histogram"), default="fast")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-file", type=Path)
    args = parser.parse_args()
    if args.make_payload:
        if args.data is None:
            raise SystemExit("--data is required with --make-payload")
        make_payload(args.data, args.make_payload, args.split_file)
    else:
        if args.payload is None:
            raise SystemExit("--payload is required without --make-payload")
        if args.output is None:
            raise SystemExit("--output is required without --make-payload")
        run(args.payload, args.mode, args.output)


if __name__ == "__main__":
    main()
