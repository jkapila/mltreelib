"""Benchmark unsupervised facades on fixed TabArena-v0.1 partitions.

Run from the repository with ``uv run python examples/guides/real_unsupervised_workflow.py``.
Use ``--release-baseline`` only with the published 0.4.11 wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from random import Random

import pandas as pd
import pyarrow
from mltreelib import AnomalyTree, ClusterTree, _core

RELEASE_SNAPSHOT_ERRORS = {
    "hst": "JSON snapshots are available for eif, isolation, extended_isolation, got 'hst'.",
    "cltree": "JSON snapshots are available for pct, predictive, clus, got 'cltree'.",
}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def run_model(factory, options, train, test, fit_kwargs, labels, reps, release):
    anomaly = factory is AnomalyTree
    predict_kwargs = {"kind": "score"} if anomaly else {}
    kind = options["kind"]
    expected_error = RELEASE_SNAPSHOT_ERRORS.get(kind) if release else None
    raw = []
    output_signatures = set()
    for _ in range(reps):
        model = factory(**options)
        start = time.perf_counter()
        model.fit(train, **fit_kwargs)
        fit = time.perf_counter() - start
        start = time.perf_counter()
        output = model.predict(test, **predict_kwargs)
        predict = time.perf_counter() - start
        assert len(output) == len(test), kind
        if anomaly:
            assert all(math.isfinite(score) for score in output), kind
            state = [model.threshold, model.predict(test)]
            assert model.threshold is not None, kind
        else:
            state = [model.n_clusters_, model.labels_]
            assert len(model.labels_) == len(train), kind
        output_signatures.add((digest(output), digest(state)))

        start = time.perf_counter()
        try:
            snapshot = model.to_bytes()
        except ValueError as error:
            if expected_error is None or str(error) != expected_error:
                raise
            raw.append({"fit_s": fit, "predict_s": predict})
            continue
        save = time.perf_counter() - start
        assert expected_error is None, f"{kind} unexpectedly supports release snapshots"
        start = time.perf_counter()
        restored = factory.from_bytes(snapshot)
        load = time.perf_counter() - start
        assert output == restored.predict(test, **predict_kwargs), kind
        restored_state = (
            [restored.threshold, restored.predict(test)]
            if anomaly
            else [restored.n_clusters_, restored.labels_]
        )
        assert state == restored_state, kind
        raw.append(
            {
                "fit_s": fit,
                "predict_s": predict,
                "save_s": save,
                "load_s": load,
                "snapshot_bytes": len(snapshot),
                "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
            }
        )

    assert len(output_signatures) == 1, f"{kind} changed across fresh fits"
    assert len({rep.get("snapshot_sha256") for rep in raw}) == 1, kind
    output_sha, state_sha = output_signatures.pop()
    summary = {
        "model": kind,
        "options": options,
        "raw": raw,
        "median_s": {
            key: statistics.median(rep[key] for rep in raw)
            for key in ("fit_s", "predict_s", "save_s", "load_s")
            if key in raw[0]
        },
        "output_sha256": output_sha,
        "state_sha256": state_sha,
        "output_distinct": len(set(output)),
        "snapshot_error": expected_error,
        "roundtrip_score_and_state_equal": True if expected_error is None else None,
    }
    if labels is not None and anomaly:
        summary["average_precision"] = average_precision(labels, output)
    if not anomaly:
        summary["test_cluster_sizes"] = {
            str(label): output.count(label) for label in set(output)
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--rows", type=int, help="Holdout rows, also the fit size for APS"
    )
    parser.add_argument("--dataset", choices=("bank", "aps"), default="bank")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--release-baseline", action="store_true")
    args = parser.parse_args()
    if args.reps < 1 or (args.rows is not None and args.rows < 1):
        parser.error("--reps and --rows must be positive")
    if args.release_baseline and version("mltreelib") != "0.4.11":
        parser.error("--release-baseline requires the 0.4.11 wheel")

    eif_options = {
        "kind": "eif",
        "max_depth": 8,
        "extension_level": 1,
        "seed": 42,
        "contamination": 0.02,
    }
    if args.dataset == "bank":
        dataset, openml = "Bank_Customer_Churn", 46911
        features = [
            "country",
            "gender",
            "age",
            "tenure",
            "balance",
            "products_number",
            "credit_score",
        ]
        target = None
        train_count, test_start, test_count = 6000, 8000, args.rows or 600
        types = ["categorical", "categorical"] + ["numeric"] * 5
        models = [
            (AnomalyTree, eif_options, {"col_names": features, "dtypes": types}),
            (
                ClusterTree,
                {"kind": "pct", "max_depth": 2, "min_rows": 100},
                {"col_names": features, "dtypes": types},
            ),
        ]
    else:
        dataset, openml = "APSFailure", 46908
        # Use distinct sensor groups and retain columns with varied missing rates.
        features = [
            "aa_000",
            "ab_000",
            "ac_000",
            "ad_000",
            "ag_005",
            "ah_000",
            "an_000",
            "ao_000",
        ]
        target = "AirPressureSystemFailure"
        train_count = test_start = test_count = args.rows or 1200
        types = ["numeric"] * len(features)
        models = [
            (AnomalyTree, eif_options, {"col_names": features, "dtypes": types}),
            (
                AnomalyTree,
                {
                    "kind": "hst",
                    "max_depth": 6,
                    "window_size": 250,
                    "seed": 42,
                    "contamination": 0.02,
                },
                {},
            ),
            (
                ClusterTree,
                {"kind": "cltree", "max_depth": 4, "min_rows": 10},
                {"col_names": features},
            ),
        ]

    path = args.data or Path("datasets") / f"{dataset}.parquet"
    df = pd.read_parquet(path, columns=features + ([target] if target else []))
    if test_start + test_count > len(df):
        parser.error(f"{dataset} must have at least {test_start + test_count} rows")
    order = list(range(len(df)))
    Random(2026).shuffle(order)
    train_ids = order[:train_count]
    test_ids = order[test_start : test_start + test_count]
    assert len(set(train_ids + test_ids)) == len(train_ids) + len(test_ids)

    train, test = [
        [
            [None if pd.isna(value) else value for value in row]
            for row in df.iloc[ids][features].itertuples(index=False, name=None)
        ]
        for ids in (train_ids, test_ids)
    ]
    labels = (df.iloc[test_ids][target] == "pos").tolist() if target else None
    missing = [
        {name: sum(row[j] is None for row in rows) for j, name in enumerate(features)}
        for rows in (train, test)
    ]
    if target:
        assert all(sum(counts.values()) > 0 for counts in missing), (
            "Expected natural missing values in APS"
        )
    binary = Path(_core.__file__)
    print(
        json.dumps(
            {
                "dataset": dataset,
                "openml_id": openml,
                "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "features": features,
                "train_count": train_count,
                "test_count": test_count,
                "split_sha256": digest([train_ids, test_ids]),
                "input_sha256": digest([train, test]),
                "train_missing": missing[0],
                "test_missing": missing[1],
                "test_positive_count": sum(labels) if labels is not None else None,
                "build": "release" if args.release_baseline else "branch",
                "package_version": version("mltreelib"),
                "extension_file": binary.name,
                "extension_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": version("numpy"),
                "pyarrow": pyarrow.__version__,
                "threads": {
                    name: os.getenv(name)
                    for name in (
                        "RAYON_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                },
                "models": [
                    run_model(
                        factory,
                        options,
                        train,
                        test,
                        kwargs,
                        labels,
                        args.reps,
                        args.release_baseline,
                    )
                    for factory, options, kwargs in models
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def average_precision(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    found = 0
    total = 0.0
    rank = 0
    while rank < len(order):
        end = rank + 1
        while end < len(order) and scores[order[end]] == scores[order[rank]]:
            end += 1
        group_positive = sum(labels[index] for index in order[rank:end])
        found += group_positive
        total += group_positive * (found / end)
        rank = end
    return total / positives


if __name__ == "__main__":
    main()
