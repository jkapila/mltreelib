"""Fetch the benchmark dataset suite into the repository dataset folder.

    uv run python scripts/fetch_datasets.py
    uv run python scripts/fetch_datasets.py --only diabetes airfoil_self_noise
    uv run python scripts/fetch_datasets.py --list

Files land in `datasets/<name>.parquet` (CSV sources such as `flchain_rdatasets`
land as `datasets/<name>.csv`). The folder is not tracked in git, so
run this script once per checkout before running the dataset examples or the
real-data tests. One dataset can also be fetched on demand through
`mltreelib.datasets.load_dataset(name)`, which caches it under
`~/.cache/mltreelib/datasets/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mltreelib.datasets import DATASET_REGISTRY, download_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = REPO_ROOT / "datasets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="dataset names to fetch. Defaults to every registered dataset.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET,
        help="folder that receives the Parquet files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the registered dataset names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for name in sorted(DATASET_REGISTRY):
            print(name)
        return

    names = args.only if args.only else sorted(DATASET_REGISTRY)
    unknown = [name for name in names if name not in DATASET_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown dataset names: {', '.join(unknown)}")

    args.target_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, name in enumerate(names, start=1):
        present = any((args.target_dir / f"{name}{ext}").exists() for ext in (".parquet", ".csv"))
        if present:
            print(f"[{index}/{len(names)}] {name}: present")
            continue
        try:
            path = download_dataset(name, target_dir=args.target_dir)
        except Exception as error:  # noqa: BLE001 - report and continue
            failures.append((name, str(error)))
            print(f"[{index}/{len(names)}] {name}: failed ({error})")
            continue
        print(f"[{index}/{len(names)}] {name}: {path.stat().st_size} bytes")

    if failures:
        print(f"\n{len(failures)} dataset(s) failed to download", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nfetched {len(names)} dataset(s) into {args.target_dir}")


if __name__ == "__main__":
    main()
