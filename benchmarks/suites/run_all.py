"""Run all 11 mltreelib core benchmark suites.

Executes each suite sequentially and reports status, latency, and assets generated.
Usage:
    uv run python benchmarks/suites/run_all.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path


SUITES = [
    ("Suite 1", "Core Tree Architecture", "core_tree_architecture.py"),
    ("Suite 2", "Structural Constraints & Honesty", "constraints_and_honesty.py"),
    ("Suite 3", "Missing Data Routing", "missing_routing_benchmark.py"),
    ("Suite 4", "Distributional & Censored Leaves", "distributional_leaf_benchmarks.py"),
    ("Suite 5", "Hierarchical Leaf Retrain", "stump_leaf_retrain.py"),
    ("Suite 6", "Statistical & Metric Splitting", "statistical_and_metric_splitting.py"),
    ("Suite 7", "Time Series Forecasting", "timeseries_additive_forecasting.py"),
    ("Suite 8", "Unsupervised Trees & Anomalies", "unsupervised_trees.py"),
    ("Suite 9", "Survival & Competing Risks", "survival_and_competing_risks.py"),
    ("Suite 10", "Solvers & Optimizers", "solver_benchmarks.py"),
    ("Suite 11", "Component & Factor Analysis", "component_benchmarks.py"),
]


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    suites_dir = root_dir / "benchmarks" / "suites"

    print("=" * 80)
    print("mltreelib Core Benchmark Suite Orchestrator")
    print(f"Running 11 benchmark suites sequentially from {suites_dir}")
    print("=" * 80)

    results = []
    t_start_total = time.perf_counter()

    for suite_id, title, filename in SUITES:
        script_path = suites_dir / filename
        if not script_path.exists():
            print(f"[MISSING] {suite_id}: {filename} not found!")
            results.append((suite_id, title, "MISSING", 0.0))
            continue

        print(f"\n>>> Running {suite_id}: {title} ({filename})...")
        t0 = time.perf_counter()
        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0

        if res.returncode == 0:
            print(f"    [PASSED] in {elapsed:.2f}s")
            results.append((suite_id, title, "PASSED", elapsed))
        else:
            print(f"    [FAILED] with exit code {res.returncode} in {elapsed:.2f}s")
            print("--- STDERR ---")
            print(res.stderr[-500:] if len(res.stderr) > 500 else res.stderr)
            results.append((suite_id, title, "FAILED", elapsed))

    total_time = time.perf_counter() - t_start_total

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Suite':<10} | {'Title':<35} | {'Status':<8} | {'Time (s)':>10}")
    print("-" * 80)
    for suite_id, title, status, elapsed in results:
        print(f"{suite_id:<10} | {title:<35} | {status:<8} | {elapsed:>10.2f}s")
    print("-" * 80)
    passed_count = sum(1 for _, _, s, _ in results if s == "PASSED")
    print(f"Total: {passed_count}/{len(SUITES)} passed in {total_time:.2f}s")
    print("=" * 80)

    if passed_count < len(SUITES):
        sys.exit(1)


if __name__ == "__main__":
    main()
