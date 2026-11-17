"""Survival Analysis & Kaplan-Meier Tree Benchmark.

Demonstrates:
1. Ingestion of 2-tuple survival targets: y = (times, events) where event=1 (death/failure), event=0 (censored)
2. Fitting non-parametric Nelson-Aalen and Kaplan-Meier estimators at tree leaves
3. Evaluating on the classic Veterans' Administration Lung Cancer study (Prentice 1973 / Kalbfleisch & Prentice 2002)
4. Predicting survival curves S(t) and cumulative hazard Lambda(t) via unified predict(X, kind="survival"|"hazard")
5. Rigorous mathematical validation: asserts that tree S(t) exactly matches the manual empirical product-limit estimator.

Run:
    PYTHONPATH=python python3 examples/survival_benchmark.py
"""

from __future__ import annotations

import csv

from mltreelib import ModelTree


def load_veterans_dataset() -> tuple[
    list[list[float]], list[float], list[float], list[str]
]:
    """Load Veterans' Administration Lung Cancer trial dataset."""
    dataset_path = "datasets/external/veteran.csv"
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Features: karno (Karnofsky performance scale 0-100), age, diagtime (months from diagnosis), prior
    cols = ["karno", "age", "diagtime", "prior"]
    X = [[float(r[c]) for c in cols] for r in rows]
    times = [float(r["time"]) for r in rows]
    events = [float(r["status"]) for r in rows]  # 1 = dead, 0 = censored

    return X, times, events, cols


def compute_manual_kaplan_meier(
    times: list[float], events: list[float], eval_grid: list[float]
) -> list[float]:
    """Compute empirical Kaplan-Meier S(t) directly from formula to validate tree outputs.

    S(t) = prod_{t_j <= t} (1 - d_j / n_j)
    """
    records = sorted(zip(times, events), key=lambda r: r[0])
    distinct_event_times = sorted(list({t for t, e in records if e > 0.0}))

    survival_curve = []
    current_s = 1.0

    # Calculate S(t) across distinct event times
    km_at_events: dict[float, float] = {}
    for tj in distinct_event_times:
        nj = sum(1.0 for ti, _ in records if ti >= tj)
        dj = sum(1.0 for ti, ei in records if abs(ti - tj) < 1e-9 and ei > 0.0)
        if nj > 0.0:
            current_s *= 1.0 - dj / nj
        km_at_events[tj] = current_s

    # Map onto eval_grid (step function)
    grid_s = []
    for t_eval in eval_grid:
        valid_times = [tj for tj in distinct_event_times if tj <= t_eval]
        if not valid_times:
            grid_s.append(1.0)
        else:
            grid_s.append(km_at_events[max(valid_times)])

    return grid_s


def validate_survival_math(
    tree: ModelTree,
    X: list[list[float]],
    times: list[float],
    events: list[float],
    t_grid: list[float],
) -> None:
    """Validate numerical correctness: assert that tree predicted S(t) matches manual Kaplan-Meier."""
    leaf_ids = tree.predict(X, kind="leaf_id")

    # Group rows by leaf
    leaf_rows: dict[int, list[int]] = {}
    for row_idx, lid in enumerate(leaf_ids):
        leaf_rows.setdefault(int(lid), []).append(row_idx)

    for lid, row_indices in leaf_rows.items():
        sub_times = [times[r] for r in row_indices]
        sub_events = [events[r] for r in row_indices]

        manual_s = compute_manual_kaplan_meier(sub_times, sub_events, t_grid)
        tree_s = tree.predict([X[row_indices[0]]], kind="survival", times=t_grid)[0]

        for idx, (m_val, t_val) in enumerate(zip(manual_s, tree_s)):
            diff = abs(m_val - t_val)
            assert diff < 1e-9, (
                f"Leaf {lid} time {t_grid[idx]}: manual S={m_val}, tree S={t_val}, diff={diff}"
            )

    print(
        "  [Pass] Exact Kaplan-Meier Assertion: Tree S(t) matches manual product-limit estimator within 1e-9."
    )


def run_benchmark():
    print("=" * 70)
    print("1. VETERANS' LUNG CANCER CLINICAL SURVIVAL BENCHMARK")
    print("=" * 70)

    X, times, events, cols = load_veterans_dataset()
    censored_count = sum(1 for e in events if e == 0.0)
    print(
        f"Total Patients: {len(X)} | Deaths (Events): {len(X) - censored_count} | Censored: {censored_count} ({censored_count / len(X) * 100:.1f}%)"
    )
    print(
        f"Follow-up Time: Min={min(times):.0f} days, Median={sorted(times)[len(times) // 2]:.0f} days, Max={max(times):.0f} days"
    )

    # Fit a survival tree on target tuple y = (times, events)
    tree = ModelTree(max_depth=1, min_rows=25, leaf_family="survival")
    tree.fit(X, (times, events), col_names=cols)

    t_grid = [10.0, 30.0, 60.0, 100.0, 180.0, 365.0]
    print(f"\nFitted Survival Tree: {tree.n_leaves()} Prognostic Cohorts")

    probes = [
        [30.0, 60.0, 10.0, 0.0],  # Low Karnofsky performance score (poor prognosis)
        [80.0, 60.0, 10.0, 0.0],  # High Karnofsky performance score (better prognosis)
    ]
    s_curves = tree.predict(probes, kind="survival", times=t_grid)
    h_curves = tree.predict(probes, kind="hazard", times=t_grid)

    print("\nPredicted Survival Probability S(t):")
    print(f"{'Cohort':25s} | " + " | ".join(f"t={t:.0f}d" for t in t_grid))
    print("-" * 75)
    print(
        f"{'Poor Prognosis (Karno=30)':25s} | "
        + " | ".join(f"{s:.3f}" for s in s_curves[0])
    )
    print(
        f"{'Good Prognosis (Karno=80)':25s} | "
        + " | ".join(f"{s:.3f}" for s in s_curves[1])
    )

    print("\nPredicted Cumulative Hazard Lambda(t):")
    print(
        f"{'Poor Prognosis (Karno=30)':25s} | "
        + " | ".join(f"{h:.3f}" for h in h_curves[0])
    )
    print(
        f"{'Good Prognosis (Karno=80)':25s} | "
        + " | ".join(f"{h:.3f}" for h in h_curves[1])
    )

    # Verify survival monotonicity: S(t) must be non-increasing in t
    for s_curve in s_curves:
        for i in range(len(s_curve) - 1):
            assert s_curve[i] >= s_curve[i + 1] - 1e-12, (
                "Survival S(t) must be non-increasing"
            )

    validate_survival_math(tree, X, times, events, t_grid)


if __name__ == "__main__":
    run_benchmark()
