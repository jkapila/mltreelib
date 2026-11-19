"""Exotic Statistical and Metric Splitting Criteria Showcase (Spec 12 & Beyond).

Demonstrates all statistical and metric splitting criteria evaluated on the
SAME unified benchmark dataset for the SAME task, enabling direct head-to-head
comparison across criteria and seamless addition of future phases.

Phases Covered:
- Baseline: Standard CART (Variance Reduction)
- Phase 12A: Model-Based Recursive Partitioning (MOB / Zeileis et al. 2008)
- Phase 12B: Conditional Inference Framework (CTree / Hothorn et al. 2006)
- Phase 12C: Metric & Kernel Splitting (Energy Distance & MMD / Gretton et al. 2012)
- Phase 12D.1: Wasserstein-2 Optimal Transport Splitting (Villani 2003, Agueh & Carlier 2011)
- Phase 12D.2: Fairness-Constrained Splitting (Demographic Parity Bounds)

Zero external dependencies: uses Python standard library and mltreelib.
"""

import math
import random
from typing import Any, Callable, Dict, List, Tuple

from mltreelib import ModelTree, distributions_to_quantiles


def generate_unified_benchmark_data(
    n: int = 300, seed: int = 42
) -> Dict[str, Any]:
    """Generate a single unified financial risk & portfolio dataset.

    Contains continuous features with structural breaks, high-cardinality
    noise factors, binary signal factors, protected attributes with proxy features,
    and multiple target modalities (scalar, multi-output vector, quantile curves).
    """
    rng = random.Random(seed)

    X: List[List[Any]] = []
    y_scalar: List[float] = []
    y_rare: List[float] = []
    y_quantiles: List[List[float]] = []
    y_vectors: List[List[float]] = []

    for _ in range(n):
        # 1. Continuous risk score in [-2.0, 2.0] with structural break at 0.0
        score = rng.uniform(-2.0, 2.0)

        # 2. High-cardinality noise categorical feature: 50 zip codes (pure noise)
        zip_code = float(rng.randint(0, 49))

        # 3. Low-cardinality binary signal feature (real predictive signal)
        tier_flag = 1.0 if rng.random() < 0.5 else 0.0

        # 4. Protected demographic attribute (e.g., gender in {0, 1})
        gender = 1.0 if rng.random() < 0.5 else 0.0

        # 5. Unfair proxy feature: heavily correlated with protected attribute
        gender_proxy = gender + rng.gauss(0.0, 0.05)

        # 6. Fair financial merit feature (clean independent score)
        fair_merit = rng.uniform(0.0, 100.0)

        # Underlying regime:
        # If score < 0.0: negative baseline, negative slope
        # If score >= 0.0: positive baseline, positive slope
        if score < 0.0:
            regime_center = -2.0
            slope = -1.5
            dist_spread = 0.4
        else:
            regime_center = 2.0
            slope = 1.5
            dist_spread = 0.8

        # Merit outcome component:
        merit_component = 1.0 if fair_merit > 50.0 else 0.0

        # Scalar response (used by CART, MOB, CTree, Fair Tree):
        # Incorporates structural break, tier signal, and proxy correlation
        y_val = (
            regime_center
            + slope * score
            + 2.5 * tier_flag
            + 3.0 * gender
            + merit_component
            + rng.gauss(0.0, 0.3)
        )

        # Distribution target: 20 quantile evaluations of future return distribution
        raw_samples = [
            regime_center + slope * score + rng.gauss(0.0, dist_spread)
            for _ in range(40)
        ]
        q_curve = distributions_to_quantiles([raw_samples], n_quantiles=20)[0]

        # Multi-output vector target: 5-dimensional risk vector embedding
        vec_target = [
            regime_center,
            slope * score,
            dist_spread,
            regime_center * 0.5 + slope,
            regime_center - slope * 0.5,
        ]

        # Rare tail catastrophic default event (approx 5% prevalence)
        rare_event = 1.0 if score < -1.4 else 0.0

        X.append([score, zip_code, tier_flag, gender, gender_proxy, fair_merit])
        y_scalar.append(y_val)
        y_rare.append(rare_event)
        y_quantiles.append(q_curve)
        y_vectors.append(vec_target)

    col_names = [
        "score",
        "zip_code",
        "tier_flag",
        "gender",
        "gender_proxy",
        "fair_merit",
    ]
    dtypes = [
        "numeric",
        "categorical",
        "categorical",
        "categorical",
        "numeric",
        "numeric",
    ]

    return {
        "X": X,
        "y_scalar": y_scalar,
        "y_rare": y_rare,
        "y_quantiles": y_quantiles,
        "y_vectors": y_vectors,
        "col_names": col_names,
        "dtypes": dtypes,
    }


def run_phase_baseline_cart(data: Dict[str, Any]) -> Dict[str, Any]:
    """Baseline: Standard CART with Variance Reduction (gain='sse')."""
    tree = ModelTree(max_depth=1, min_rows=15, gain="sse", prune="none")
    tree.fit(
        data["X"],
        data["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    return {
        "phase": "Baseline",
        "name": "Standard CART (SSE)",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "note": "Standard variance reduction; susceptible to cardinality selection bias.",
    }


def run_phase_12a_mob(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12A: Model-Based Recursive Partitioning (MOB / Zeileis et al. 2008)."""
    tree = ModelTree(
        gain="mob",
        split_test="mob",
        max_depth=1,
        min_rows=15,
        leaf="linear",
        prune="none",
    )
    tree.fit(
        data["X"],
        data["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    diag = tree.split_diagnostics()
    p_val = diag[0].get("p_value", 0.0) if diag and diag[0] else 0.0
    return {
        "phase": "Phase 12A",
        "name": "MOB (Parameter Instability)",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "p_value": p_val,
        "note": f"Score residuals supLM test; detected instability (p={p_val:.4e}).",
    }


def run_phase_12b_ctree(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12B: Conditional Inference Trees (CTree / Hothorn et al. 2006)."""
    tree = ModelTree(
        gain="ctree",
        split_test="ctree",
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["X"],
        data["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    diag = tree.split_diagnostics()
    p_val = diag[0].get("p_value", 0.0) if diag and diag[0] else 0.0
    return {
        "phase": "Phase 12B",
        "name": "CTree (Permutation Rank)",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "p_value": p_val,
        "note": "Strictly unbiased variable selection; immune to 50-level cardinality noise.",
    }


def run_phase_12c_metric_energy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12C: Energy Distance Metric Splitting on Vector Targets."""
    tree = ModelTree(
        gain="energy_distance",
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["X"],
        data["y_vectors"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    return {
        "phase": "Phase 12C.1",
        "name": "Energy Distance (Székely)",
        "target_type": "5D Vector",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "note": "Non-parametric metric splitting on multi-output vector embeddings.",
    }


def run_phase_12c_metric_mmd(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12C: Maximum Mean Discrepancy (MMD) with RBF Kernel."""
    tree = ModelTree(
        gain="mmd",
        mmd_kernel="rbf",
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["X"],
        data["y_vectors"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    return {
        "phase": "Phase 12C.2",
        "name": "MMD (Kernel RKHS)",
        "target_type": "5D Vector",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "note": "Reproducing Kernel Hilbert Space distance; distinguishes higher-moment regimes.",
    }


def run_phase_12d1_wasserstein(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12D.1: Wasserstein-2 Optimal Transport Splitting on Distribution Curves."""
    tree = ModelTree(
        gain="wasserstein",
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["X"],
        data["y_quantiles"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    # Sample vector prediction
    test_x = [data["X"][0]]
    preds = tree.predict_vector(test_x)
    pred_len = len(preds[0]) if preds else 0
    return {
        "phase": "Phase 12D.1",
        "name": "Wasserstein-2 Optimal Transport",
        "target_type": "20-Quantile Curve",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "note": f"Fréchet barycenter dispersion reduction; predicts {pred_len}-point quantile distributions.",
    }


def run_phase_12d2_fairness(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12D.2: Fairness-Constrained Splitting with Demographic Parity Bounds."""
    # Fit tree with strict fairness constraint on gender (col 3)
    tree = ModelTree(
        gain="fair",
        fairness_penalty=1.0,
        protected_col=3,
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["X"],
        data["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}

    # Measure demographic parity gap on predictions
    preds = tree.predict(data["X"])
    g1 = [
        preds[i] for i, row in enumerate(data["X"]) if row[3] == 1.0
    ]
    g0 = [
        preds[i] for i, row in enumerate(data["X"]) if row[3] == 0.0
    ]
    gap = abs((sum(g1) / len(g1)) - (sum(g0) / len(g0))) if g1 and g0 else 0.0

    return {
        "phase": "Phase 12D.2",
        "name": "Demographic Parity Fair Tree",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "dp_gap": gap,
        "note": f"Penalized disparity; demographic parity gap constrained to {gap:.3f}.",
    }


def run_phase_hellinger(data: Dict[str, Any]) -> Dict[str, Any]:
    """Hellinger Distance Splitting for Extreme Imbalance (Cieslak & Chawla 2008)."""
    tree = ModelTree(gain="hellinger", max_depth=1, min_rows=5, prune="none")
    tree.fit(data["X"], data["y_rare"], col_names=data["col_names"], dtypes=data["dtypes"])
    root = tree.root_split() or {}
    return {
        "phase": "Phase 12E.1",
        "name": "Hellinger Distance (Skew Invariant)",
        "target_type": "Binary Rare",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "note": "Skew-invariant divergence; isolates rare catastrophic tail risk without score collapse.",
    }


def run_phase_maxstat(data: Dict[str, Any]) -> Dict[str, Any]:
    """Maximally Selected Rank Statistics (Hothorn & Lausen 2003)."""
    tree = ModelTree(gain="maxstat", max_depth=1, min_rows=15, prune="none")
    tree.fit(data["X"], data["y_scalar"], col_names=data["col_names"], dtypes=data["dtypes"])
    root = tree.root_split() or {}
    return {
        "phase": "Phase 12E.2",
        "name": "MaxStat Rank Statistic",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "note": "Standardized two-sample linear rank divergence; invariant to monotone target scalings.",
    }


def run_phase_sporf(data: Dict[str, Any]) -> Dict[str, Any]:
    """SPORF: Sparse Projection Oblique Randomer Forest Splitting (Tomita et al. 2020)."""
    tree = ModelTree(sporf_d=2, sporf_projections=30, max_depth=1, min_rows=15, seed=42, prune="none")
    tree.fit(data["X"], data["y_scalar"], col_names=data["col_names"], dtypes=data["dtypes"])
    root = tree.root_split() or {}
    ob_eq = root.get("oblique", {}).get("equation", "univariate")
    return {
        "phase": "Phase 12F",
        "name": "SPORF Oblique Projection",
        "target_type": "Scalar",
        "split_col": root.get("col", "None") if "oblique" not in root else f"{ob_eq}",
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "note": f"Evaluates random linear combinations of 2 features; hyperplane: {ob_eq}.",
    }


# Extensible Phase Registry — add new phases here as they are developed
BENCHMARK_PHASES: List[
    Tuple[str, str, Callable[[Dict[str, Any]], Dict[str, Any]]]
] = [
    ("Baseline", "Standard CART (SSE)", run_phase_baseline_cart),
    ("Phase 12A", "MOB (Parameter Instability)", run_phase_12a_mob),
    ("Phase 12B", "CTree (Permutation Rank)", run_phase_12b_ctree),
    ("Phase 12C.1", "Energy Distance Metric Splitting", run_phase_12c_metric_energy),
    ("Phase 12C.2", "MMD Kernel Splitting", run_phase_12c_metric_mmd),
    ("Phase 12D.1", "Wasserstein-2 Optimal Transport", run_phase_12d1_wasserstein),
    ("Phase 12D.2", "Demographic Parity Fair Splitting", run_phase_12d2_fairness),
    ("Phase 12E.1", "Hellinger Distance (Skew Invariant)", run_phase_hellinger),
    ("Phase 12E.2", "MaxStat Rank Splitting", run_phase_maxstat),
    ("Phase 12F", "SPORF Oblique Splitting", run_phase_sporf),
]


def main():
    print("=" * 86)
    print("  MLTREELIB: EXOTIC STATISTICAL AND METRIC SPLITTING CRITERIA BENCHMARK SUITE")
    print("  Unified Task: Credit Portfolio Risk Valuation on Single Shared Dataset")
    print("=" * 86)

    data = generate_unified_benchmark_data(n=300, seed=42)
    print(f"Dataset summary:")
    print(f"  Observations:  {len(data['X'])} rows")
    print(f"  Features ({len(data['col_names'])}): {data['col_names']}")
    print(f"  Data Types:    {data['dtypes']}")
    print(f"  Target Modes:  Scalar (Loss/Return), 5D Embedding Vector, 20-Point Quantile Curve")
    print("-" * 86)

    results: List[Dict[str, Any]] = []

    for phase_id, phase_label, runner_fn in BENCHMARK_PHASES:
        print(f"\n>>> Running {phase_id}: {phase_label}...")
        res = runner_fn(data)
        results.append(res)
        thresh_str = (
            f"{res['threshold']:.3f}"
            if isinstance(res["threshold"], float)
            else str(res["threshold"])
        )
        print(f"    Split Chosen:   Feature '{res['split_col']}' at threshold {thresh_str}")
        print(f"    Objective Gain: {res['gain']:.4f}")
        print(f"    Theoretical:    {res['note']}")

    print("\n" + "=" * 86)
    print("  UNIFIED BENCHMARK COMPARATIVE SUMMARY (SAME DATA / SAME TASK)")
    print("=" * 86)
    header = f"{'Phase':<12} | {'Criterion / Model':<32} | {'Target':<10} | {'Split':<12} | {'Gain':<10}"
    print(header)
    print("-" * 86)

    for r in results:
        col_str = str(r["split_col"])
        gain_str = f"{r['gain']:.2f}"
        print(
            f"{r['phase']:<12} | {r['name']:<32} | {r['target_type']:<10} | {col_str:<12} | {gain_str:<10}"
        )

    print("=" * 86)
    print("\nKey Takeaways:")
    print("1. Standard CART: Greedily splits solely on empirical variance reduction.")
    print("2. MOB (12A): Uses parameter instability tests on score residuals, detecting structural breaks.")
    print("3. CTree (12B): Completely unbiased against high-cardinality noise via permutation linear ranks.")
    print("4. Metric Criteria (12C): Optimizes non-parametric multi-target embeddings without distributions.")
    print("5. Wasserstein-2 (12D.1): Directly splits and predicts entire continuous distribution curves.")
    print("6. Fair Splitting (12D.2): Restricts disparate impact on protected attributes via TV penalties.")
    print("\n[Extensibility]: New phases (Spec 13+) can be registered directly into BENCHMARK_PHASES.")


if __name__ == "__main__":
    main()
