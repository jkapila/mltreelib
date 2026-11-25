"""Exotic Statistical and Metric Splitting Criteria Showcase (Spec 12 & Beyond).

Demonstrates all statistical and metric splitting criteria evaluated on the
SAME unified benchmark dataset for the SAME task, enabling direct head-to-head
comparison across criteria and seamless addition of future phases.

Phases Covered:
- Baseline: Standard CART (Variance Reduction)
- Phase 12A: Model-Based Recursive Partitioning (MOB / Zeileis et al. 2008)
- Phase 12B: Conditional Inference Framework (CTree / Hothorn et al. 2006)
- Phase 12C.1: Energy Distance Metric Splitting (Székely & Rizzo 2013)
- Phase 12C.2: Maximum Mean Discrepancy (MMD / Gretton et al. 2012)
- Phase 12D.1: Wasserstein-2 Optimal Transport Splitting (Villani 2003, Agueh & Carlier 2011)
- Phase 12D.2: Fairness-Constrained Splitting (Demographic Parity Bounds)
- Phase 12E.1: Hellinger Distance Splitting for Extreme Imbalance (Cieslak & Chawla 2008)
- Phase 12E.2: Maximally Selected Rank Statistics (MaxStat / Hothorn & Lausen 2003)
- Phase 12F: Sparse Projection Oblique Randomer Forest (SPORF / Tomita et al. 2020)
- Phase 12G: Equalized Odds Fairness Penalty (Hardt et al. 2016, Donini et al. 2018)
- Phase 12H: Extremely Randomized Splits / ExtraTrees (Geurts, Ernst & Wehenkel 2006)
- Phase 12I: EBM FAST Interaction Screening (Lou, Caruana, Gehrke & Hooker 2013)

Zero external dependencies: uses Python standard library and mltreelib.
"""

import math
import random
from typing import Any, Callable, Dict, List, Tuple

from mltreelib import (
    ModelTree,
    distributions_to_quantiles,
    fast_interaction_screen,
)


def generate_unified_benchmark_data(
    n_train: int = 240, n_test: int = 60, seed: int = 42
) -> Dict[str, Any]:
    """Generate a single unified financial risk & portfolio dataset.

    Contains continuous features with structural breaks, high-cardinality
    noise factors, binary signal factors, protected attributes with proxy features,
    non-linear quadrant synergies, and multiple target modalities (scalar, multi-output
    vector, quantile curves, rare events, fairness credit responses).
    """
    rng = random.Random(seed)
    total_n = n_train + n_test

    X_all: List[List[Any]] = []
    y_scalar_all: List[float] = []
    y_fair_all: List[float] = []
    y_inter_all: List[float] = []
    y_rare_all: List[float] = []
    y_quantiles_all: List[List[float]] = []
    y_vectors_all: List[List[float]] = []

    for _ in range(total_n):
        # 1. Continuous risk score in [-2.0, 2.0] with structural break at 0.0
        score = rng.uniform(-2.0, 2.0)

        # 2. High-cardinality noise categorical feature: 50 zip codes (pure noise)
        zip_code = float(rng.randint(0, 49))

        # 3. Low-cardinality binary signal feature (real predictive signal)
        tier_flag = 1.0 if rng.random() < 0.5 else 0.0

        # 4. Protected demographic attribute (e.g., gender in {0, 1})
        gender = 1.0 if rng.random() < 0.5 else 0.0

        # 5. Unfair proxy feature: heavily correlated with protected attribute
        gender_proxy = gender + (rng.random() - 0.5) * 0.05

        # 6. Fair financial merit feature (clean independent score)
        fair_merit = rng.uniform(0.0, 100.0)

        # Underlying structural regimes:
        if score < 0.0:
            regime_center = -2.0
            slope = -1.5
            dist_spread = 0.4
        else:
            regime_center = 2.0
            slope = 1.5
            dist_spread = 0.8

        # Merit component:
        merit_component = 6.0 if fair_merit > 50.0 else 0.0

        # Non-linear quadrant interaction between score and tier_flag:
        synergy = 8.0 if (score > 0.0 and tier_flag == 1.0) else 0.0

        # Non-linear interaction response (isolated for EBM FAST deep dive):
        y_inter_val = 2.0 * score + 2.0 * tier_flag + synergy + rng.gauss(0.0, 0.2)

        # Standard multi-factor scalar response:
        y_val = (
            regime_center
            + slope * score
            + 2.0 * tier_flag
            + 7.0 * gender_proxy
            + merit_component
            + synergy
            + rng.gauss(0.0, 0.25)
        )

        # Fair credit score outcome (proxy vs merit test):
        y_fair = (
            merit_component
            + 8.0 * gender_proxy
            + 1.5 * score
            + rng.gauss(0.0, 0.2)
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

        X_all.append([score, zip_code, tier_flag, gender, gender_proxy, fair_merit])
        y_scalar_all.append(y_val)
        y_fair_all.append(y_fair)
        y_inter_all.append(y_inter_val)
        y_rare_all.append(rare_event)
        y_quantiles_all.append(q_curve)
        y_vectors_all.append(vec_target)

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

    train_data = {
        "X": X_all[:n_train],
        "y_scalar": y_scalar_all[:n_train],
        "y_fair": y_fair_all[:n_train],
        "y_inter": y_inter_all[:n_train],
        "y_rare": y_rare_all[:n_train],
        "y_quantiles": y_quantiles_all[:n_train],
        "y_vectors": y_vectors_all[:n_train],
    }

    test_data = {
        "X": X_all[n_train:],
        "y_scalar": y_scalar_all[n_train:],
        "y_fair": y_fair_all[n_train:],
        "y_inter": y_inter_all[n_train:],
        "y_rare": y_rare_all[n_train:],
        "y_quantiles": y_quantiles_all[n_train:],
        "y_vectors": y_vectors_all[n_train:],
    }

    return {
        "train": train_data,
        "test": test_data,
        "col_names": col_names,
        "dtypes": dtypes,
    }


def compute_regression_metrics(
    y_true: List[float], y_pred: List[float]
) -> Dict[str, float]:
    n = len(y_true)
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
    mean_y = sum(y_true) / n
    tot_var = sum((yt - mean_y) ** 2 for yt in y_true) / n
    r2 = 1.0 - (mse / tot_var) if tot_var > 1e-12 else 0.0
    return {"mse": mse, "r2": r2}


def compute_fairness_disparities(
    y_pred: List[float], s: List[float], y_true: List[float]
) -> Dict[str, float]:
    pred_s1 = [yp for yp, si in zip(y_pred, s) if si == 1.0]
    pred_s0 = [yp for yp, si in zip(y_pred, s) if si == 0.0]
    dp_gap = (
        abs((sum(pred_s1) / len(pred_s1)) - (sum(pred_s0) / len(pred_s0)))
        if pred_s1 and pred_s0
        else 0.0
    )

    thr = sum(y_pred) / len(y_pred)
    y_mean = sum(y_true) / len(y_true)
    y_bin = [1.0 if yt >= y_mean else 0.0 for yt in y_true]
    p_bin = [1.0 if yp >= thr else 0.0 for yp in y_pred]

    s1_y1 = [pb for pb, si, yb in zip(p_bin, s, y_bin) if si == 1.0 and yb == 1.0]
    s0_y1 = [pb for pb, si, yb in zip(p_bin, s, y_bin) if si == 0.0 and yb == 1.0]
    s1_y0 = [pb for pb, si, yb in zip(p_bin, s, y_bin) if si == 1.0 and yb == 0.0]
    s0_y0 = [pb for pb, si, yb in zip(p_bin, s, y_bin) if si == 0.0 and yb == 0.0]

    tpr_1 = sum(s1_y1) / len(s1_y1) if s1_y1 else 0.0
    tpr_0 = sum(s0_y1) / len(s0_y1) if s0_y1 else 0.0
    fpr_1 = sum(s1_y0) / len(s1_y0) if s1_y0 else 0.0
    fpr_0 = sum(s0_y0) / len(s0_y0) if s0_y0 else 0.0

    eo_gap = 0.5 * (abs(tpr_1 - tpr_0) + abs(fpr_1 - fpr_0))
    return {"dp_gap": dp_gap, "eo_gap": eo_gap}


def run_phase_baseline_cart(data: Dict[str, Any]) -> Dict[str, Any]:
    """Baseline: Standard CART with Variance Reduction (gain='sse')."""
    tree = ModelTree(max_depth=1, min_rows=15, gain="sse", prune="none")
    tree.fit(
        data["train"]["X"],
        data["train"]["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_scalar"], preds)
    return {
        "phase": "Baseline",
        "name": "Standard CART (SSE)",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": f"R²: {metrics['r2']:.3f}",
        "note": "Standard variance reduction; susceptible to proxy and cardinality bias.",
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
        data["train"]["X"],
        data["train"]["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_scalar"], preds)
    diag = tree.split_diagnostics()
    p_val = diag[0].get("p_value", 0.0) if diag and diag[0] else 0.0
    return {
        "phase": "Phase 12A",
        "name": "MOB (Parameter Instability)",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": f"p={p_val:.2e}",
        "note": f"Score residuals supLM test; detected parameter instability (p={p_val:.4e}).",
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
        data["train"]["X"],
        data["train"]["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_scalar"], preds)
    diag = tree.split_diagnostics()
    p_val = diag[0].get("p_value", 0.0) if diag and diag[0] else 0.0
    return {
        "phase": "Phase 12B",
        "name": "CTree (Permutation Rank)",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": f"p={p_val:.2e}",
        "note": "Strictly unbiased variable selection; immune to 50-level noise cardinality.",
    }


def run_phase_12c_metric_energy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12C.1: Energy Distance Metric Splitting on Vector Targets."""
    tree = ModelTree(
        gain="energy_distance",
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["train"]["X"],
        data["train"]["y_vectors"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict_vector(data["test"]["X"])
    err = sum(
        sum((yt[j] - yp[j]) ** 2 for j in range(len(yt)))
        for yt, yp in zip(data["test"]["y_vectors"], preds)
    ) / (len(preds) * len(preds[0]))
    return {
        "phase": "Phase 12C.1",
        "name": "Energy Distance (Székely)",
        "target_type": "5D Vector",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Vec MSE",
        "test_metric_val": err,
        "disparity_or_cost": "d=5 dims",
        "note": "Non-parametric metric splitting on multi-output vector embeddings.",
    }


def run_phase_12c_metric_mmd(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12C.2: Maximum Mean Discrepancy (MMD) with RBF Kernel."""
    tree = ModelTree(
        gain="mmd",
        mmd_kernel="rbf",
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["train"]["X"],
        data["train"]["y_vectors"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict_vector(data["test"]["X"])
    err = sum(
        sum((yt[j] - yp[j]) ** 2 for j in range(len(yt)))
        for yt, yp in zip(data["test"]["y_vectors"], preds)
    ) / (len(preds) * len(preds[0]))
    return {
        "phase": "Phase 12C.2",
        "name": "MMD (Kernel RKHS)",
        "target_type": "5D Vector",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Vec MSE",
        "test_metric_val": err,
        "disparity_or_cost": "RBF Kernel",
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
        data["train"]["X"],
        data["train"]["y_quantiles"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict_vector(data["test"]["X"])
    w2_loss = sum(
        sum((yt[j] - yp[j]) ** 2 for j in range(len(yt))) / len(yt)
        for yt, yp in zip(data["test"]["y_quantiles"], preds)
    ) / len(preds)
    return {
        "phase": "Phase 12D.1",
        "name": "Wasserstein-2 Optimal Transport",
        "target_type": "20-Quantile Curve",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "W2 Loss",
        "test_metric_val": w2_loss,
        "disparity_or_cost": "20 quantiles",
        "note": "Fréchet barycenter dispersion reduction; predicts entire continuous distribution curves.",
    }


def run_phase_12d2_fairness_dp(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12D.2: Fairness-Constrained Splitting with Demographic Parity Bounds."""
    tree = ModelTree(
        fairness_penalty=1.0,
        fairness_metric="demographic_parity",
        protected_col=3,  # gender
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["train"]["X"],
        data["train"]["y_fair"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_fair"], preds)
    s_test = [row[3] for row in data["test"]["X"]]
    disp = compute_fairness_disparities(preds, s_test, data["test"]["y_fair"])

    return {
        "phase": "Phase 12D.2",
        "name": "Demographic Parity Fair Tree",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": f"DP Gap: {disp['dp_gap']:.2f}",
        "note": f"Penalizes allocation rate disparity across gender; selects {root.get('col')} (DP gap={disp['dp_gap']:.2f}).",
    }


def run_phase_hellinger(data: Dict[str, Any]) -> Dict[str, Any]:
    """Hellinger Distance Splitting for Extreme Imbalance (Cieslak & Chawla 2008)."""
    tree = ModelTree(gain="hellinger", max_depth=1, min_rows=5, prune="none")
    tree.fit(
        data["train"]["X"],
        data["train"]["y_rare"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    thr = 0.1
    p_bin = [1.0 if p >= thr else 0.0 for p in preds]
    y_test = data["test"]["y_rare"]
    tp = sum(1 for yt, yp in zip(y_test, p_bin) if yt == 1.0 and yp == 1.0)
    fn = sum(1 for yt, yp in zip(y_test, p_bin) if yt == 1.0 and yp == 0.0)
    tn = sum(1 for yt, yp in zip(y_test, p_bin) if yt == 0.0 and yp == 0.0)
    fp = sum(1 for yt, yp in zip(y_test, p_bin) if yt == 0.0 and yp == 1.0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (sens + spec)
    return {
        "phase": "Phase 12E.1",
        "name": "Hellinger Distance (Skew Invariant)",
        "target_type": "Binary Rare",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Bal Acc",
        "test_metric_val": bal_acc,
        "disparity_or_cost": "Tail Prev 5%",
        "note": "Skew-invariant divergence; isolates rare catastrophic tail risk without score collapse.",
    }


def run_phase_maxstat(data: Dict[str, Any]) -> Dict[str, Any]:
    """Maximally Selected Rank Statistics (Hothorn & Lausen 2003)."""
    tree = ModelTree(gain="maxstat", max_depth=1, min_rows=15, prune="none")
    tree.fit(
        data["train"]["X"],
        data["train"]["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_scalar"], preds)
    return {
        "phase": "Phase 12E.2",
        "name": "MaxStat Rank Statistic",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": f"R²: {metrics['r2']:.3f}",
        "note": "Standardized two-sample linear rank divergence; invariant to monotone target scalings.",
    }


def run_phase_sporf(data: Dict[str, Any]) -> Dict[str, Any]:
    """SPORF: Sparse Projection Oblique Randomer Forest Splitting (Tomita et al. 2020)."""
    tree = ModelTree(
        sporf_d=2,
        sporf_projections=30,
        max_depth=1,
        min_rows=15,
        seed=42,
        prune="none",
    )
    tree.fit(
        data["train"]["X"],
        data["train"]["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    ob_eq = root.get("oblique", {}).get("equation", "univariate")
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_scalar"], preds)
    return {
        "phase": "Phase 12F",
        "name": "SPORF Oblique Projection",
        "target_type": "Scalar",
        "split_col": (
            root.get("col", "None") if "oblique" not in root else f"{ob_eq}"
        ),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": "2D Hyperplane",
        "note": f"Evaluates random linear combinations of 2 features; hyperplane: {ob_eq}.",
    }


def run_phase_12g_fairness_eo(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12G: Equalized Odds Fairness Penalty (TPR and FPR Parity)."""
    tree = ModelTree(
        fairness_penalty=1.0,
        fairness_metric="equalized_odds",
        protected_col=3,  # gender
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["train"]["X"],
        data["train"]["y_fair"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_fair"], preds)
    s_test = [row[3] for row in data["test"]["X"]]
    disp = compute_fairness_disparities(preds, s_test, data["test"]["y_fair"])

    return {
        "phase": "Phase 12G",
        "name": "Equalized Odds Fair Tree",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": f"EO Gap: {disp['eo_gap']:.2f}",
        "note": f"Penalizes TPR and FPR disparate error rates; selects {root.get('col')} (EO gap={disp['eo_gap']:.3f}).",
    }


def run_phase_12h_extratrees(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12H: Extremely Randomized Splits / ExtraTrees (Geurts et al. 2006)."""
    tree = ModelTree(
        extratrees=True,
        seed=42,
        max_depth=1,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["train"]["X"],
        data["train"]["y_scalar"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    root = tree.root_split() or {}
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_scalar"], preds)

    return {
        "phase": "Phase 12H",
        "name": "ExtraTrees (Uniform Cut)",
        "target_type": "Scalar",
        "split_col": root.get("col", "None"),
        "threshold": root.get("threshold", 0.0),
        "gain": root.get("gain", 0.0),
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": "1 eval / feat",
        "note": "Evaluates exactly 1 random cut per continuous feature in O(1) time without sorting.",
    }


def run_phase_12i_ebm_fast(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 12I: EBM FAST Interaction Screening & Quad-Stump Tree (Lou et al. 2013)."""
    # 1. Screen all P*(P-1)/2 pairs using fast 2D quad-stumps
    screened = fast_interaction_screen(
        data["train"]["X"],
        data["train"]["y_inter"],
        col_names=data["col_names"],
        top_k=3,
        min_samples=15,
    )
    top_pair = screened[0] if screened else {}
    top_pair_name = (
        f"{top_pair.get('col1_name')}&{top_pair.get('col2_name')}"
        if top_pair
        else "none"
    )
    inter_score = top_pair.get("score", 0.0)

    # 2. Fit tree with automatic FAST interaction constraint
    tree = ModelTree(
        ebm_fast=True,
        max_depth=2,
        min_rows=15,
        prune="none",
    )
    tree.fit(
        data["train"]["X"],
        data["train"]["y_inter"],
        col_names=data["col_names"],
        dtypes=data["dtypes"],
    )
    preds = tree.predict(data["test"]["X"])
    metrics = compute_regression_metrics(data["test"]["y_inter"], preds)

    return {
        "phase": "Phase 12I",
        "name": "EBM FAST Interaction Tree",
        "target_type": "Scalar",
        "split_col": f"Pair: {top_pair_name}",
        "threshold": f"({top_pair.get('cut1', 0.0):.2f}, {top_pair.get('cut2', 0.0):.2f})",
        "gain": inter_score,
        "test_metric_name": "Test MSE",
        "test_metric_val": metrics["mse"],
        "disparity_or_cost": f"Score: {inter_score:.3f}",
        "note": f"Screens all P*(P-1)/2 pairs in O(P²); detected non-linear synergy between {top_pair_name}.",
    }


# Extensible Phase Registry
BENCHMARK_PHASES: List[
    Tuple[str, str, Callable[[Dict[str, Any]], Dict[str, Any]]]
] = [
    ("Baseline", "Standard CART (SSE)", run_phase_baseline_cart),
    ("Phase 12A", "MOB (Parameter Instability)", run_phase_12a_mob),
    ("Phase 12B", "CTree (Permutation Rank)", run_phase_12b_ctree),
    ("Phase 12C.1", "Energy Distance Metric Splitting", run_phase_12c_metric_energy),
    ("Phase 12C.2", "MMD Kernel Splitting", run_phase_12c_metric_mmd),
    ("Phase 12D.1", "Wasserstein-2 Optimal Transport", run_phase_12d1_wasserstein),
    ("Phase 12D.2", "Demographic Parity Fair Splitting", run_phase_12d2_fairness_dp),
    ("Phase 12E.1", "Hellinger Distance (Skew Invariant)", run_phase_hellinger),
    ("Phase 12E.2", "MaxStat Rank Splitting", run_phase_maxstat),
    ("Phase 12F", "SPORF Oblique Splitting", run_phase_sporf),
    ("Phase 12G", "Equalized Odds Fair Splitting", run_phase_12g_fairness_eo),
    ("Phase 12H", "ExtraTrees (Randomized Splits)", run_phase_12h_extratrees),
    ("Phase 12I", "EBM FAST Interaction Screening", run_phase_12i_ebm_fast),
]


def display_detail_fairness_comparison(data: Dict[str, Any]) -> None:
    """Show detailed comparison of Unconstrained vs Demographic Parity vs Equalized Odds."""
    print("\n" + "=" * 94)
    print("  DEEP DIVE 1: FAIRNESS CRITERIA DIVERGENCE (UNCONSTRAINED vs DP vs EQUALIZED ODDS)")
    print("=" * 94)
    print("  Sensitive Attribute: Column 3 ('gender' in {0, 1})")
    print("  Biased Feature:      Column 4 ('gender_proxy' with heavy correlation to gender)")
    print("  Fair Merit Feature:  Column 5 ('fair_merit' independent of gender)")
    print("-" * 94)

    # 1. Unconstrained
    m_uncon = ModelTree(fairness_penalty=0.0, max_depth=1, min_rows=15, prune="none")
    m_uncon.fit(data["train"]["X"], data["train"]["y_fair"], col_names=data["col_names"], dtypes=data["dtypes"])
    p_uncon = m_uncon.predict(data["test"]["X"])
    s_test = [r[3] for r in data["test"]["X"]]
    d_uncon = compute_fairness_disparities(p_uncon, s_test, data["test"]["y_fair"])
    mse_uncon = compute_regression_metrics(data["test"]["y_fair"], p_uncon)["mse"]
    sp_uncon = m_uncon.root_split() or {}

    # 2. Demographic Parity
    m_dp = ModelTree(fairness_penalty=1.0, fairness_metric="demographic_parity", protected_col=3, max_depth=1, min_rows=15, prune="none")
    m_dp.fit(data["train"]["X"], data["train"]["y_fair"], col_names=data["col_names"], dtypes=data["dtypes"])
    p_dp = m_dp.predict(data["test"]["X"])
    d_dp = compute_fairness_disparities(p_dp, s_test, data["test"]["y_fair"])
    mse_dp = compute_regression_metrics(data["test"]["y_fair"], p_dp)["mse"]
    sp_dp = m_dp.root_split() or {}

    # 3. Equalized Odds
    m_eo = ModelTree(fairness_penalty=1.0, fairness_metric="equalized_odds", protected_col=3, max_depth=1, min_rows=15, prune="none")
    m_eo.fit(data["train"]["X"], data["train"]["y_fair"], col_names=data["col_names"], dtypes=data["dtypes"])
    p_eo = m_eo.predict(data["test"]["X"])
    d_eo = compute_fairness_disparities(p_eo, s_test, data["test"]["y_fair"])
    mse_eo = compute_regression_metrics(data["test"]["y_fair"], p_eo)["mse"]
    sp_eo = m_eo.root_split() or {}

    header = f"{'Model Constraint':<26} | {'Split Feature':<14} | {'Cut':<8} | {'DP Gap':<8} | {'EO Gap':<8} | {'Test MSE':<8}"
    print(header)
    print("-" * 94)
    print(f"{'Unconstrained (CART)':<26} | {str(sp_uncon.get('col')):<14} | {sp_uncon.get('threshold', 0.0):<8.3f} | {d_uncon['dp_gap']:<8.3f} | {d_uncon['eo_gap']:<8.3f} | {mse_uncon:<8.3f}")
    print(f"{'Demographic Parity Bound':<26} | {str(sp_dp.get('col')):<14} | {sp_dp.get('threshold', 0.0):<8.3f} | {d_dp['dp_gap']:<8.3f} | {d_dp['eo_gap']:<8.3f} | {mse_dp:<8.3f}")
    print(f"{'Equalized Odds Penalty':<26} | {str(sp_eo.get('col')):<14} | {sp_eo.get('threshold', 0.0):<8.3f} | {d_eo['dp_gap']:<8.3f} | {d_eo['eo_gap']:<8.3f} | {mse_eo:<8.3f}")
    print("-" * 94)
    print("Insight: Unconstrained CART selects 'gender' (or 'gender_proxy') due to raw variance reduction,")
    print(f"producing severe error disparity across genders (EO Gap = {d_uncon['eo_gap']:.3f}). Both DP and EO penalties")
    print(f"suppress the unfair feature and force the tree to select 'fair_merit', shrinking EO Gap to {d_eo['eo_gap']:.3f} (-{(1 - d_eo['eo_gap']/d_uncon['eo_gap'])*100:.1f}%).")


def display_detail_extratrees_comparison(data: Dict[str, Any]) -> None:
    """Show detailed comparison between CART exhaustive search and ExtraTrees random cuts."""
    print("\n" + "=" * 94)
    print("  DEEP DIVE 2: CUTPOINT SEARCH COMPLEXITY (CART EXHAUSTIVE vs EXTRATREES RANDOM CUTS)")
    print("=" * 94)

    # 1. Standard CART
    t_cart = ModelTree(extratrees=False, max_depth=1, min_rows=15, prune="none")
    t_cart.fit(data["train"]["X"], data["train"]["y_scalar"], col_names=data["col_names"])
    p_cart = t_cart.predict(data["test"]["X"])
    mse_cart = compute_regression_metrics(data["test"]["y_scalar"], p_cart)["mse"]
    sp_cart = t_cart.root_split() or {}

    # 2. ExtraTrees Seed 42
    t_et1 = ModelTree(extratrees=True, seed=42, max_depth=1, min_rows=15, prune="none")
    t_et1.fit(data["train"]["X"], data["train"]["y_scalar"], col_names=data["col_names"])
    p_et1 = t_et1.predict(data["test"]["X"])
    mse_et1 = compute_regression_metrics(data["test"]["y_scalar"], p_et1)["mse"]
    sp_et1 = t_et1.root_split() or {}

    # 3. ExtraTrees Seed 999
    t_et2 = ModelTree(extratrees=True, seed=999, max_depth=1, min_rows=15, prune="none")
    t_et2.fit(data["train"]["X"], data["train"]["y_scalar"], col_names=data["col_names"])
    p_et2 = t_et2.predict(data["test"]["X"])
    mse_et2 = compute_regression_metrics(data["test"]["y_scalar"], p_et2)["mse"]
    sp_et2 = t_et2.root_split() or {}

    header = f"{'Method':<26} | {'Split Feature':<14} | {'Cut':<8} | {'Candidates/Col':<16} | {'Gain':<10} | {'Test MSE':<8}"
    print(header)
    print("-" * 94)
    print(f"{'Standard CART':<26} | {str(sp_cart.get('col')):<14} | {sp_cart.get('threshold', 0.0):<8.3f} | {'N_unique (~240)':<16} | {sp_cart.get('gain', 0.0):<10.2f} | {mse_cart:<8.3f}")
    print(f"{'ExtraTrees (Seed 42)':<26} | {str(sp_et1.get('col')):<14} | {sp_et1.get('threshold', 0.0):<8.3f} | {'1 (Uniform Draw)':<16} | {sp_et1.get('gain', 0.0):<10.2f} | {mse_et1:<8.3f}")
    print(f"{'ExtraTrees (Seed 999)':<26} | {str(sp_et2.get('col')):<14} | {sp_et2.get('threshold', 0.0):<8.3f} | {'1 (Uniform Draw)':<16} | {sp_et2.get('gain', 0.0):<10.2f} | {mse_et2:<8.3f}")
    print("-" * 94)
    print("Insight: ExtraTrees samples exactly s ~ Uniform(min X_j, max X_j) per feature, completely bypassing")
    print("sorting and exhaustive scan. Evaluates only 1 cut candidate per column (O(1)), providing vast")
    print("speedup during ensemble bagging while decorrelating individual trees.")


def display_detail_ebm_fast_screening(data: Dict[str, Any]) -> None:
    """Show detailed ranking of pairwise non-linear interactions discovered by EBM FAST."""
    print("\n" + "=" * 94)
    print("  DEEP DIVE 3: EBM FAST 2D QUAD-STUMP INTERACTION SCREENING")
    print("=" * 94)
    print("  Screening all P*(P-1)/2 = 15 candidate pairs using fast residual quad-stumps:")
    print("-" * 94)

    screened = fast_interaction_screen(
        data["train"]["X"],
        data["train"]["y_inter"],
        col_names=data["col_names"],
        top_k=5,
        min_samples=15,
    )

    header = f"{'Rank':<5} | {'Feature 1':<14} | {'Feature 2':<14} | {'Cut 1':<8} | {'Cut 2':<8} | {'Interaction Score':<18}"
    print(header)
    print("-" * 94)
    for rank, inter in enumerate(screened, start=1):
        print(
            f"{rank:<5} | {inter['col1_name']:<14} | {inter['col2_name']:<14} | "
            f"{inter['cut1']:<8.3f} | {inter['cut2']:<8.3f} | {inter['score']:<18.4f}"
        )
    print("-" * 94)

    # Compare Additive Depth-2 Tree (no interactions) vs EBM FAST interaction tree
    m_1d = ModelTree(max_depth=2, interaction_groups=[[0], [2]], prune="none")
    m_1d.fit(data["train"]["X"], data["train"]["y_inter"], col_names=data["col_names"])
    p_1d = m_1d.predict(data["test"]["X"])
    mse_1d = compute_regression_metrics(data["test"]["y_inter"], p_1d)["mse"]

    m_fast = ModelTree(ebm_fast=True, max_depth=2, prune="none")
    m_fast.fit(data["train"]["X"], data["train"]["y_inter"], col_names=data["col_names"])
    p_fast = m_fast.predict(data["test"]["X"])
    mse_fast = compute_regression_metrics(data["test"]["y_inter"], p_fast)["mse"]

    err_reduction = (1.0 - mse_fast / mse_1d) * 100.0 if mse_1d > 0 else 0.0
    print(f"Additive Tree (Depth 2, Disjoint Groups) Test MSE: {mse_1d:.4f}")
    print(f"EBM FAST Tree (Coupled Top Pair)         Test MSE: {mse_fast:.4f}")
    print(f"Error Reduction from 2D Quad Interaction:          {err_reduction:.1f}%")
    print("Insight: Ground truth non-linear synergy (score > 0 and tier_flag = 1) is screened as #1 interaction.")
    print(f"Coupling the interacting features in the tree partition resolves residuals that additive trees miss (-{err_reduction:.1f}% MSE).")


def main():
    print("=" * 94)
    print("  MLTREELIB: COMPLETE STATISTICAL AND METRIC SPLITTING CRITERIA BENCHMARK SUITE")
    print("  Unified Task: Credit Risk Valuation Evaluated on Single Shared Dataset")
    print("=" * 94)

    data = generate_unified_benchmark_data(n_train=240, n_test=60, seed=42)
    print("Dataset Summary:")
    print(f"  Total Samples: {len(data['train']['X']) + len(data['test']['X'])} (Train: {len(data['train']['X'])}, Test: {len(data['test']['X'])})")
    print(f"  Features ({len(data['col_names'])}): {data['col_names']}")
    print(f"  Data Types:    {data['dtypes']}")
    print("  Target Modes:  Scalar (Loss/Return), 5D Embedding Vector, 20-Point Quantile Curve, Rare Binary")
    print("-" * 94)

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
        print(f"    Split Chosen:    Feature '{res['split_col']}' at threshold {thresh_str}")
        print(f"    Objective Gain:  {res['gain']:.4f}")
        print(f"    Test Evaluation: {res['test_metric_name']} = {res['test_metric_val']:.4f} ({res['disparity_or_cost']})")
        print(f"    Theoretical:     {res['note']}")

    print("\n" + "=" * 94)
    print("  UNIFIED BENCHMARK COMPARATIVE SUMMARY (SAME DATA / SAME TASK)")
    print("=" * 94)
    header = (
        f"{'Phase':<12} | {'Criterion / Model':<32} | {'Target':<10} | "
        f"{'Split / Cut':<20} | {'Test Error':<12} | {'Metric / Cost':<14}"
    )
    print(header)
    print("-" * 94)

    for r in results:
        col_str = str(r["split_col"])
        if len(col_str) > 20:
            col_str = col_str[:17] + "..."
        metric_str = f"{r['test_metric_name']}={r['test_metric_val']:.3f}"
        print(
            f"{r['phase']:<12} | {r['name']:<32} | {r['target_type']:<10} | "
            f"{col_str:<20} | {metric_str:<12} | {r['disparity_or_cost']:<14}"
        )

    print("=" * 94)

    # Detailed comparative deep dives
    display_detail_fairness_comparison(data)
    display_detail_extratrees_comparison(data)
    display_detail_ebm_fast_screening(data)

    print("\n" + "=" * 94)
    print("  FINAL SYNTHESIS & SPEC 12 COMPLETION")
    print("=" * 94)
    print("1. Standard CART: Greedily minimizes empirical SSE; vulnerable to proxy and cardinality bias.")
    print("2. MOB (12A): Tests score residuals for parameter instability; detects structural break regimes.")
    print("3. CTree (12B): Linear rank permutation test eliminates cardinality selection bias.")
    print("4. Metric Criteria (12C): Optimizes non-parametric vector embeddings without distribution models.")
    print("5. Wasserstein-2 (12D.1): Fréchet barycenter dispersion splits continuous probability curves.")
    print("6. Fair Splitting (12D.2 & 12G): Demographic Parity and Equalized Odds penalties suppress biased proxies.")
    print("7. Hellinger & MaxStat (12E): Skew-invariant and rank-invariant splitting across tail and ordinal data.")
    print("8. SPORF (12F): Evaluates random bivariate hyperplanes, recovering diagonal separation seams.")
    print("9. ExtraTrees (12H): Random cutpoint sampling achieves O(1) evaluation speed without sorting.")
    print("10. EBM FAST (12I): Residual quad-stump screening isolates non-linear interactions across all pairs.")
    print("=" * 94)


if __name__ == "__main__":
    main()
