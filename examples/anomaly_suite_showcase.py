"""Comprehensive Showcase of mltreelib Base Anomaly Trees on Real Benchmark Data.

Demonstrates all base anomaly tree paradigms on real data (credit-g):
1. ExtendedIsolationTree (EIF): Oblique hyperplane cuts + native categorical projection.
2. DensityEstimationTree (DET): Integrated Squared Error density and log-density anomaly scoring.
3. HalfSpaceTree (HST): Streaming mass scoring and online point insertion.
4. RandomCutTree (RCT): Collusive Displacement (CoDisp) for masked outliers.
5. ActiveAnomalyTree (AAD): Dynamic human feedback loop updating node weights.
6. ContextAnomalyTree (CAT): Sequential categorical transitions and surprise scoring in bits.
"""

from pathlib import Path
import pyarrow.parquet as pq

from mltreelib import (
    ActiveAnomalyTree,
    ContextAnomalyTree,
    DensityEstimationTree,
    ExtendedIsolationTree,
    HalfSpaceTree,
    RandomCutTree,
)


def main():
    path = Path("datasets/credit-g.parquet")
    if not path.exists():
        print(f"Dataset not found at {path}")
        return

    # Use PyArrow ParquetFile with context manager as requested
    with pq.ParquetFile(path) as pf:
        table = pf.read()
        df = table.to_pandas()

    num_cols = [
        "duration_months",
        "credit_amount",
        "installment_rate_percent",
        "age_years",
        "existing_credits_count",
    ]
    cat_cols = [
        "checking_status",
        "credit_history",
        "credit_purpose",
        "savings_status",
        "housing",
    ]

    all_cols = num_cols + cat_cols
    dtypes = ["float"] * len(num_cols) + ["cat"] * len(cat_cols)

    # Convert dataframe to rows with proper types
    data_mixed = df[all_cols].values.tolist()
    data_num = df[num_cols].values.tolist()

    print("=" * 80)
    print("mltreelib Base Anomaly Tree Suite Showcase (credit-g dataset)")
    print(f"Loaded {len(df)} records across {len(all_cols)} attributes ({len(num_cols)} numeric, {len(cat_cols)} categorical).")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Extended Isolation Tree (EIF) with Categoricals
    # ---------------------------------------------------------
    print("\n[1] Extended Isolation Tree (EIF) - Oblique Cuts with Native Categoricals")
    eif = ExtendedIsolationTree(max_depth=10, extension_level=2, seed=42)
    eif.fit(data_mixed, col_names=all_cols, dtypes=dtypes)

    scores_eif = eif.anomaly_score(data_mixed)
    path_lens = eif.path_length(data_mixed)

    top_eif_idx = sorted(range(len(scores_eif)), key=lambda i: scores_eif[i], reverse=True)[:3]
    print("  Top 3 anomalous records via EIF:")
    for rank, idx in enumerate(top_eif_idx, 1):
        row = df.iloc[idx]
        print(f"   {rank}. Row {idx:3d} | score: {scores_eif[idx]:.4f} | path: {path_lens[idx]:.2f}")
        print(f"      Credit: ${row['credit_amount']:.0f}, Duration: {row['duration_months']} mo, Age: {row['age_years']}")
        print(f"      History: {row['credit_history']}, Purpose: {row['credit_purpose']}, Checking: {row['checking_status']}")

    # ---------------------------------------------------------
    # 2. Density Estimation Tree (DET)
    # ---------------------------------------------------------
    print("\n[2] Density Estimation Tree (DET) - Min ISE Density Partitioning")
    det = DensityEstimationTree(max_depth=6, min_samples_split=10, min_leaf_samples=5)
    det.fit(data_num, col_names=num_cols, dtypes=["float"] * len(num_cols))

    densities = det.predict_density(data_num)
    det_scores = det.predict_anomaly_score(data_num)

    top_det_idx = sorted(range(len(det_scores)), key=lambda i: det_scores[i], reverse=True)[:3]
    print("  Top 3 lowest-density records via DET:")
    for rank, idx in enumerate(top_det_idx, 1):
        row = df.iloc[idx]
        print(f"   {rank}. Row {idx:3d} | -ln(density): {det_scores[idx]:.2f} | density: {densities[idx]:.2e}")
        print(f"      Credit: ${row['credit_amount']:.0f}, Duration: {row['duration_months']} mo, Age: {row['age_years']}")

    # ---------------------------------------------------------
    # 3. Half-Space Tree (HST) - Streaming Mass Scoring
    # ---------------------------------------------------------
    print("\n[3] Half-Space Tree (HST) - Streaming Mass Anomaly Detection")
    hst = HalfSpaceTree(max_depth=8, seed=42)
    hst.fit(data_num)

    hst_scores = hst.anomaly_score(data_num)
    top_hst_idx = sorted(range(len(hst_scores)), key=lambda i: hst_scores[i], reverse=True)[:3]
    print("  Top 3 HST anomalies (sparse mass subspaces):")
    for rank, idx in enumerate(top_hst_idx, 1):
        row = df.iloc[idx]
        print(f"   {rank}. Row {idx:3d} | anomaly score: {hst_scores[idx]:.4f} | mass score: {hst.mass_score([data_num[idx]])[0]:.1f}")
        print(f"      Credit: ${row['credit_amount']:.0f}, Duration: {row['duration_months']} mo, Age: {row['age_years']}")

    # Online streaming demonstration
    synth_outlier = [120.0, 50000.0, 10.0, 95.0, 8.0]
    score_before = hst.anomaly_score([synth_outlier])[0]
    hst.insert(synth_outlier)
    score_after = hst.anomaly_score([synth_outlier])[0]
    print(f"  Streaming insertion test: synthetic outlier score {score_before:.4f} -> {score_after:.4f} (mass updated online)")

    # ---------------------------------------------------------
    # 4. Random Cut Tree (RCT) - Collusive Displacement (CoDisp)
    # ---------------------------------------------------------
    print("\n[4] Random Cut Tree (RCT) - Collusive Displacement (CoDisp)")
    rct = RandomCutTree(seed=42)
    rct.fit(data_num)

    codisp_scores = rct.codisp(data_num)
    top_rct_idx = sorted(range(len(codisp_scores)), key=lambda i: codisp_scores[i], reverse=True)[:3]
    print("  Top 3 RCT anomalies via CoDisp:")
    for rank, idx in enumerate(top_rct_idx, 1):
        row = df.iloc[idx]
        print(f"   {rank}. Row {idx:3d} | CoDisp: {codisp_scores[idx]:.2f}")
        print(f"      Credit: ${row['credit_amount']:.0f}, Duration: {row['duration_months']} mo, Age: {row['age_years']}")

    # ---------------------------------------------------------
    # 5. Active Anomaly Tree (AAD) - Human-in-the-Loop Feedback
    # ---------------------------------------------------------
    print("\n[5] Active Anomaly Tree (AAD) - Active Human-in-the-Loop Feedback")
    aad = ActiveAnomalyTree(max_depth=8, seed=42)
    aad.fit(data_num)

    initial_scores = aad.anomaly_score(data_num)
    initial_top = sorted(range(len(initial_scores)), key=lambda i: initial_scores[i], reverse=True)[:5]
    print(f"  Initial top candidate presented to analyst: Row {initial_top[0]} (score: {initial_scores[initial_top[0]]:.4f})")

    # Analyst confirms candidate is a true positive anomaly
    reviewed_row = data_num[initial_top[0]]
    aad.update_feedback(reviewed_row, is_anomaly=True, eta=0.8)

    updated_scores = aad.anomaly_score(data_num)
    print(f"  Feedback incorporated (+0.8): Row {initial_top[0]} score increased to {updated_scores[initial_top[0]]:.4f}")
    new_top = sorted(range(len(updated_scores)), key=lambda i: updated_scores[i], reverse=True)[:3]
    print(f"  Re-ranked top candidates after analyst label: {new_top}")

    # ---------------------------------------------------------
    # 6. Context Anomaly Tree (CAT) - Sequence / Discrete Surprise
    # ---------------------------------------------------------
    print("\n[6] Context Anomaly Tree (CAT) - Sequence Transition Surprise (in bits)")
    cat = ContextAnomalyTree(max_depth=4, alphabet_size=20)

    # Encode purpose category to integer IDs and build transaction sequences
    unique_purposes = {p: i for i, p in enumerate(df["credit_purpose"].unique())}
    sequences = []
    # Build synthetic sequential credit history sequences across client cohorts
    for grp_idx in range(0, len(df) - 4, 4):
        seq = [unique_purposes[df["credit_purpose"].iloc[grp_idx + k]] for k in range(4)]
        sequences.append(seq)

    cat.fit(sequences)

    # Evaluate common transition vs rare anomalous transition
    test_context = sequences[0][:2]
    expected_sym = sequences[0][2]
    rare_sym = 19  # rare / unobserved symbol

    prob_exp = cat.conditional_prob(test_context, expected_sym)
    prob_rare = cat.conditional_prob(test_context, rare_sym)
    surprise_exp = cat.surprise_score(test_context, expected_sym)
    surprise_rare = cat.surprise_score(test_context, rare_sym)

    print(f"  Context sequence: {test_context}")
    print(f"  Expected transition {expected_sym}: P = {prob_exp:.4f} | Surprise = {surprise_exp:.2f} bits")
    print(f"  Rare transition     {rare_sym}: P = {prob_rare:.4f} | Surprise = {surprise_rare:.2f} bits")
    print(f"  Surprise elevation: {surprise_rare / max(surprise_exp, 1e-6):.2f}x higher")

    print("\n" + "=" * 80)
    print("All base anomaly tree paradigms demonstrated successfully on live data.")
    print("=" * 80)


if __name__ == "__main__":
    main()
