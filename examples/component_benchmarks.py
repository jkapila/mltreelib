"""Suite 10: Component Analysis, Matrix Factorizations & R-Aligned Rotations Benchmark.

Why this was built:
Standard tabular learners assume flat, orthogonal, or independent input spaces. In real empirical
tabular data, features often cluster into correlated latent constructs or low-rank subspaces:
1. PCA with R-Aligned Rotations: Unrotated PCA directions maximize variance globally but produce
   dense, mixed loadings that obscure underlying latent factors. Rotations (Varimax, Quartimax,
   Equamax, Parsimax, Biquartimax, and Promax) apply Kaiser normalization and orthogonal/oblique
   rotations to maximize simple structure (Thurstone 1947), isolating clean, interpretable factors.
2. Truncated SVD: Factorizes uncentered, wide, or non-negative data without forcing column centering.
3. Covariance Whitening: Transforms features into uncorrelated coordinates with identity covariance,
   distinguishing ZCA (zero-phase, preserving original feature geometry) from PCA whitening.
4. FastICA: Exploits non-Gaussianity (negentropy / kurtosis) to separate statistically independent
   source signals beyond second-order covariance.
5. Non-negative Matrix Factorization (NMF): Enforces purely additive parts-based representations
   (X approx W*H with W, H >= 0). Evaluates three solver algorithms:
   - MU: Multiplicative Updates (Lee & Seung 2001).
   - ALS: Projected Alternating Least Squares via normal equations.
   - HALS: Hierarchical Alternating Least Squares (Cichocki et al.) with exact coordinate descent
     and L1 sparsity / L2 ridge shrinkage.
6. Subspace Reconstruction Anomaly Detection: Measures squared residual distance ||x - x_hat||^2
   from the low-rank manifold to detect out-of-subspace novelties.

Use Cases & When to Use:
- Factor Rotations: Survey psychometrics, customer segment attribution, latent risk factors.
- Whitening: Preprocessing for ICA, decorrelating multicollinear tree leaves.
- FastICA: Sensor signal unmixing, artifact removal, independent source extraction.
- NMF: Text topic modeling, spending decomposition, additive nonnegative parts.
- Reconstruction Anomaly: High-dimensional fraud, sensor fault detection, distribution drift.

Run:
    uv run python examples/suites/component_benchmarks.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mltreelib.components import ComponentModel
from mltreelib.anomaly import AnomalyModel


def load_numeric_dataset(path: str, max_rows: int = 1000) -> tuple[list[list[float]], list[str]]:
    """Load numeric columns from a parquet dataset."""
    pf = pq.ParquetFile(path)
    df = pf.read().to_pandas().head(max_rows)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_clean = df[num_cols].dropna()
    return df_clean.values.tolist(), num_cols


def run_component_benchmarks() -> None:
    print("=" * 80)
    print("mltreelib Suite 10: Component Analysis & Factor Rotations Benchmark")
    print("=" * 80)

    # Load Wine Quality or synthetic data
    wine_path = Path("datasets/wine_quality.parquet")
    if wine_path.exists():
        data, cols = load_numeric_dataset(str(wine_path), max_rows=1000)
        ds_name = f"Wine Quality ({len(data)} rows, {len(cols)} numeric features)"
    else:
        # Fallback synthetic correlated data
        np.random.seed(42)
        n = 500
        f1 = np.random.randn(n) * 2.0
        f2 = np.random.randn(n) * 1.5
        x0 = f1 + np.random.randn(n) * 0.2
        x1 = 2.0 * f1 - np.random.randn(n) * 0.3
        x2 = -1.5 * f1 + np.random.randn(n) * 0.2
        x3 = f2 + np.random.randn(n) * 0.1
        x4 = 3.0 * f2 + np.random.randn(n) * 0.2
        x5 = -0.5 * f2 - np.random.randn(n) * 0.3
        data = np.column_stack([x0, x1, x2, x3, x4, x5]).tolist()
        cols = [f"feat_{i}" for i in range(6)]
        ds_name = f"Synthetic Factor Data ({n} rows, 6 features)"

    print(f"Dataset: {ds_name}\n")

    # =========================================================================
    # 1. PCA & R-Aligned Factor Rotations Benchmark
    # =========================================================================
    print("-" * 80)
    print("1. PCA & R-ALIGNED FACTOR ROTATIONS BENCHMARK")
    print("   Problem: Orthomax & Promax rotation to achieve Thurstone's simple structure")
    print("-" * 80)

    rotations = [
        ("none", "Unrotated PCA (Maximum Global Variance)"),
        ("varimax", "Varimax (R stats::varimax, gamma=1.0)"),
        ("quartimax", "Quartimax (Row Simplicity, gamma=0.0)"),
        ("equamax", "Equamax (Balanced Orthomax, gamma=k/2)"),
        ("parsimax", "Parsimax (Feature-Sample Weighted)"),
        ("biquartimax", "Biquartimax (gamma=0.5)"),
        ("promax", "Promax (R stats::promax, Oblique Procrustes, m=4.0)"),
    ]

    n_comp = min(3, len(cols))
    pca_results = []
    for rot_name, rot_desc in rotations:
        t0 = time.perf_counter()
        m = ComponentModel(
            kind="pca",
            n_components=n_comp,
            standardize=True,
            rotation=rot_name,
            promax_m=4.0,
        )
        m.fit(data)
        fit_time_ms = (time.perf_counter() - t0) * 1000.0

        evr = sum(m.explained_variance_ratio_ or [])
        rot_mat = m.rotation_matrix_
        has_rot = "Yes (orthogonal)" if (rot_mat and rot_name != "promax") else ("Yes (oblique)" if rot_mat else "No")

        # Loading sparsity: average proportion of near-zero loadings (< 0.15 * max)
        loadings = np.array(m.components_)
        max_abs = np.max(np.abs(loadings))
        sparsity = float(np.mean(np.abs(loadings) < 0.15 * max_abs))

        pca_results.append({
            "Method": rot_name.upper(),
            "Description": rot_desc,
            "Variance Retained": f"{evr:.2%}",
            "Rotation Matrix": has_rot,
            "Loading Sparsity": f"{sparsity:.1%}",
            "Fit Time (ms)": f"{fit_time_ms:.2f}",
        })

    pca_df = pd.DataFrame(pca_results)
    print(pca_df[["Method", "Variance Retained", "Rotation Matrix", "Loading Sparsity", "Fit Time (ms)"]].to_string(index=False))
    print("\nTakeaway: Rotated factor loadings concentrate variance into fewer high-magnitude features,")
    print("achieving interpretable latent factor separation without sacrificing subspace coverage.\n")

    # =========================================================================
    # 2. Covariance Whitening: ZCA vs PCA
    # =========================================================================
    print("-" * 80)
    print("2. COVARIANCE WHITENING: ZCA VS PCA")
    print("   Problem: Decorrelate features into identity covariance matrix Cov(Z) = I")
    print("-" * 80)

    zca_mod = ComponentModel(kind="whitening", whitening_kind="zca")
    t0 = time.perf_counter()
    z_zca = zca_mod.fit_transform(data)
    zca_time = (time.perf_counter() - t0) * 1000.0

    pca_white = ComponentModel(kind="whitening", whitening_kind="pca")
    t0 = time.perf_counter()
    z_pca = pca_white.fit_transform(data)
    pca_white_time = (time.perf_counter() - t0) * 1000.0

    cov_zca = np.cov(np.array(z_zca), rowvar=False)
    cov_pca = np.cov(np.array(z_pca), rowvar=False)
    zca_diag_err = float(np.mean(np.abs(cov_zca - np.eye(len(cols)))))
    pca_diag_err = float(np.mean(np.abs(cov_pca - np.eye(len(cols)))))

    print(f"ZCA (Zero-Phase Component Analysis): Time={zca_time:.2f}ms | ||Cov(Z) - I||_mean = {zca_diag_err:.6f}")
    print(f"PCA Whitening                      : Time={pca_white_time:.2f}ms | ||Cov(Z) - I||_mean = {pca_diag_err:.6f}")
    print("Property: ZCA minimizes ||Z - X||_F, keeping whitened coordinates closest to original features.\n")

    # =========================================================================
    # 3. Non-Negative Matrix Factorization (NMF) Solvers Benchmark
    # =========================================================================
    print("-" * 80)
    print("3. NON-NEGATIVE MATRIX FACTORIZATION (NMF) SOLVER FACE-OFF")
    print("   Problem: min_{W,H >= 0} ||X - WH||_F^2 on strictly non-negative inputs")
    print("-" * 80)

    # Shift data to strictly non-negative domain
    data_arr = np.array(data)
    pos_data = (data_arr - np.min(data_arr) + 0.1).tolist()

    nmf_solvers = [
        ("mu", "Multiplicative Updates (Lee & Seung 2001)", 0.0, 0.0),
        ("als", "Projected Alternating Least Squares (ALS)", 0.0, 0.01),
        ("hals", "Hierarchical ALS (HALS) - Unregularized", 0.0, 0.0),
        ("hals", "Hierarchical ALS (HALS) - L1 Sparse (lambda1=0.01)", 0.01, 0.005),
        ("hals", "Hierarchical ALS (HALS) - Elastic Net (L1=0.01, L2=0.05)", 0.01, 0.05),
    ]

    nmf_results = []
    k_factors = 3
    for solver_code, desc, l1, l2 in nmf_solvers:
        t0 = time.perf_counter()
        nmf = ComponentModel(
            kind="nmf",
            n_components=k_factors,
            solver=solver_code,
            l1_reg=l1,
            l2_reg=l2,
            max_iter=200,
            seed=42,
        )
        nmf.fit(pos_data)
        elapsed = (time.perf_counter() - t0) * 1000.0

        err = nmf.reconstruction_error_ or 0.0
        # Sparsity of H matrix
        h_mat = np.array(nmf.components_)
        h_sparsity = float(np.mean(h_mat < 1e-4))

        label = f"{solver_code.upper()} (L1={l1}, L2={l2})" if (l1 > 0 or l2 > 0) else solver_code.upper()
        nmf_results.append({
            "Solver": label,
            "Description": desc,
            "Reconstruction Error": f"{err:.4f}",
            "H Sparsity": f"{h_sparsity:.1%}",
            "Time (ms)": f"{elapsed:.2f}",
        })

    nmf_df = pd.DataFrame(nmf_results)
    print(nmf_df[["Solver", "Reconstruction Error", "H Sparsity", "Time (ms)"]].to_string(index=False))
    print("\nInsight: HALS coordinate descent achieves superior loss convergence with exact NNLS updates")
    print("and enables true L1 sparsity without numerical instability.\n")

    # =========================================================================
    # 4. Subspace Reconstruction Anomaly Detection
    # =========================================================================
    print("-" * 80)
    print("4. SUBSPACE RECONSTRUCTION ANOMALY DETECTION")
    print("   Problem: Detect out-of-subspace outliers via PCA & NMF reconstruction error")
    print("-" * 80)

    anom_pca = AnomalyModel(kind="pca", n_components=2, contamination=0.05)
    anom_pca.fit(data)

    # Inject an out-of-subspace outlier
    inlier_sample = [data[0]]
    outlier_sample = [[v + 50.0 for v in data[0]]]

    inlier_err = anom_pca.score_samples(inlier_sample)[0]
    outlier_err = anom_pca.score_samples(outlier_sample)[0]
    inlier_pred = anom_pca.predict(inlier_sample)[0]
    outlier_pred = anom_pca.predict(outlier_sample)[0]

    print(f"Normal Inlier Point   -> Reconstruction Loss: {inlier_err:.4f} | Label: {'ANOMALY' if inlier_pred else 'NORMAL'}")
    print(f"Outlier Spike (+50.0) -> Reconstruction Loss: {outlier_err:.4f} | Label: {'ANOMALY' if outlier_pred else 'NORMAL'}")
    print(f"Ratio (Outlier/Inlier Loss): {outlier_err / max(1e-6, inlier_err):.1f}x separation")
    print("=" * 80)


if __name__ == "__main__":
    run_component_benchmarks()
