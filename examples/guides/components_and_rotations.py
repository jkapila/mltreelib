"""Component Analysis, R-Aligned Factor Rotations, and Matrix Factorization Showcase.

Demonstrates:
1. PCA with R-aligned rotations (Varimax, Quartimax, Equamax, Parsimax, Biquartimax, and Promax).
2. Truncated SVD for uncentered rank reduction.
3. Whitening (ZCA vs PCA).
4. FastICA independent source separation.
5. Non-negative Matrix Factorization (MU, Projected ALS, and Hierarchical ALS with L1/L2).
6. ModelReport diagnostics and reconstruction errors.
"""

from __future__ import annotations

import math
from mltreelib.components import ComponentModel


def generate_synthetic_factor_data(n_samples: int = 200) -> list[list[float]]:
    """Generate 6-dimensional data driven by 2 distinct latent factors with noise."""
    data: list[list[float]] = []
    for i in range(n_samples):
        # Latent factor 1 (affects cols 0, 1, 2)
        f1 = math.sin(i * 0.1) * 3.0
        # Latent factor 2 (affects cols 3, 4, 5)
        f2 = ((i % 20) / 10.0 - 1.0) * 4.0

        noise = (i % 7) * 0.05
        row = [
            f1 + noise,
            1.5 * f1 - 0.5 * noise,
            -0.8 * f1 + noise,
            f2 + 0.5 * noise,
            2.0 * f2 - noise,
            -1.2 * f2 + noise,
        ]
        data.append(row)
    return data


def run_showcase() -> None:
    data = generate_synthetic_factor_data(200)
    print("=" * 78)
    print("Component Analysis & Factor Rotations Showcase")
    print("=" * 78)

    # 1. Classical PCA (Unrotated)
    print("\n[1] Classical PCA (Unrotated)")
    pca_raw = ComponentModel(kind="pca", n_components=2, standardize=True, rotation="none")
    pca_raw.fit(data)
    evr = pca_raw.explained_variance_ratio_ or []
    print(f"Explained Variance Ratio: {[f'{v:.4f}' for v in evr]}")
    print(f"Cumulative Variance: {sum(evr):.4f}")

    # 2. R-Aligned Orthomax Rotations
    print("\n[2] R-Aligned Factor Rotations on PCA Loadings")
    rotations = ["varimax", "quartimax", "equamax", "parsimax", "biquartimax", "promax"]
    for rot in rotations:
        m = ComponentModel(kind="pca", n_components=2, standardize=True, rotation=rot)
        m.fit(data)
        rot_mat = m.rotation_matrix_
        t_shape = f"{len(rot_mat)}x{len(rot_mat[0])}" if rot_mat else "None"
        print(f"  Rotation: {rot.upper():<12} | Rotation Matrix T: {t_shape}")

    # 3. Truncated SVD
    print("\n[3] Truncated SVD (Uncentered)")
    svd = ComponentModel(kind="svd", n_components=2)
    svd_scores = svd.fit_transform(data)
    print(f"SVD Transformed Shape: {len(svd_scores)}x{len(svd_scores[0])}")

    # 4. Whitening (ZCA vs PCA)
    print("\n[4] Covariance Whitening")
    zca = ComponentModel(kind="whitening", whitening_kind="zca")
    whitened = zca.fit_transform(data)
    print(f"ZCA Whitened Matrix Shape: {len(whitened)}x{len(whitened[0])}")

    # 5. FastICA Source Separation
    print("\n[5] FastICA Source Separation")
    ica = ComponentModel(kind="ica", n_components=2, seed=42)
    ica_sources = ica.fit_transform(data)
    print(f"FastICA Recovered Sources: {len(ica_sources)}x{len(ica_sources[0])}")

    # 6. Non-negative Matrix Factorization (NMF) Solvers
    print("\n[6] NMF Solvers: MU vs ALS vs HALS")
    pos_data = [[abs(val) + 0.1 for val in row] for row in data]
    for solver in ["mu", "als", "hals"]:
        nmf = ComponentModel(
            kind="nmf",
            n_components=2,
            solver=solver,
            l1_reg=0.01 if solver == "hals" else 0.0,
            l2_reg=0.01,
            seed=42,
        )
        nmf.fit(pos_data)
        err = nmf.reconstruction_error_ or 0.0
        print(f"  NMF Solver: {solver.upper():<6} | Frobenius Error: {err:.4f}")

    # 7. ModelReport Diagnostics
    print("\n[7] ModelReport Diagnostics (PCA with Varimax)")
    varimax_pca = ComponentModel(kind="pca", n_components=2, rotation="varimax")
    varimax_pca.fit(data)
    print(varimax_pca.report.to_text())


if __name__ == "__main__":
    run_showcase()
