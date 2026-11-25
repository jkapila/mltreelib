"""Educational Deep-Dive: Factor Rotations in Principal Component Analysis.

Why this was built:
Principal Component Analysis (PCA) is commonly used for dimensionality reduction, but raw PCA
directions are notoriously difficult to interpret as domain concepts. Raw PCA chooses axes that
maximize global variance in decreasing order. As a result:
1. First Principal Component (PC1) almost always acts as a "size" or "general magnitude" factor,
   with dense, positive loadings on nearly all variables.
2. Subsequent components act as bipolar contrast vectors (+ versus -) that mix unrelated features.

Factor Rotation solves the Rotational Indeterminacy problem:
Since any orthogonal matrix T satisfies T^T T = I, the reconstructed subspace is mathematically identical:
    X_hat = Z V^T = (Z T) (V T)^T = Z* (V*)^T
Rotating the loadings matrix L = V * diag(s) via T does NOT lose any explained variance. Instead, it
rotates the axes to satisfy Thurstone's (1947) classic criteria for "Simple Structure":
- Each factor should have high loadings on a few features and near-zero loadings on the rest.
- Each feature should be explained primarily by one factor.

The Orthomax & Promax Family:
- Varimax (R stats::varimax, gamma=1.0): Maximizes the variance of squared loadings across variables
  within each factor. This produces clean, distinct factor definitions.
- Quartimax (gamma=0.0): Maximizes the variance of squared loadings across factors within each variable.
- Equamax (gamma=k/2): A balanced compromise between variable simplicity and factor simplicity.
- Parsimax (gamma = p*(k-1)/(p+k-2)): Weighting based on feature dimension and factor count.
- Biquartimax (gamma=0.5): Intermediate between Quartimax and Varimax.
- Promax (R stats::promax, oblique): Relaxes the orthogonality constraint to reflect real-world
  phenomena where latent factors naturally correlate (e.g., math and verbal ability). Computes
  Varimax, raises loadings to power m (default 4.0), and solves the Procrustes regression.

Run:
    uv run python examples/guides/pca_rotations_educational.py
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from mltreelib.components import ComponentModel


def generate_educational_aptitude_data(n: int = 500, seed: int = 42) -> tuple[list[list[float]], list[str]]:
    """Generate 8-dimensional cognitive aptitude test scores driven by two latent factors:

    Factor 1 (Quantitative): Math, Physics, Logic, Coding
    Factor 2 (Verbal): Reading, Essay, Vocabulary, Grammar
    """
    rng = np.random.default_rng(seed)

    # Underlying latent abilities (with natural 0.35 correlation between them)
    cov = [[1.0, 0.35], [0.35, 1.0]]
    factors = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    quant = factors[:, 0]
    verbal = factors[:, 1]

    # Quantitative battery
    math_score = 500 + 100 * (0.85 * quant + rng.normal(0, 0.3, n))
    physics_score = 500 + 100 * (0.80 * quant + rng.normal(0, 0.35, n))
    logic_score = 500 + 100 * (0.75 * quant + rng.normal(0, 0.35, n))
    coding_score = 500 + 100 * (0.70 * quant + rng.normal(0, 0.4, n))

    # Verbal battery
    reading_comp = 500 + 100 * (0.85 * verbal + rng.normal(0, 0.3, n))
    essay_score = 500 + 100 * (0.80 * verbal + rng.normal(0, 0.35, n))
    vocab_score = 500 + 100 * (0.75 * verbal + rng.normal(0, 0.35, n))
    grammar_score = 500 + 100 * (0.70 * verbal + rng.normal(0, 0.4, n))

    feature_names = [
        "Math",
        "Physics",
        "Logic",
        "Coding",
        "Reading",
        "Essay",
        "Vocabulary",
        "Grammar",
    ]
    matrix = np.column_stack([
        math_score,
        physics_score,
        logic_score,
        coding_score,
        reading_comp,
        essay_score,
        vocab_score,
        grammar_score,
    ])
    return matrix.tolist(), feature_names


def format_loadings_table(components: list[list[float]], feature_names: list[str], factor_names: list[str]) -> str:
    """Format factor loadings as an aligned ASCII matrix."""
    lines = []
    header = f"{'Feature':<16} | " + " | ".join(f"{name:>12}" for name in factor_names)
    lines.append(header)
    lines.append("-" * len(header))

    p = len(feature_names)
    k = len(components)
    for j in range(p):
        row_str = f"{feature_names[j]:<16} | "
        vals = []
        for i in range(k):
            val = components[i][j]
            # Emphasize dominant loadings with an asterisk
            tag = "*" if abs(val) >= 0.35 else " "
            vals.append(f"{val:>10.4f} {tag}")
        row_str += " | ".join(vals)
        lines.append(row_str)
    return "\n".join(lines)


def run_educational_demonstration() -> None:
    print("=" * 80)
    print("EDUCATIONAL GUIDE: FACTOR ROTATIONS IN PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 80)
    print("Domain: Cognitive Aptitude Battery (8 tests spanning Quantitative & Verbal domains)")
    print("Sample Size: 500 applicants | Number of Factors to Extract: 2\n")

    data, features = generate_educational_aptitude_data(n=500, seed=42)

    # -------------------------------------------------------------------------
    # 1. Unrotated PCA
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("1. UNROTATED PCA (Standard SVD Eigenvector Projections)")
    print("-" * 80)
    pca_raw = ComponentModel(kind="pca", n_components=2, standardize=True, rotation="none")
    pca_raw.fit(data)

    ev1, ev2 = pca_raw.explained_variance_ratio_[:2]
    print(f"Variance Explained: PC1 = {ev1:.1%}, PC2 = {ev2:.1%} (Total = {ev1 + ev2:.1%})\n")
    print(format_loadings_table(pca_raw.components_, features, ["PC1 (Global)", "PC2 (Bipolar)"]))
    print("\nObservation on Unrotated PCA:")
    print("- PC1 has uniformly positive, heavy loadings across ALL 8 tests (general intelligence 'g').")
    print("- PC2 is a bipolar contrast (+ Quant vs - Verbal), but both domains are entangled.")
    print("- A practitioner cannot name either component as a distinct, standalone operational trait.\n")

    # -------------------------------------------------------------------------
    # 2. Varimax Orthogonal Rotation (stats::varimax)
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("2. VARIMAX ROTATION (Kaiser-Normalized Orthogonal Simple Structure, gamma=1.0)")
    print("-" * 80)
    pca_varimax = ComponentModel(kind="pca", n_components=2, standardize=True, rotation="varimax")
    pca_varimax.fit(data)

    print(f"Variance Explained: Retains exact same {sum(pca_varimax.explained_variance_ratio_):.1%} total subspace variance!\n")
    print(format_loadings_table(pca_varimax.components_, features, ["Factor 1 (Verbal)", "Factor 2 (Quant)"]))

    rot_m = pca_varimax.rotation_matrix_
    print(f"\nOrthogonal Rotation Matrix T (2x2):")
    print(f"  [[{rot_m[0][0]:+.4f}, {rot_m[0][1]:+.4f}],")
    print(f"   [{rot_m[1][0]:+.4f}, {rot_m[1][1]:+.4f}]]")
    print(f"Orthogonality Check: T^T T approx I (dot product = {rot_m[0][0]*rot_m[0][1] + rot_m[1][0]*rot_m[1][1]:.2e})")

    print("\nObservation on Varimax:")
    print("- Thurstone's Simple Structure is achieved cleanly!")
    print("- Factor 1 isolates Verbal tests (Reading, Essay, Vocab, Grammar: loadings > 0.40).")
    print("- Factor 2 isolates Quantitative tests (Math, Physics, Logic, Coding: loadings > 0.40).")
    print("- Cross-domain loadings drop close to zero. The factors now possess undeniable business meaning.\n")

    # -------------------------------------------------------------------------
    # 3. Promax Oblique Rotation (stats::promax)
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("3. PROMAX ROTATION (Oblique Procrustes Solution, Target Power m=4.0)")
    print("-" * 80)
    pca_promax = ComponentModel(kind="pca", n_components=2, standardize=True, rotation="promax", promax_m=4.0)
    pca_promax.fit(data)

    print(format_loadings_table(pca_promax.components_, features, ["Factor 1 (Verbal)", "Factor 2 (Quant)"]))

    print("\nObservation on Promax:")
    print("- By relaxing the rigid 90-degree orthogonality constraint, Promax accommodates the natural")
    print("  correlation between real-world traits (e.g., verbal and mathematical aptitude).")
    print("- Notice how minor cross-loadings are driven even closer to zero than under Varimax!\n")

    # -------------------------------------------------------------------------
    # 4. Comparative Summary of the Rotation Family
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("4. COMPARATIVE ROTATION CHEAT-SHEET: WHEN TO USE WHICH METHOD")
    print("-" * 80)
    methods = [
        ("Raw PCA", "none", "Compression & filtering only; global variance maximization."),
        ("Varimax", "varimax", "RECOMMENDED DEFAULT. Orthogonal, isolates distinct domain factors."),
        ("Quartimax", "quartimax", "Simplifies variables; use when each feature must belong to only 1 factor."),
        ("Equamax", "equamax", "Balanced compromise when number of factors k is moderate to large."),
        ("Parsimax", "parsimax", "Optimal when number of variables p is small relative to factors k."),
        ("Biquartimax", "biquartimax", "Slightly less aggressive factor simplification than Varimax (gamma=0.5)."),
        ("Promax", "promax", "RECOMMENDED FOR SOCIAL/ECONOMIC DATA. Permits realistic oblique factor correlation."),
    ]
    summary_df = pd.DataFrame(methods, columns=["Method", "Argument", "When to Use"])
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_educational_demonstration()
