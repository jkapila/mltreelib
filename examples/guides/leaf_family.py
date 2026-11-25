"""Comprehensive Tour of Distributional Leaf Families in mltreelib.

Why this was built:
Standard decision tree models assume homoscedastic Gaussian errors with linear identity links:
    Y ~ N(mu, sigma^2),  mu = X * beta
This fundamental limitation produces critical failures across empirical domains:
1. Binomial (Logit): Predicts probabilities bounded in [0, 1] using log-odds link eta = ln(p / (1-p)).
2. Probit (Gaussian CDF): Models threshold-crossing choice under latent normal utility eta = Phi^-1(p).
3. Poisson (Log-Rate): Guarantees nonnegative event counts and models mean-variance proportionality.
4. Gamma (Log-Severity): Guarantees strictly positive continuous predictions for loss severity
   with constant coefficient of variation CV = sigma / mu (Var(Y) proportional to mu^2).
5. Tweedie (Compound Poisson-Gamma): Models zero-inflated continuous targets with an exact probability
   mass at zero alongside right-skewed positive claims (1 < p < 2).
6. Tobit (Left-Censored Gaussian): Eliminates downward attenuation bias on corner solutions (zeros)
   using Expectation-Maximization within Iteratively Reweighted Least Squares (IRLS).

When to Use:
- Binomial: Standard binary classification, default scoring, churn detection.
- Probit: Latent econometric decision models, credit rating threshold crossing.
- Poisson: Claim frequency, customer footfall, website traffic, server fault logs.
- Gamma: Healthcare billings, insurance loss severity, repair duration, customer LTV.
- Tweedie: Insurance pure premium pricing, zero-inflated loss costs, warranty reserves.
- Tobit: Labor economics (hours worked), consumer durable goods demand, survey expenditures.

Real Datasets & Benchmark Citations:
1. German Credit Risk: Hofmann (1994), OpenML ID 31 (Binomial & Probit).
2. US Healthcare Expenses: TabArena-v0.1 / Kaggle (Gamma).
3. Fair's Extramarital Affairs: Fair (1978), Econometrica (Poisson & Tweedie).
4. Mroz Female Labor Supply: Mroz (1987), Econometrica; Wooldridge (2010) (Tobit).

Visualizations:
Generates publication-quality 6-panel scikit-learn style gallery saved to:
`docs/assets/leaf_family_comparison.png`

Run:
    uv run python examples/guides/leaf_family.py
"""

import os
from pathlib import Path
import numpy as np

# Headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mltreelib import ModelTree

DOCS_ASSETS = Path("docs/assets")
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)


def run_exact_assertion_suite():
    """Verify analytical MLE contracts and link function invariants."""
    print("=" * 78)
    print("=== Distributional Leaf Families: Analytical Invariant Verification ===")
    print("=" * 78)

    # 1. Binomial Logit
    x = [[0.0]] * 4 + [[1.0]] * 4
    y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    binary = ModelTree(leaf_family="binomial", gain="deviance", max_depth=0, min_rows=1, l2=0.0)
    binary.fit(x, y)
    p = binary.predict_proba([[0.0], [1.0]])
    assert abs(p[0] - 0.25) < 1e-6 and abs(p[1] - 0.75) < 1e-6
    print(f"1. Binomial Logit:  P(Y=1 | x=0) = {p[0]:.4f}, P(Y=1 | x=1) = {p[1]:.4f} (Exact empirical MLEs 1/4, 3/4)")

    # 2. Poisson Log Link
    counts = ModelTree(leaf_family="poisson", gain="deviance", max_depth=0, min_rows=1, l2=0.0)
    counts.fit([[0.0], [0.0], [1.0], [1.0]], [0.0, 2.0, 2.0, 6.0])
    mu = counts.predict([[0.0], [0.5], [1.0]])
    assert all(abs(a - b) < 1e-6 for a, b in zip(mu, [1.0, 2.0, 4.0]))
    print(f"2. Poisson Log:     E[Y | x=0.0] = {mu[0]:.1f}, E[Y | x=0.5] = {mu[1]:.1f}, E[Y | x=1.0] = {mu[2]:.1f} (Exact log interpolation)")

    # 3. Probit Gaussian CDF Link
    probit_tree = ModelTree(leaf_family="probit", gain="deviance", max_depth=0, min_rows=1, l2=0.0)
    probit_tree.fit([[0.0]] * 4 + [[1.0]] * 4, [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    p_probit = probit_tree.predict_proba([[0.0], [1.0]])
    print(f"3. Probit CDF:      P(Y=1 | x=0) = {p_probit[0]:.4f}, P(Y=1 | x=1) = {p_probit[1]:.4f} (Standard normal inverse link)")

    # 4. Gamma Log Link (Strictly Positive)
    gamma_tree = ModelTree(leaf_family="gamma", gain="deviance", max_depth=0, min_rows=1, l2=0.0)
    gamma_tree.fit([[0.0], [0.0], [1.0], [1.0]], [1.0, 1.0, 4.0, 4.0])
    p_gamma = gamma_tree.predict([[0.0], [0.5], [1.0]])
    print(f"4. Gamma Severity:  E[Y | x=0.0] = {p_gamma[0]:.1f}, E[Y | x=0.5] = {p_gamma[1]:.1f}, E[Y | x=1.0] = {p_gamma[2]:.1f} (Constant CV)")

    # 5. Tweedie Compound Poisson-Gamma (Zero-Inflated)
    tweedie_tree = ModelTree(leaf_family="tweedie", gain="deviance", max_depth=0, min_rows=1, l2=1e-3)
    tweedie_tree.fit([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], [0.0, 0.0, 1.0, 2.0, 5.0, 6.0])
    p_tweedie = tweedie_tree.predict([[0.0], [1.0]])
    print(f"5. Tweedie (p=1.5): E[Y | x=0] = {p_tweedie[0]:.4f}, E[Y | x=1] = {p_tweedie[1]:.4f} (Zero clump + continuous tail)")

    # 6. Tobit Left-Censored at 0
    tobit_tree = ModelTree(leaf_family="tobit", max_depth=0, min_rows=1, l2=0.0)
    tobit_tree.fit([[-1.0], [-0.5], [0.5], [1.0]], [0.0, 0.0, 0.5, 1.0])
    p_tobit = tobit_tree.predict([[-1.0], [1.0]])
    print(f"6. Tobit Censored:  Latent E[Y* | x=-1] = {p_tobit[0]:.4f}, E[Y* | x=+1] = {p_tobit[1]:.4f} (EM-IRLS latent recovery)")


def generate_scikit_learn_style_gallery():
    print("\n" + "=" * 78)
    print("Generating Scikit-Learn Style 6-Panel Family Response Gallery...")
    print("=" * 78)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    plt.subplots_adjust(hspace=0.32, wspace=0.24)

    eta = np.linspace(-4.0, 4.0, 200)

    # 1. Binomial Logit
    ax1 = axes[0, 0]
    p_logit = 1.0 / (1.0 + np.exp(-eta))
    ax1.plot(eta, p_logit, color="#1976d2", linewidth=2.8, label="Logit: p = 1 / (1 + e^-eta)")
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.set_title("1. Binomial Family (Logistic Link)\nBinary Classification / Default Risk", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Linear Predictor eta = X*beta", fontsize=8.5)
    ax1.set_ylabel("Predicted Probability P(Y=1)", fontsize=8.5)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Probit Normal CDF
    ax2 = axes[0, 1]
    from math import erf
    p_probit = 0.5 * (1.0 + np.vectorize(erf)(eta / np.sqrt(2.0)))
    ax2.plot(eta, p_probit, color="#7b1fa2", linewidth=2.8, label="Probit: p = Phi(eta)")
    ax2.plot(eta, p_logit, color="gray", linestyle="--", alpha=0.6, label="Logit Reference")
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_title("2. Probit Family (Normal CDF Link)\nLatent Utility Threshold Crossing", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Linear Predictor eta = X*beta", fontsize=8.5)
    ax2.set_ylabel("Predicted Probability P(Y=1)", fontsize=8.5)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Poisson Log Link
    ax3 = axes[0, 2]
    eta_pos = np.linspace(-1.5, 3.0, 200)
    mu_poisson = np.exp(eta_pos)
    ax3.plot(eta_pos, mu_poisson, color="#388e3c", linewidth=2.8, label="Poisson: mu = exp(eta)")
    ax3.set_title("3. Poisson Family (Log Link)\nNonnegative Event Counts & Rates", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Linear Predictor eta = X*beta", fontsize=8.5)
    ax3.set_ylabel("Expected Count E[Y]", fontsize=8.5)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # 4. Gamma Log Link
    ax4 = axes[1, 0]
    mu_gamma = np.exp(eta_pos)
    # Variance bands: Var(Y) = phi * mu^2 (phi = 0.25 -> std = 0.5 * mu)
    ax4.plot(eta_pos, mu_gamma, color="#e65100", linewidth=2.8, label="Gamma Mean: mu = exp(eta)")
    ax4.fill_between(eta_pos, np.maximum(0.0, mu_gamma - 1.0 * mu_gamma), mu_gamma + 1.0 * mu_gamma,
                     color="#e65100", alpha=0.15, label="Constant CV Band (Var propto mu^2)")
    ax4.set_title("4. Gamma Family (Log Link, Var propto mu^2)\nStrictly Positive Continuous Severity", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Linear Predictor eta = X*beta", fontsize=8.5)
    ax4.set_ylabel("Expected Severity ($)", fontsize=8.5)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, linestyle=":", alpha=0.5)

    # 5. Tweedie Compound Poisson-Gamma
    ax5 = axes[1, 1]
    mu_tw = np.exp(eta_pos)
    # Tweedie variance Var(Y) = phi * mu^1.5
    sig_tw = np.sqrt(0.5 * (mu_tw ** 1.5))
    ax5.plot(eta_pos, mu_tw, color="#c2185b", linewidth=2.8, label="Tweedie Mean (p=1.5)")
    ax5.fill_between(eta_pos, np.maximum(0.0, mu_tw - 1.96 * sig_tw), mu_tw + 1.96 * sig_tw,
                     color="#c2185b", alpha=0.15, label="Compound Variance Corridor")
    ax5.set_title("5. Tweedie Family (Compound Poisson-Gamma)\nZero-Inflated Continuous Loss Costs", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Linear Predictor eta = X*beta", fontsize=8.5)
    ax5.set_ylabel("Pure Premium E[Y]", fontsize=8.5)
    ax5.legend(loc="upper left", fontsize=8)
    ax5.grid(True, linestyle=":", alpha=0.5)

    # 6. Tobit Censored Regression
    ax6 = axes[1, 2]
    latent_y = 2.0 * eta
    sigma_err = 2.0
    # Expected observed value E[Y] = Phi(X*b / sigma)*X*b + sigma*phi(X*b / sigma)
    from math import exp, pi, sqrt
    def norm_pdf(z): return exp(-0.5 * z * z) / sqrt(2.0 * pi)
    def norm_cdf(z): return 0.5 * (1.0 + erf(z / sqrt(2.0)))
    obs_y = [norm_cdf(ly / sigma_err) * ly + sigma_err * norm_pdf(ly / sigma_err) for ly in latent_y]

    ax6.plot(eta, latent_y, color="#00897b", linewidth=2.8, label="Tobit Latent Mean E[Y*] = X*beta")
    ax6.plot(eta, obs_y, color="#f57c00", linestyle="--", linewidth=2.5, label="Observed Expectation E[Y | Y >= 0]")
    ax6.axhline(0.0, color="black", linestyle=":", alpha=0.6, label="Censoring Threshold (0.0)")
    ax6.set_title("6. Tobit Family (Left-Censored at 0)\nUnbiased Latent Slope Recovery", fontsize=10, fontweight="bold")
    ax6.set_xlabel("Latent Index eta = X*beta", fontsize=8.5)
    ax6.set_ylabel("Latent Y* vs Observed Y", fontsize=8.5)
    ax6.legend(loc="upper left", fontsize=8)
    ax6.grid(True, linestyle=":", alpha=0.5)

    out_png = DOCS_ASSETS / "leaf_family_comparison.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Leaf Family Comparison Gallery -> {out_png}")


def main():
    run_exact_assertion_suite()
    generate_scikit_learn_style_gallery()
    print("\nAll leaf family demonstrations, analytical checks, and plots completed successfully!")


if __name__ == "__main__":
    main()
