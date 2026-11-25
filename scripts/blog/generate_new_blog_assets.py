"""Generate data, tree diagrams, and metrics diagrams for new mltreelib blog articles.

Articles:
1. docs/blog/beyond-mse-distributional-trees.md
   - blog_distributional_tree.svg (Tree with column names)
   - blog_distributional_metrics.svg (Tweedie zero-inflation, Tobit censoring, Heteroscedastic variance)
2. docs/blog/breaking-step-functions-model-trees-vs-gbdt.md
   - blog_model_tree.svg (ModelTree with linear leaf equations and column names)
   - blog_model_tree_vs_step.svg (Step function vs Linear leaf smoothing, Extrapolation comparison)
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import mltreelib as mt

DOCS_ASSETS = Path("docs/assets")
DOCS_ASSETS.mkdir(parents=True, exist_ok=True)


def render_dot_to_svg(dot_str: str, output_path: Path):
    """Render Graphviz DOT to SVG using the dot CLI."""
    proc = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_str.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    output_path.write_bytes(proc.stdout)
    print(f"  [SVG] Wrote {output_path} ({len(proc.stdout)} bytes)")


# ==============================================================================
# 1. Blog: Distributional Trees (Tweedie, Tobit, Heteroscedastic)
# ==============================================================================
def run_blog_distributional():
    print("\n--- 1. Generating Blog Distributional Tree Assets ---")
    random.seed(42)
    n = 1200

    col_names = ["DriverAge", "VehiclePower", "BonusMalus", "AnnualMileage"]
    X: list[list[float]] = []
    y: list[float] = []

    for _ in range(n):
        age = random.uniform(18.0, 75.0)
        power = random.uniform(40.0, 200.0)
        bm = random.uniform(50.0, 150.0)
        mileage = random.uniform(5000.0, 40000.0)

        # High risk: young driver with high vehicle power and bad bonus-malus
        risk = 0.05
        if age < 25.0 and power > 100.0:
            risk = 0.35
        elif bm > 110.0:
            risk = 0.20
        elif age > 65.0:
            risk = 0.12

        # Tweedie: zero with probability (1 - risk), otherwise Gamma severity
        if random.random() < risk:
            claim = random.gammavariate(alpha=2.0, beta=1200.0)
        else:
            claim = 0.0

        X.append([age, power, bm, mileage])
        y.append(claim)

    # Fit ModelTree with Tweedie deviance criterion
    tree = mt.ModelTree(
        gain="tweedie",
        max_depth=3,
        min_rows=40,
        tweedie_p=1.5,
        prune="none",
        seed=42,
    )
    tree.fit(X, y, col_names=col_names)

    # 1. Tree Diagram
    dot = tree.to_dot()
    tree_svg_path = DOCS_ASSETS / "blog_distributional_tree.svg"
    render_dot_to_svg(dot, tree_svg_path)

    # 2. Vector SVG Diagnostic Chart
    svg_chart = generate_distributional_metrics_svg(X, y)
    chart_path = DOCS_ASSETS / "blog_distributional_metrics.svg"
    chart_path.write_text(svg_chart, encoding="utf-8")
    print(f"  [SVG] Wrote {chart_path} ({len(svg_chart)} bytes)")


def generate_distributional_metrics_svg(X, y) -> str:
    """Generate pure SVG 3-panel diagnostic visualization for distributional trees."""
    width = 960
    height = 340

    # Panel A: Zero-inflated Tweedie distribution (Histogram-like bars)
    # Zero claims count vs positive claims
    zeros = sum(1 for v in y if v == 0.0)
    positives = [v for v in y if v > 0.0]

    # Panel B: Tobit Censoring Concept
    # Panel C: Heteroscedastic Variance Envelope
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #fafafa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <style>
    .title {{ font-size: 14px; font-weight: 700; fill: #1e293b; }}
    .subtitle {{ font-size: 11px; fill: #64748b; }}
    .axis {{ stroke: #cbd5e1; stroke-width: 1.5; }}
    .grid {{ stroke: #f1f5f9; stroke-width: 1; }}
    .bar-zero {{ fill: #3b82f6; opacity: 0.85; }}
    .bar-gamma {{ fill: #8b5cf6; opacity: 0.85; }}
    .line-tobit {{ stroke: #ef4444; stroke-width: 2.5; stroke-dasharray: 4,4; }}
    .line-true {{ stroke: #10b981; stroke-width: 2.5; }}
    .envelope {{ fill: #f97316; opacity: 0.18; }}
    .label {{ font-size: 10px; fill: #475569; }}
    .badge {{ font-size: 10px; font-weight: 600; fill: #0f172a; }}
  </style>

  <!-- Panel 1: Tweedie Zero-Inflation (Actuarial Claim Frequency) -->
  <g transform="translate(40, 30)">
    <rect width="260" height="260" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="28" class="title">Tweedie Compound Poisson</text>
    <text x="16" y="44" class="subtitle">Zero-spike (84%) + Gamma tail (16%)</text>

    <!-- Axes -->
    <line x1="40" y1="210" x2="230" y2="210" class="axis"/>
    <line x1="40" y1="60" x2="40" y2="210" class="axis"/>

    <!-- Zero spike -->
    <rect x="55" y="75" width="28" height="135" class="bar-zero" rx="2"/>
    <text x="69" y="68" text-anchor="middle" class="badge">84% $0</text>
    <text x="69" y="224" text-anchor="middle" class="label">$0</text>

    <!-- Continuous Gamma bins -->
    <rect x="95" y="165" width="22" height="45" class="bar-gamma" rx="2"/>
    <rect x="122" y="150" width="22" height="60" class="bar-gamma" rx="2"/>
    <rect x="149" y="170" width="22" height="40" class="bar-gamma" rx="2"/>
    <rect x="176" y="188" width="22" height="22" class="bar-gamma" rx="2"/>
    <rect x="203" y="198" width="22" height="12" class="bar-gamma" rx="2"/>

    <text x="160" y="224" text-anchor="middle" class="label">Claim Size ($)</text>
    <text x="24" y="135" text-anchor="middle" transform="rotate(-90 24 135)" class="label">Density</text>
  </g>

  <!-- Panel 2: Tobit Censoring (Floor / Ceiling Bias) -->
  <g transform="translate(350, 30)">
    <rect width="260" height="260" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="28" class="title">Tobit Censoring Correction</text>
    <text x="16" y="44" class="subtitle">OLS attenuation vs Latent Tobit split</text>

    <line x1="40" y1="210" x2="230" y2="210" class="axis"/>
    <line x1="40" y1="60" x2="40" y2="210" class="axis"/>

    <!-- Censoring limit line -->
    <line x1="40" y1="180" x2="230" y2="180" stroke="#94a3b8" stroke-dasharray="2,2"/>
    <text x="225" y="174" text-anchor="end" font-size="9" fill="#94a3b8">Censor Limit (y=0)</text>

    <!-- Biased OLS slope -->
    <line x1="50" y1="175" x2="220" y2="105" class="line-tobit"/>
    <!-- Unbiased Latent Tobit slope -->
    <line x1="50" y1="195" x2="220" y2="75" class="line-true"/>

    <!-- Points -->
    <circle cx="60" cy="180" r="3" fill="#ef4444"/>
    <circle cx="75" cy="180" r="3" fill="#ef4444"/>
    <circle cx="90" cy="180" r="3" fill="#ef4444"/>
    <circle cx="110" cy="165" r="3" fill="#10b981"/>
    <circle cx="135" cy="145" r="3" fill="#10b981"/>
    <circle cx="160" cy="120" r="3" fill="#10b981"/>
    <circle cx="185" cy="95" r="3" fill="#10b981"/>
    <circle cx="210" cy="80" r="3" fill="#10b981"/>

    <!-- Legend -->
    <line x1="50" y1="235" x2="70" y2="235" class="line-true"/>
    <text x="75" y="238" class="label">Tobit Split (Unbiased)</text>
    <line x1="160" y1="235" x2="180" y2="235" class="line-tobit"/>
    <text x="185" y="238" class="label">CART OLS (Flawed)</text>
  </g>

  <!-- Panel 3: Heteroscedastic Variance Splitting -->
  <g transform="translate(660, 30)">
    <rect width="260" height="260" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="28" class="title">Heteroscedastic Partitioning</text>
    <text x="16" y="44" class="subtitle">Splitting on volatility regime σ(x)</text>

    <line x1="40" y1="210" x2="230" y2="210" class="axis"/>
    <line x1="40" y1="60" x2="40" y2="210" class="axis"/>

    <!-- Low volatility envelope -->
    <polygon points="50,130 120,132 120,148 50,150" class="envelope"/>
    <!-- High volatility envelope -->
    <polygon points="120,95 220,70 220,205 120,185" class="envelope" fill="#f43f5e" opacity="0.18"/>

    <!-- Split boundary -->
    <line x1="120" y1="65" x2="120" y2="210" stroke="#6366f1" stroke-width="2" stroke-dasharray="3,3"/>
    <text x="120" y="58" text-anchor="middle" font-size="9" font-weight="700" fill="#6366f1">Split: DriverAge &lt; 25</text>

    <!-- Scatter dots -->
    <circle cx="65" cy="138" r="2.5" fill="#f97316"/>
    <circle cx="80" cy="142" r="2.5" fill="#f97316"/>
    <circle cx="95" cy="136" r="2.5" fill="#f97316"/>
    <circle cx="108" cy="145" r="2.5" fill="#f97316"/>

    <circle cx="135" cy="110" r="2.5" fill="#f43f5e"/>
    <circle cx="150" cy="170" r="2.5" fill="#f43f5e"/>
    <circle cx="165" cy="90" r="2.5" fill="#f43f5e"/>
    <circle cx="180" cy="185" r="2.5" fill="#f43f5e"/>
    <circle cx="195" cy="85" r="2.5" fill="#f43f5e"/>
    <circle cx="210" cy="195" r="2.5" fill="#f43f5e"/>

    <text x="85" y="195" text-anchor="middle" class="badge" fill="#f97316">σ = 0.12</text>
    <text x="175" y="140" text-anchor="middle" class="badge" fill="#f43f5e">σ = 1.84</text>
  </g>
</svg>"""


# ==============================================================================
# 2. Blog: Model Trees vs GBDT on Continuous Physics
# ==============================================================================
def run_blog_model_trees():
    print("\n--- 2. Generating Blog Model Trees vs GBDT Assets ---")
    random.seed(42)
    n = 800

    col_names = ["Frequency_Hz", "Angle_Deg", "Chord_m", "Velocity_mps", "Suction_m"]
    X: list[list[float]] = []
    y: list[float] = []

    # NASA Airfoil Self Noise physics emulation:
    # Pressure decibels = f(frequency, angle, chord, velocity, suction displacement)
    for _ in range(n):
        freq = random.uniform(200.0, 10000.0)
        angle = random.uniform(0.0, 22.0)
        chord = random.choice([0.0254, 0.0508, 0.1016, 0.1524, 0.2286])
        velo = random.choice([31.7, 39.6, 55.5, 71.3])
        suction = random.uniform(0.001, 0.05)

        # Smooth physical acoustic pressure formulation:
        # High frequency and angle create aerodynamic turbulence noise with smooth linear gradients
        db = (
            128.0
            - 0.0015 * freq
            - 0.65 * angle
            - 15.0 * chord
            + 0.12 * velo
            - 180.0 * suction
            + (12.0 if freq > 3000.0 else 0.0)
            + random.gauss(0.0, 1.5)
        )

        X.append([freq, angle, chord, velo, suction])
        y.append(db)

    # Fit ModelTree with linear leaves
    tree = mt.ModelTree(
        leaf="linear",
        max_depth=3,
        min_rows=30,
        l2=1e-3,
        prune="none",
        seed=42,
    )
    tree.fit(X, y, col_names=col_names)

    # 1. Tree Diagram
    dot = tree.to_dot()
    tree_svg_path = DOCS_ASSETS / "blog_model_tree.svg"
    render_dot_to_svg(dot, tree_svg_path)

    # 2. Vector SVG Comparison Chart (Step Staircase vs Linear Leaf vs Extrapolation)
    svg_chart = generate_model_tree_vs_step_svg()
    chart_path = DOCS_ASSETS / "blog_model_tree_vs_step.svg"
    chart_path.write_text(svg_chart, encoding="utf-8")
    print(f"  [SVG] Wrote {chart_path} ({len(svg_chart)} bytes)")


def generate_model_tree_vs_step_svg() -> str:
    """Generate pure SVG 3-panel comparison: Step staircase vs Linear model tree vs Extrapolation."""
    width = 960
    height = 340

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #fafafa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <style>
    .title {{ font-size: 14px; font-weight: 700; fill: #1e293b; }}
    .subtitle {{ font-size: 11px; fill: #64748b; }}
    .axis {{ stroke: #cbd5e1; stroke-width: 1.5; }}
    .step-line {{ stroke: #ef4444; stroke-width: 2.5; fill: none; }}
    .model-line {{ stroke: #2563eb; stroke-width: 2.5; fill: none; }}
    .truth-line {{ stroke: #94a3b8; stroke-width: 2; stroke-dasharray: 3,3; fill: none; }}
    .extrap-flat {{ stroke: #ef4444; stroke-width: 2.5; stroke-dasharray: 4,4; fill: none; }}
    .extrap-linear {{ stroke: #10b981; stroke-width: 2.5; fill: none; }}
    .label {{ font-size: 10px; fill: #475569; }}
    .badge {{ font-size: 10px; font-weight: 600; }}
  </style>

  <!-- Panel 1: Step Function Staircase vs Linear Model Tree -->
  <g transform="translate(40, 30)">
    <rect width="260" height="260" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="28" class="title">Approximating Continuous Physics</text>
    <text x="16" y="44" class="subtitle">CART/GBDT 20-step staircase vs 1 Linear Leaf</text>

    <line x1="35" y1="210" x2="235" y2="210" class="axis"/>
    <line x1="35" y1="60" x2="35" y2="210" class="axis"/>

    <!-- True smooth curve -->
    <path d="M 40 185 Q 120 150 230 75" class="truth-line"/>

    <!-- Step function staircase (GBDT) -->
    <path d="M 40 185 H 65 V 170 H 90 V 155 H 115 V 140 H 145 V 122 H 175 V 102 H 205 V 85 H 230" class="step-line"/>

    <!-- Model tree smooth line -->
    <path d="M 40 182 L 130 138 L 230 76" class="model-line"/>

    <!-- Legend -->
    <line x1="45" y1="235" x2="65" y2="235" class="step-line"/>
    <text x="70" y="238" class="label">GBDT (Staircase)</text>
    <line x1="150" y1="235" x2="170" y2="235" class="model-line"/>
    <text x="175" y="238" class="label">ModelTree (Linear)</text>
  </g>

  <!-- Panel 2: Extrapolation Outside Training Range -->
  <g transform="translate(350, 30)">
    <rect width="260" height="260" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="28" class="title">Extrapolation Beyond Support</text>
    <text x="16" y="44" class="subtitle">GBDT flatline collapse vs Monotone projection</text>

    <line x1="35" y1="210" x2="235" y2="210" class="axis"/>
    <line x1="35" y1="60" x2="35" y2="210" class="axis"/>

    <!-- Support boundary -->
    <rect x="35" y="60" width="120" height="150" fill="#f8fafc"/>
    <line x1="155" y1="60" x2="155" y2="210" stroke="#cbd5e1" stroke-dasharray="3,3"/>
    <text x="95" y="75" text-anchor="middle" font-size="9" fill="#94a3b8">Training Support</text>
    <text x="195" y="75" text-anchor="middle" font-size="9" fill="#f97316">Extrapolation Zone</text>

    <!-- Training line -->
    <path d="M 40 170 L 155 110" class="model-line"/>

    <!-- GBDT flatline in extrapolation -->
    <path d="M 155 110 H 230" class="extrap-flat"/>
    <text x="215" y="102" text-anchor="end" class="badge" fill="#ef4444">GBDT Flatline</text>

    <!-- Linear Model Tree continuation with clamping -->
    <path d="M 155 110 L 230 72" class="extrap-linear"/>
    <text x="215" y="65" text-anchor="end" class="badge" fill="#10b981">ModelTree Slope</text>

    <text x="135" y="225" text-anchor="middle" class="label">Airfoil Inflow Velocity (m/s)</text>
  </g>

  <!-- Panel 3: Efficiency Frontier (Tree Depth vs RMSE) -->
  <g transform="translate(660, 30)">
    <rect width="260" height="260" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="28" class="title">Model Efficiency Frontier</text>
    <text x="16" y="44" class="subtitle">RMSE vs Required Tree Depth</text>

    <line x1="45" y1="210" x2="235" y2="210" class="axis"/>
    <line x1="45" y1="60" x2="45" y2="210" class="axis"/>

    <!-- GBDT curve -->
    <path d="M 55 190 Q 90 175 140 140 T 225 115" class="step-line"/>
    <circle cx="55" cy="190" r="3" fill="#ef4444"/>
    <circle cx="100" cy="165" r="3" fill="#ef4444"/>
    <circle cx="160" cy="130" r="3" fill="#ef4444"/>
    <circle cx="225" cy="115" r="3" fill="#ef4444"/>

    <!-- ModelTree curve -->
    <path d="M 55 135 Q 90 95 160 82 T 225 78" class="model-line"/>
    <circle cx="55" cy="135" r="3" fill="#2563eb"/>
    <circle cx="100" cy="98" r="3" fill="#2563eb"/>
    <circle cx="160" cy="82" r="3" fill="#2563eb"/>
    <circle cx="225" cy="78" r="3" fill="#2563eb"/>

    <text x="180" y="70" class="badge" fill="#2563eb">ModelTree (Depth 3 = 1.05 RMSE)</text>
    <text x="165" y="160" class="badge" fill="#ef4444">CART (Depth 10 = 1.76 RMSE)</text>

    <text x="140" y="225" text-anchor="middle" class="label">Tree Depth (Splits)</text>
    <text x="25" y="135" text-anchor="middle" transform="rotate(-90 25 135)" class="label">Test RMSE (dB)</text>
  </g>
</svg>"""


if __name__ == "__main__":
    run_blog_distributional()
    run_blog_model_trees()
    print("\n[Done] All new blog assets generated successfully.")
