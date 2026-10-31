"""Plotting for the metrics module — matplotlib is an OPTIONAL extra.

Every function renders one chart from data that `mltreelib.Metrics` returns
(or computes it from `y_true`/scores itself). The library core stays
plotting-free; this module is the thin, standard rendering layer (matplotlib
is the wheel, we do not reinvent charting).

    uv run --with matplotlib python -c "from mltreelib import plots"

Conventions: each function takes `ax=None` (creates a figure when absent) and
returns the Axes, so charts compose into subplots.
"""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "mltreelib.plots needs matplotlib (a dev extra): uv run --with matplotlib ..."
    ) from e

from mltreelib import Metrics

# Unified color palette matching mltreelib documentation theme (Teal & Deep Orange)
PALETTE = {
    "primary": "#00897b",        # Teal (main curve, model line, positive waterfall)
    "accent": "#f4511e",         # Deep Orange (comparison curve, negative waterfall, second model)
    "secondary": "#0288d1",      # Blue / Cerulean (lift, reference indicators)
    "muted": "#80cbc4",          # Soft Teal (background bars, distribution shares)
    "dark": "#263238",           # Charcoal (actuals, final totals)
    "gray": "#78909c",           # Slate gray (baselines, secondary bars)
    "light_neutral": "#b0bec5",  # Light gray (diagonal random lines, thresholds)
    "warning": "#fbc02d",        # Amber (residual waterfall steps)
}


def _ax(ax):
    if ax is not None:
        return ax, None
    _, ax = plt.subplots()
    return ax, True


def _style_axis(ax, grid: bool = True):
    """Apply clean, modern styling to a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cfd8dc")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#cfd8dc")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors="#37474f", labelsize=9)
    ax.xaxis.label.set_color("#263238")
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_color("#263238")
    ax.yaxis.label.set_size(10)
    ax.title.set_color("#263238")
    ax.title.set_size(11)
    ax.title.set_weight("semibold")
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#cfd8dc", zorder=0)


# ---------------------------------------------------------------- ranking

def plot_roc(y_true, scores, ax=None):
    """ROC curve + AUC annotation (Metrics().roc_curve)."""
    ax, own = _ax(ax)
    _style_axis(ax)
    r = Metrics().roc_curve(y_true, scores)
    pts = r["points"]
    fpr = [p[0] for p in pts]
    tpr = [p[1] for p in pts]
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.0, label=f"Model (AUC = {r['auc']:.3f})", zorder=3)
    ax.fill_between(fpr, tpr, color=PALETTE["muted"], alpha=0.15, zorder=2)
    ax.plot([0, 1], [0, 1], color=PALETTE["light_neutral"], lw=1.0, ls="--", label="Random", zorder=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC = {r['auc']:.3f})")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="lower right")
    if own:
        return ax


def plot_pr(y_true, scores, ax=None):
    """Precision-recall curve + average precision annotation."""
    ax, own = _ax(ax)
    _style_axis(ax)
    r = Metrics().pr_curve(y_true, scores)
    pts = r["points"]
    rec = [p[0] for p in pts]
    prec = [p[1] for p in pts]
    prev = sum(1 for y in y_true if y > 0) / max(len(y_true), 1)
    ax.plot(rec, prec, color=PALETTE["primary"], lw=2.0, label=f"Model (AP = {r['average_precision']:.3f})", zorder=3)
    ax.fill_between(rec, prec, color=PALETTE["muted"], alpha=0.15, zorder=2)
    ax.axhline(prev, color=PALETTE["light_neutral"], lw=1.0, ls="--", label=f"Baseline ({prev:.2f})", zorder=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall (AP = {r['average_precision']:.3f})")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="lower left")
    if own:
        return ax


def plot_cumulative_gain(y_true, scores, ax=None):
    """Cumulative gains (modelplotpy/kds): capture vs population fraction."""
    ax, own = _ax(ax)
    _style_axis(ax)
    pts = Metrics().gain_curve(y_true, scores)["points"]
    pop = [p[0] for p in pts]
    cap = [p[1] for p in pts]
    ax.plot(pop, cap, color=PALETTE["primary"], lw=2.0, label="Model", zorder=3)
    ax.fill_between(pop, cap, pop, color=PALETTE["muted"], alpha=0.15, zorder=2)
    ax.plot([0, 1], [0, 1], color=PALETTE["light_neutral"], lw=1.0, ls="--", label="Random baseline", zorder=1)
    ax.set_xlabel("Population Fraction")
    ax.set_ylabel("Cumulative Capture Rate")
    ax.set_title("Cumulative Gains")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="lower right")
    if own:
        return ax


def plot_lift(y_true, scores, ax=None):
    """Cumulative lift (kds/modelplotpy): capture / population fraction."""
    ax, own = _ax(ax)
    _style_axis(ax)
    pts = Metrics().lift_curve(y_true, scores)["points"]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PALETTE["secondary"], lw=2.0, label="Cumulative Lift", zorder=3)
    ax.axhline(1.0, color=PALETTE["accent"], lw=1.0, ls="--", label="Baseline (Lift = 1.0)", zorder=1)
    ax.set_xlabel("Population Fraction")
    ax.set_ylabel("Lift Multiplier")
    ax.set_title("Cumulative Lift")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="upper right")
    if own:
        return ax


def plot_ks(y_true, scores, ax=None):
    """KS statistic chart (kds/PerfScoreCard): TPR and FPR curves with the
    maximum gap marked."""
    ax, own = _ax(ax)
    _style_axis(ax)
    r = Metrics().ks_statistic(y_true, scores)
    pts = r["points"]
    sc = [p[2] for p in pts]
    tpr = [p[1] for p in pts]
    fpr = [p[0] for p in pts]
    ax.plot(sc, tpr, color=PALETTE["primary"], lw=1.8, label="TPR (Events)", zorder=3)
    ax.plot(sc, fpr, color=PALETTE["accent"], lw=1.8, label="FPR (Non-events)", zorder=3)
    ax.axvline(r["score_at_ks"], color=PALETTE["secondary"], lw=1.2, ls="--", zorder=2)
    ax.annotate(
        f"KS = {r['ks']:.3f}\nThreshold = {r['score_at_ks']:.3f}",
        xy=(r["score_at_ks"], (r["tpr_at_ks"] + r["fpr_at_ks"]) / 2),
        xytext=(0.06, 0.82),
        textcoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#b0bec5", lw=0.8),
        fontsize=8.5,
    )
    ax.set_xlabel("Score Threshold (descending)")
    ax.set_ylabel("Cumulative Rate")
    ax.set_title("Kolmogorov-Smirnov (KS) Statistic")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="center right")
    if own:
        return ax


def plot_decile(y_true, scores, n_buckets: int = 10, ax=None):
    """Decile profile (kds/PerfScoreCard): event rate bars + cumulative capture line."""
    ax, own = _ax(ax)
    _style_axis(ax, grid=False)
    rows = Metrics().decile_table(y_true, scores, n_buckets)
    d = [r["decile"] for r in rows]
    rates = [r["event_rate"] for r in rows]
    caps = [r["cum_capture"] for r in rows]

    ax.bar(d, rates, color=PALETTE["muted"], edgecolor="#4db6ac", linewidth=0.8, label="Event rate", zorder=2)
    ax.set_xlabel("Decile (1 = highest score)")
    ax.set_ylabel("Decile Event Rate", color=PALETTE["dark"])
    ax.tick_params(axis="y", labelcolor=PALETTE["dark"])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#cfd8dc")
    ax2.plot(d, caps, color=PALETTE["accent"], lw=2.0, marker="o", markersize=4.5, label="Cum. capture", zorder=4)
    ax2.set_ylabel("Cumulative Capture", color=PALETTE["accent"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])

    ax.set_title("Decile Performance Profile")
    if own:
        return ax


def plot_calibration(y_true, scores, n_bins: int = 10, ax=None):
    """Reliability diagram (calibration curve) vs the diagonal."""
    ax, own = _ax(ax)
    _style_axis(ax)
    bins = Metrics().calibration(y_true, scores, n_bins)["bins"]
    mean_scores = [b["mean_score"] for b in bins]
    observed = [b["observed"] for b in bins]
    ax.plot([0, 1], [0, 1], color=PALETTE["light_neutral"], lw=1.0, ls="--", label="Perfect calibration", zorder=1)
    ax.plot(mean_scores, observed, marker="o", markersize=5, lw=1.8, color=PALETTE["primary"], label="Model", zorder=3)
    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Observed Event Rate")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="upper left")
    if own:
        return ax


def plot_response(y_true, scores, n_buckets: int = 10, ax=None):
    """Response + cumulative response (modelplotpy) from the decile profile."""
    ax, own = _ax(ax)
    _style_axis(ax)
    rows = Metrics().decile_table(y_true, scores, n_buckets)
    d = [r["decile"] for r in rows]
    ax.plot(d, [r["event_rate"] for r in rows], marker="o", markersize=5, color=PALETTE["secondary"], lw=1.8, label="Decile Response", zorder=3)
    ax.plot(d, [r["cum_event_rate"] for r in rows], marker="s", markersize=5, color=PALETTE["primary"], lw=1.8, label="Cumulative Response", zorder=3)
    ax.set_xlabel("Decile (1 = top scores)")
    ax.set_ylabel("Response Rate")
    ax.set_title("Response by Decile")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="upper right")
    if own:
        return ax


# ---------------------------------------------------------- model comparison

def plot_double_lift(y_true, scores_a, scores_b, n_buckets: int = 10,
                      model_type: str = "additive", label_a: str = "Model A",
                      label_b: str = "Model B", ax=None):
    """Double lift chart (CAS monograph): actual vs two models along A's ranking."""
    ax, own = _ax(ax)
    _style_axis(ax)
    rows = Metrics().double_lift(y_true, scores_a, scores_b, n_buckets)
    denom = 1.0
    if model_type == "ratio":
        denom = sum(r["actual"] * r["n"] for r in rows) / max(sum(r["n"] for r in rows), 1)
        if denom == 0:
            denom = 1.0
    d = [r["bucket"] for r in rows]
    ax.plot(d, [r["actual"] / denom for r in rows], marker="o", markersize=5, color=PALETTE["dark"], lw=2.0, ls="--", label="Actual", zorder=4)
    ax.plot(d, [r["pred_a"] / denom for r in rows], marker="^", markersize=5, color=PALETTE["primary"], lw=1.8, label=label_a, zorder=3)
    ax.plot(d, [r["pred_b"] / denom for r in rows], marker="s", markersize=5, color=PALETTE["accent"], lw=1.8, label=label_b, zorder=3)
    ax.set_xlabel(f"Bucket sorted by {label_a} (1 = highest)")
    ax.set_ylabel("Ratio to Mean Actual" if model_type == "ratio" else "Mean Value")
    ax.set_title(f"Double Lift Chart ({model_type.capitalize()})")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="upper right")
    if own:
        return ax


# -------------------------------------------------------------- stability

def plot_psi(reference, current, n_bins: int = 10, ax=None):
    """Stability chart: side-by-side share bars + PSI contribution curve."""
    ax, own = _ax(ax)
    _style_axis(ax, grid=False)
    r = Metrics().stability_index(reference, current, n_bins)
    bins = r["bins"]
    x = list(range(len(bins)))
    w = 0.35

    ax.bar([i - w / 2 for i in x], [b["reference_share"] for b in bins], width=w, color=PALETTE["muted"], edgecolor="#4db6ac", linewidth=0.8, label="Reference Share", zorder=2)
    ax.bar([i + w / 2 for i in x], [b["current_share"] for b in bins], width=w, color=PALETTE["accent"], alpha=0.85, edgecolor="#d84315", linewidth=0.8, label="Current Share", zorder=2)
    ax.set_xlabel("Score / Feature Bin")
    ax.set_ylabel("Population Share", color=PALETTE["dark"])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#cfd8dc")
    ax2.plot(x, [b["contribution"] for b in bins], color=PALETTE["dark"], lw=1.8, marker="o", markersize=4, label="PSI Contribution", zorder=4)
    ax2.set_ylabel("Bin PSI Contribution", color=PALETTE["dark"])

    ax.set_title(f"Population Stability Index (PSI = {r['psi']:.4f})")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=8.5, loc="upper left")
    if own:
        return ax


# ------------------------------------------------ explanation & diagnostics

def plot_importance(importances, ax=None):
    """Horizontal bars of model.feature_importance() (name, share) pairs."""
    ax, own = _ax(ax)
    _style_axis(ax)
    names = [n for n, _ in importances][::-1]
    shares = [s for _, s in importances][::-1]
    y_pos = range(len(names))
    bars = ax.barh(y_pos, shares, height=0.6, color=PALETTE["primary"], edgecolor="#00695c", linewidth=0.8, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Share of Total Split Gain")
    ax.set_title(f"Feature Importance ({len(importances)} features)")

    # Label shares on the bars
    max_s = max(shares) if shares else 1.0
    for bar, s in zip(bars, shares):
        ax.text(bar.get_width() + max_s * 0.015, bar.get_y() + bar.get_height() / 2, f"{s:.1%}", va="center", ha="left", fontsize=8.5, color="#37474f")
    ax.set_xlim(0, max_s * 1.18)
    if own:
        return ax


def plot_breakdown(bd, ax=None):
    """Waterfall of model.break_down(row): baseline -> steps -> prediction."""
    ax, own = _ax(ax)
    _style_axis(ax)
    steps = list(bd["steps"])
    labels = ["Baseline"] + [n for n, _ in steps] + ["Residual", "Prediction"]

    # Baseline
    ax.bar(0, bd["baseline"], color=PALETTE["gray"], edgecolor="#546e7a", linewidth=0.8, zorder=3)
    running = bd["baseline"]

    # Steps
    for i, (_, d) in enumerate(steps, start=1):
        c = PALETTE["primary"] if d >= 0 else PALETTE["accent"]
        ec = "#00695c" if d >= 0 else "#bf360c"
        ax.bar(i, d, bottom=running, color=c, edgecolor=ec, linewidth=0.8, zorder=3)
        # Connector line
        ax.plot([i - 0.5, i + 0.5], [running + d, running + d], color="#90a4ae", lw=0.8, ls=":", zorder=2)
        running += d

    # Residual
    ax.bar(len(labels) - 2, bd["residual"], bottom=running, color=PALETTE["warning"], edgecolor="#f57f17", linewidth=0.8, zorder=3)
    running += bd["residual"]

    # Final prediction
    ax.bar(len(labels) - 1, bd["prediction"], color=PALETTE["dark"], edgecolor="#102027", linewidth=0.8, zorder=3)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Target Contribution")
    ax.set_title(f"Breakdown Waterfall ({bd['baseline']:.1f} → {bd['prediction']:.1f})")
    if own:
        return ax


def plot_residuals(y_true, y_pred, ax=None):
    """Regression diagnostics: Predicted vs Actual with equality line."""
    ax, own = _ax(ax)
    _style_axis(ax)
    ax.scatter(y_pred, y_true, color=PALETTE["primary"], alpha=0.55, edgecolor="none", s=22, zorder=3)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    pad = (mx - mn) * 0.05
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], color=PALETTE["accent"], lw=1.2, ls="--", label="Ideal (y = ŷ)", zorder=2)
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual Value")
    ax.set_title("Predicted vs. Actual")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=9, loc="upper left")
    if own:
        return ax


def plot_confusion_matrix(confusion, class_names=None, ax=None):
    """Confusion matrix heatmap with cell counts and percentages."""
    ax, own = _ax(ax)
    _style_axis(ax, grid=False)
    n_classes = len(confusion)
    names = class_names or [f"Class {i}" for i in range(n_classes)]
    total = sum(sum(row) for row in confusion)

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    mat = np.array(confusion, dtype=float)
    teal_cmap = LinearSegmentedColormap.from_list("teal_map", ["#f0fdfa", "#b2dfdb", "#80cbc4", "#00897b", "#004d40"])
    im = ax.imshow(mat, cmap=teal_cmap, interpolation="nearest")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Cell labels
    max_val = mat.max()
    for i in range(n_classes):
        for j in range(n_classes):
            val = mat[i, j]
            pct = (val / total * 100) if total > 0 else 0
            text_color = "#ffffff" if val > max_val * 0.55 else "#263238"
            ax.text(j, i, f"{int(val):,}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontsize=9, weight="semibold")
    if own:
        return ax


def plot_variance_shares(shares, ax=None):
    """Horizontal bars of realized variance shares (generate() truth)."""
    ax, own = _ax(ax)
    _style_axis(ax)
    keys = list(shares.keys())[::-1]
    vals = [shares[k] for k in keys]
    y_pos = range(len(keys))
    bars = ax.barh(y_pos, vals, height=0.55, color=PALETTE["primary"], edgecolor="#00695c", linewidth=0.8, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([k.capitalize() for k in keys])
    ax.set_xlabel("Realized Variance Share")
    ax.set_title("Synthetic Generator Ground Truth")

    max_v = max(vals) if vals else 1.0
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max_v * 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.1%}", va="center", ha="left", fontsize=8.5, color="#37474f")
    ax.set_xlim(0, max_v * 1.22)
    if own:
        return ax


def plot_decomposition(values, trend, detrended, deseason, residual,
                       dates=None, axes=None):
    """Four-panel additive decomposition (statsmodels convention)."""
    created = None
    if axes is None:
        created, axes = plt.subplots(2, 2, figsize=(11, 5.8), sharex=True)
    for row in axes:
        for ax in row:
            _style_axis(ax)

    axes[0][0].plot(values, color=PALETTE["primary"], lw=1.2, label="Original series")
    axes[0][0].plot(trend, color=PALETTE["accent"], lw=1.8, label="Trend (Centered MA)")
    axes[0][0].set_title("Observed Series + Trend")
    axes[0][0].legend(frameon=True, framealpha=0.9, edgecolor="#cfd8dc", fontsize=8.5)

    axes[0][1].plot(detrended, color=PALETTE["secondary"], lw=1.2)
    axes[0][1].set_title("Detrended (Observed − Trend)")

    axes[1][0].plot(deseason, color=PALETTE["muted"], lw=1.2)
    axes[1][0].set_title("Deseasonalized (Observed − Season)")

    axes[1][1].plot(residual, color=PALETTE["dark"], lw=1.2)
    axes[1][1].set_title("Residual")

    if dates is not None:
        step = max(1, len(dates) // 8)
        ticks = list(range(0, len(dates), step))
        labels = [dates[i] for i in ticks]
        for ax in axes[1]:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    if created is not None:
        return axes
