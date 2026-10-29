"""Plotting for the metrics module — matplotlib is an OPTIONAL extra.

Every function renders one chart from data that `mltreelib.Metrics` returns
(or computes it from `y_true`/scores itself). The library core stays
plotting-free; this module is the thin, standard rendering layer (matplotlib
is the wheel, we do not reinvent charting).

    uv run --with matplotlib python -c "from mltreelib import plots"

Conventions: each function takes `ax=None` (creates a figure when absent) and
returns the Axes, so charts compose into subplots.
"""

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "mltreelib.plots needs matplotlib (a dev extra): uv run --with matplotlib ..."
    ) from e

from mltreelib import Metrics


def _ax(ax):
    if ax is not None:
        return ax, None
    _, ax = plt.subplots()
    return ax, True


# ---------------------------------------------------------------- ranking

def plot_roc(y_true, scores, ax=None):
    """ROC curve + AUC annotation (Metrics().roc_curve)."""
    ax, own = _ax(ax)
    r = Metrics().roc_curve(y_true, scores)
    pts = r["points"]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#2a7", lw=1.8)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC (AUC = {r['auc']:.3f})")
    if own:
        return ax


def plot_pr(y_true, scores, ax=None):
    """Precision-recall curve + average precision annotation."""
    ax, own = _ax(ax)
    r = Metrics().pr_curve(y_true, scores)
    pts = r["points"]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#c63", lw=1.8)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"precision-recall (AP = {r['average_precision']:.3f})")
    if own:
        return ax


def plot_cumulative_gain(y_true, scores, ax=None):
    """Cumulative gains (modelplotpy/kds): capture vs population fraction."""
    ax, own = _ax(ax)
    pts = Metrics().gain_curve(y_true, scores)["points"]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#2a7", lw=1.8,
            label="model")
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--", label="random")
    ax.set_xlabel("population fraction")
    ax.set_ylabel("capture rate")
    ax.set_title("cumulative gains")
    ax.legend()
    if own:
        return ax


def plot_lift(y_true, scores, ax=None):
    """Cumulative lift (kds/modelplotpy): capture / population fraction."""
    ax, own = _ax(ax)
    pts = Metrics().lift_curve(y_true, scores)["points"]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#06c", lw=1.8)
    ax.axhline(1.0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("population fraction")
    ax.set_ylabel("lift")
    ax.set_title("cumulative lift")
    if own:
        return ax


def plot_ks(y_true, scores, ax=None):
    """KS statistic chart (kds/PerfScoreCard): TPR and FPR curves with the
    maximum gap marked."""
    ax, own = _ax(ax)
    r = Metrics().ks_statistic(y_true, scores)
    pts = r["points"]
    ax.plot([p[2] for p in pts], [p[1] for p in pts], color="#2a7", lw=1.5,
            label="TPR")
    ax.plot([p[2] for p in pts], [p[0] for p in pts], color="#c63", lw=1.5,
            label="FPR")
    ax.axvline(r["score_at_ks"], color="#06c", lw=1.0, ls="--")
    ax.annotate(f"KS = {r['ks']:.3f}\nscore@KS = {r['score_at_ks']:.3f}",
                xy=(r["score_at_ks"], r["tpr_at_ks"]), xytext=(0.05, 0.9),
                textcoords="axes fraction")
    ax.set_xlabel("score (descending sweep)")
    ax.set_ylabel("rate")
    ax.set_title("KS statistic")
    ax.legend()
    if own:
        return ax


def plot_decile(y_true, scores, n_buckets=10, ax=None):
    """Decile profile (kds/PerfScoreCard): event rate bars + cumulative
    capture line."""
    ax, own = _ax(ax)
    rows = Metrics().decile_table(y_true, scores, n_buckets)
    d = [r["decile"] for r in rows]
    ax.bar(d, [r["event_rate"] for r in rows], color="#8ab", label="event rate")
    ax.set_xlabel("decile (1 = top scores)")
    ax.set_ylabel("event rate")
    ax2 = ax.twinx()
    ax2.plot(d, [r["cum_capture"] for r in rows], color="#c63", lw=1.8,
             label="cum capture")
    ax2.set_ylabel("cumulative capture")
    ax.set_title("decile profile")
    if own:
        return ax


def plot_calibration(y_true, scores, n_bins=10, ax=None):
    """Reliability diagram (calibration curve) vs the diagonal."""
    ax, own = _ax(ax)
    bins = Metrics().calibration(y_true, scores, n_bins)["bins"]
    ax.plot([b["mean_score"] for b in bins], [b["observed"] for b in bins],
            marker="o", color="#2a7", label="model")
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--", label="perfect")
    ax.set_xlabel("mean predicted score")
    ax.set_ylabel("observed rate")
    ax.set_title("calibration")
    ax.legend()
    if own:
        return ax


def plot_response(y_true, scores, n_buckets=10, ax=None):
    """Response + cumulative response (modelplotpy) from the decile profile."""
    ax, own = _ax(ax)
    rows = Metrics().decile_table(y_true, scores, n_buckets)
    d = [r["decile"] for r in rows]
    ax.plot(d, [r["event_rate"] for r in rows], marker="o", color="#06c",
            label="response")
    ax.plot(d, [r["cum_event_rate"] for r in rows], marker="s", color="#2a7",
            label="cumulative response")
    ax.set_xlabel("decile (1 = top scores)")
    ax.set_ylabel("rate")
    ax.set_title("response")
    ax.legend()
    if own:
        return ax


# ---------------------------------------------------------- model comparison

def plot_double_lift(y_true, scores_a, scores_b, n_buckets=10,
                     model_type="additive", label_a="model A",
                     label_b="model B", ax=None):
    """Double lift chart (CAS monograph): actual vs two models along A's
    ranking. `model_type` "additive" plots raw means; "ratio" indexes by the
    overall actual mean (loss-ratio style)."""
    ax, own = _ax(ax)
    rows = Metrics().double_lift(y_true, scores_a, scores_b, n_buckets)
    denom = 1.0
    if model_type == "ratio":
        denom = sum(r["actual"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
        if denom == 0:
            denom = 1.0
    d = [r["bucket"] for r in rows]
    ax.plot(d, [r["actual"] / denom for r in rows], marker="o", color="#111",
            lw=2.0, label="actual")
    ax.plot(d, [r["pred_a"] / denom for r in rows], marker="^", color="#06c",
            label=label_a)
    ax.plot(d, [r["pred_b"] / denom for r in rows], marker="s", color="#c63",
            label=label_b)
    ax.set_xlabel(f"bucket by {label_a} (1 = highest)")
    ax.set_ylabel("ratio to overall actual" if model_type == "ratio" else "mean")
    ax.set_title(f"double lift ({model_type})")
    ax.legend()
    if own:
        return ax


# -------------------------------------------------------------- stability

def plot_psi(reference, current, n_bins=10, ax=None):
    """Stability chart: per-bin share shift + contribution bars (PSI/CSI)."""
    ax, own = _ax(ax)
    r = Metrics().stability_index(reference, current, n_bins)
    bins = r["bins"]
    x = list(range(len(bins)))
    ax.bar(x, [b["reference_share"] for b in bins], color="#8ab",
           label="reference")
    ax.bar(x, [b["current_share"] for b in bins], color="#c63", alpha=0.6,
           label="current")
    ax2 = ax.twinx()
    ax2.plot(x, [b["contribution"] for b in bins], color="#111", lw=1.2,
             marker="o", label="PSI contribution")
    ax.set_xlabel("bin")
    ax.set_ylabel("share")
    ax2.set_ylabel("contribution")
    ax.set_title(f"stability (PSI = {r['psi']:.4f})")
    ax.legend()
    if own:
        return ax
