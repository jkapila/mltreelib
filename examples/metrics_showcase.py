"""Every Metrics surface and every plot, in one runnable showcase.

No surface is left without a demonstration: reports, curves, probability and
margin scores, classification and multilabel scorecards, agreement, stability,
double lift, distances - and all ten charts from mltreelib.plots.

    uv sync --extra plotting        # matplotlib + pygraphviz + seaborn
    uv run python examples/metrics_showcase.py
"""

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mltreelib import Metrics, ModelTree, OneHotEncoder, plots

m = Metrics()

# ---------------------------------------------------------------- data
# Real binary problem: bank churn. The tree's predictions double as scores.
df = pd.read_parquet("datasets/Bank_Customer_Churn.parquet")
y = (df["churn"] == "Yes").astype(float).tolist()
x = df.drop(columns=["churn"]).copy()
for col in x.select_dtypes(include="category").columns:
    enc = OneHotEncoder()
    enc.fit([None if v is None else str(v) for v in x[col].astype(object)])
    dummies = enc.transform([None if v is None else str(v) for v in x[col].astype(object)])
    x = x.drop(columns=[col])
    for level, values in dummies.items():
        x[f"{col}_{level}"] = values
X = [[None if pd.isna(v) else float(v) for v in row] for row in x.values.tolist()]

n_tr = int(0.8 * len(X))
tree = ModelTree(max_depth=4, min_rows=40)
tree.fit(X[:n_tr], y[:n_tr])
scores = [min(1.0, max(0.0, p)) for p in tree.predict(X[n_tr:])]
truth = y[n_tr:]
hard = [1.0 if p >= 0.5 else 0.0 for p in scores]
base = [sum(truth) / len(truth)] * len(truth)  # mean predictor
print(f"churn: {len(truth)} test rows, event rate {sum(truth)/len(truth):.3f}")

# ------------------------------------------------------------- reports
print("\n== reports ==")
reg = m.regression(truth, scores)
print(f"regression: rmse {reg['rmse']:.3f}  r2 {reg['r2']:.3f}  "
      f"deciles {len(reg['deciles'])}")
clf = m.classification([int(v) for v in truth], [int(v) for v in hard])
print(f"classification: accuracy {clf['accuracy']:.3f}  "
      f"macro f1 {clf['macro_f1']:.3f}  confusion {clf['confusion']}")
med = sorted(scores)[len(scores) // 2]
grp = m.regression_by_group(
    ["high" if s >= med else "low" for s in scores], truth, scores
)
print(f"regression_by_group: {sorted(grp)}")
cgrp = m.classification_by_group(
    ["high" if s >= med else "low" for s in scores],
    [int(v) for v in truth], [int(v) for v in hard],
)
print("classification_by_group:", {k: round(v["accuracy"], 3) for k, v in cgrp.items()})

# -------------------------------------------------------------- curves
print("\n== curves ==")
roc = m.roc_curve(truth, scores)
pr = m.pr_curve(truth, scores)
ks = m.ks_statistic(truth, scores)
dec = m.decile_table(truth, scores)
cal = m.calibration(truth, scores)
print(f"roc: auc {roc['auc']:.3f} over {len(roc['points'])} points")
print(f"pr:  ap {pr['average_precision']:.3f}")
print(f"ks:  {ks['ks']:.3f} at score {ks['score_at_ks']:.3f} "
      f"(tpr {ks['tpr_at_ks']:.2f} / fpr {ks['fpr_at_ks']:.2f})")
print(f"gain/decile 1: capture {dec[0]['cum_capture']:.3f}  lift {dec[0]['lift']:.2f}")
print(f"calibration: {len(cal['bins'])} non-empty bins")

# ------------------------------------------- probability & margin scores
print("\n== probability & margin scores ==")
print(f"log_loss {m.log_loss(truth, scores):.3f}  "
      f"brier {m.brier_score(truth, scores):.3f}  "
      f"hinge {m.hinge_loss(truth, [2*p - 1 for p in scores]):.3f}")

# --------------------------------------------- classification scorecard
print("\n== classification scorecard ==")
card = m.classification_scores([int(v) for v in truth], [int(v) for v in hard])
print("  " + "  ".join(f"{k}={v:.3f}" for k, v in card.items()
                       if isinstance(v, float)))

# --------------------------------------------------- multilabel scorecard
print("\n== multilabel ==")
random.seed(3)
ml_true = [[1.0 if random.random() < 0.4 else 0.0 for _ in range(4)] for _ in range(50)]
ml_scores = [[min(1.0, max(0.0, v + random.uniform(-0.3, 0.3))) for v in row]
             for row in ml_true]
ml = m.multilabel_scores(ml_true, ml_scores)
print(f"  subset_accuracy {ml['subset_accuracy']:.3f}  "
      f"hamming {ml['hamming_loss']:.3f}  lrap {ml['label_ranking_ap']:.3f}  "
      f"coverage {ml['coverage_error']:.2f}")

# ------------------------------------------------------- agreement (kappa)
print("\n== agreement ==")
for w in ("none", "linear", "quadratic"):
    print(f"  weighted_kappa({w}): {m.weighted_kappa(truth, hard, w):.3f}")

# ---------------------------------------------------------- stability PSI/CSI
print("\n== stability ==")
h = len(scores) // 2
psi = m.stability_index(scores[:h], scores[h:])
print(f"  PSI (score halves): {psi['psi']:.4f}")
age_i = list(x.columns).index("age")
csi = m.stability_index([r[age_i] for r in X[:n_tr]], [r[age_i] for r in X[n_tr:]])
print(f"  CSI (age train vs test): {csi['psi']:.4f}")

# ------------------------------------------------- double lift (CAS chart)
print("\n== double lift (tree vs mean baseline) ==")
dl = m.double_lift(truth, scores, base, 5)
for r in dl:
    print(f"  bucket {r['bucket']}: n={r['n']} actual={r['actual']:.3f} "
          f"tree={r['pred_a']:.3f} base={r['pred_b']:.3f}")

# ------------------------------------------------------------- distances
print("\n== pairwise distances ==")
a, b = [[0.0, 0.0], [1.0, 1.0]], [[3.0, 4.0], [0.0, 0.0]]
for metric in ("euclidean", "manhattan", "chebyshev", "minkowski", "cosine",
               "correlation", "hamming", "jaccard"):
    d = m.pairwise_distances(a, b, metric)
    print(f"  {metric:<12} {[[round(v, 3) for v in row] for row in d]}")

# ----------------------------------------------------------------- plots
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
plots.plot_roc(truth, scores, ax=axes[0][0])
plots.plot_pr(truth, scores, ax=axes[0][1])
plots.plot_cumulative_gain(truth, scores, ax=axes[0][2])
plots.plot_lift(truth, scores, ax=axes[0][3])
plots.plot_ks(truth, scores, ax=axes[0][4])
plots.plot_decile(truth, scores, ax=axes[1][0])
plots.plot_calibration(truth, scores, ax=axes[1][1])
plots.plot_response(truth, scores, ax=axes[1][2])
plots.plot_double_lift(truth, scores, base, model_type="ratio", ax=axes[1][3])
plots.plot_psi(scores[:h], scores[h:], ax=axes[1][4])
fig.suptitle("mltreelib.plots - the full chart surface")
fig.tight_layout()
fig.savefig("docs/assets/metrics_panels.png", dpi=110)
print("\nwrote docs/assets/metrics_panels.png (10 charts)")
