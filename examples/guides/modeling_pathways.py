"""Reproducible, synthetic examples for the practical modeling guides.

Run with: uv run python examples/guides/modeling_pathways.py
Requires matplotlib to render the six figures.
"""

from pathlib import Path
from random import Random
from math import exp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mltreelib import AnomalyModel, ClusterModel, MetricsReport, ModelTree

OUT = Path("docs/assets/guides")
OUT.mkdir(parents=True, exist_ok=True)


def regression():
    rng = Random(41)
    x = [[rng.uniform(0, 10)] for _ in range(3000)]
    y = [2 + 3 * row[0] + rng.gauss(0, 2) for row in x]
    model = ModelTree(leaf="linear", max_depth=2, min_rows=20)
    model.fit(x[:2000], y[:2000], col_names=["volume"])
    pred = model.predict(x[2000:])
    report = MetricsReport("regression", y_true=y[2000:], y_pred=pred)
    print("regression held-out RMSE:", round(report["rmse"], 3))
    print("regression splits:", model.splits())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y[2000:], pred, s=18, alpha=.65)
    lo, hi = min(y[2000:]), max(y[2000:])
    ax.plot([lo, hi], [lo, hi], "--", color="black", label="perfect prediction")
    ax.set(xlabel="Actual outcome", ylabel="Predicted outcome", title="Regression: held-out predictions")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "regression.svg"); plt.close(fig)


def classification():
    rng = Random(42)
    x = [[rng.uniform(-3, 3)] for _ in range(3000)]
    y = [float(rng.random() < 1 / (1 + exp(-r[0]))) for r in x]
    model = ModelTree(leaf="mean", leaf_family="binomial", gain="deviance", max_depth=2, min_rows=25)
    model.fit(x[:2000], y[:2000], col_names=["risk_signal"])
    scores = model.predict_proba(x[2000:])
    labels = [int(p >= .5) for p in scores]
    report = MetricsReport("classification", y_true=[int(v) for v in y[2000:]], y_pred=labels)
    print("classification held-out confusion:", report.tables["confusion"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r[0] for r in x[2000:]], scores, c=y[2000:], cmap="coolwarm", s=18, alpha=.7)
    ax.axhline(.5, color="black", linestyle="--", label="example threshold")
    ax.set(xlabel="Risk signal", ylabel="Predicted probability of class 1", title="Classification: held-out scores")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "classification.svg"); plt.close(fig)


def clustering():
    rng = Random(43)
    x = [[rng.gauss(center, .4), rng.gauss(center, .4)] for center in (-2, 2) for _ in range(1000)]
    model = ClusterModel(kind="pct", max_depth=2, min_rows=10)
    model.fit(x, col_names=["measure_a", "measure_b"])
    labels = model.predict(x)
    print("clustering labels:", sorted(set(labels)))
    print("clustering rules:", model.rules())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r[0] for r in x], [r[1] for r in x], c=labels, cmap="viridis", s=18)
    ax.set(xlabel="Measure A", ylabel="Measure B", title="Clustering: fitted segments")
    fig.tight_layout(); fig.savefig(OUT / "clustering.svg"); plt.close(fig)


def anomaly():
    rng = Random(44)
    train = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(2000)]
    review = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(1000)] + [[5., 5.], [-5., 4.]]
    model = AnomalyModel(kind="eif", max_depth=10, extension_level=1, seed=44)
    model.fit(train, col_names=["measure_a", "measure_b"])
    scores = model.score_samples(review)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    print("anomaly top review indices:", top)
    fig, ax = plt.subplots(figsize=(6, 4))
    points = ax.scatter([r[0] for r in review], [r[1] for r in review], c=scores, cmap="magma", s=28)
    fig.colorbar(points, ax=ax, label="Anomaly score")
    ax.set(xlabel="Measure A", ylabel="Measure B", title="Anomaly: review scores")
    fig.tight_layout(); fig.savefig(OUT / "anomaly.svg"); plt.close(fig)


def survival():
    rng = Random(45)
    x = [[rng.uniform(0, 1)] for _ in range(3000)]
    event_time = [rng.expovariate(.03 + .07 * r[0]) for r in x]
    censor_time = [rng.uniform(5, 40) for _ in x]
    times = [min(t, c) for t, c in zip(event_time, censor_time)]
    events = [float(t <= c) for t, c in zip(event_time, censor_time)]
    model = ModelTree(leaf_family="survival", max_depth=1, min_rows=20, prune="none")
    model.fit(x[:2000], (times[:2000], events[:2000]), col_names=["risk_signal"])
    grid = [float(i) for i in range(0, 41)]
    curves = model.predict([[.1], [.9]], kind="survival", times=grid)
    print("survival at day 20:", [round(curve[20], 3) for curve in curves])
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in zip(("low risk signal", "high risk signal"), curves):
        ax.step(grid, curve, where="post", label=name)
    ax.set(xlabel="Time", ylabel="Estimated survival probability", ylim=(0, 1), title="Survival: fitted leaf curves")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "survival.svg"); plt.close(fig)


def uplift():
    rng = Random(46)
    x = [[rng.uniform(-1, 1)] for _ in range(3000)]
    treatment = [float(rng.random() < .5) for _ in x]
    outcome = [5 + (2 if r[0] >= 0 else -1) * w + rng.gauss(0, 1) for r, w in zip(x, treatment)]
    model = ModelTree(gain="causal", leaf="mean", max_depth=1, min_rows=40)
    model.fit(x[:2000], (outcome[:2000], treatment[:2000]), col_names=["segment_signal"])
    scores = model.predict(x[2000:], kind="uplift")
    print("uplift held-out score range:", round(min(scores), 3), round(max(scores), 3))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r[0] for r in x[2000:]], scores, s=14, alpha=.45)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="Segment signal", ylabel="Estimated treatment effect", title="Uplift: held-out scores")
    fig.tight_layout(); fig.savefig(OUT / "uplift.svg"); plt.close(fig)


if __name__ == "__main__":
    for example in (regression, classification, clustering, anomaly, survival, uplift):
        example()
