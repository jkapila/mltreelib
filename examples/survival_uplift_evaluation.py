"""Held-out synthetic survival and randomized uplift evaluations.

Run: uv run python examples/survival_uplift_evaluation.py
Requires matplotlib. Known simulation truth is used only for diagnostics.
"""
from math import exp, sqrt
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mltreelib import ModelTree

out = Path("docs/assets/guides")
out.mkdir(parents=True, exist_ok=True)


def ipcw_brier(times, events, prediction, horizon, censor_survival):
    """Fixed-horizon Brier score with independent censoring weights."""
    total = 0.0
    for t, event, s in zip(times, events, prediction):
        if t <= horizon and event:
            total += s * s / censor_survival(t)
        elif t > horizon:
            total += (1 - s) ** 2 / censor_survival(horizon)
    return total / len(times)


def survival():
    rng = Random(450)
    x = [[rng.random()] for _ in range(5000)]
    rate = [.03 + .07 * row[0] for row in x]
    event_time = [rng.expovariate(v) for v in rate]
    censor_time = [rng.uniform(5, 40) for _ in x]
    times = [min(t, c) for t, c in zip(event_time, censor_time)]
    event = [float(t <= c) for t, c in zip(event_time, censor_time)]
    tree = ModelTree(leaf_family="survival", max_depth=2, min_rows=100, prune="none")
    tree.fit(x[:3000], (times[:3000], event[:3000]), col_names=["risk_signal"])
    baseline = ModelTree(leaf_family="survival", max_depth=0, min_rows=100)
    baseline.fit(x[:3000], (times[:3000], event[:3000]), col_names=["risk_signal"])
    grid = [10., 20., 30.]
    pred = tree.predict(x[3000:], kind="survival", times=grid)
    base = baseline.predict(x[3000:], kind="survival", times=grid)
    # Censoring is Uniform(5, 40) by construction. Real data must estimate G on training data.
    def g(t):
        return 1.0 if t <= 5 else (40.0 - t) / 35.0
    assert abs(ipcw_brier([1., 3.], [1., 0.], [.2, .8], 2., lambda _: 1.) - .04) < 1e-12
    tree_scores, base_scores = [], []
    print("survival train/test:", 3000, 2000)
    print("survival test events/censored:", int(sum(event[3000:])), int(len(event[3000:])-sum(event[3000:])))
    for j, t in enumerate(grid):
        p = [row[j] for row in pred]
        bp = [row[j] for row in base]
        a = ipcw_brier(times[3000:], event[3000:], p, t, g)
        b = ipcw_brier(times[3000:], event[3000:], bp, t, g)
        oracle_mse = sum((v - exp(-rate[i + 3000] * t)) ** 2 for i, v in enumerate(p)) / len(p)
        tree_scores.append(a); base_scores.append(b)
        print(f"survival t={t:.0f}: IPCW Brier tree={a:.4f} baseline={b:.4f} synthetic truth MSE={oracle_mse:.4f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, tree_scores, marker="o", label="tree")
    ax.plot(grid, base_scores, marker="o", label="global survival baseline")
    ax.set(xlabel="Time horizon", ylabel="IPCW Brier score, lower is better", title="Survival: independent test set")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "survival-evaluation.svg"); plt.close(fig)


def policy_value(y, treatment, select, cost):
    """Randomized difference in means among selected, scaled to all records."""
    yt = [v for v,w,s in zip(y,treatment,select) if s and w == 1]
    yc = [v for v,w,s in zip(y,treatment,select) if s and w == 0]
    n = sum(select)
    if not n:
        return 0.0, 0, 0, 0
    if not yt or not yc:
        raise ValueError("Selected group needs treated and control rows")
    value = n / len(y) * (sum(yt)/len(yt) - sum(yc)/len(yc) - cost)
    return value, n, len(yt), len(yc)


def uplift():
    rng = Random(460)
    x = [[rng.uniform(-1, 1)] for _ in range(6000)]
    treatment = [float(rng.random() < .5) for _ in x]
    tau = [2.0 if row[0] >= 0 else -1.0 for row in x]
    y = [5 + effect * w + rng.gauss(0, 1) for effect,w in zip(tau,treatment)]
    model = ModelTree(gain="causal", leaf="mean", max_depth=2, min_rows=100, prune="none")
    model.fit(x[:3000], (y[:3000], treatment[:3000]), col_names=["segment_signal"])
    val = model.predict(x[3000:4000], kind="uplift")
    candidates = [-100., 0., .5, 1., 100.]
    cost = .5
    threshold = max(candidates, key=lambda t: policy_value(y[3000:4000], treatment[3000:4000], [v > t for v in val], cost)[0])
    pred = model.predict(x[4000:], kind="uplift")
    yt, wt, truth = y[4000:], treatment[4000:], tau[4000:]
    selections = {"selected policy": [v > threshold for v in pred], "treat all": [True]*len(pred), "treat none": [False]*len(pred)}
    print("uplift train/validation/test:", 3000, 1000, 2000)
    print("uplift validation threshold:", threshold, "treatment cost:", cost)
    print("uplift synthetic PEHE:", round(sqrt(sum((p-t)**2 for p,t in zip(pred,truth))/len(pred)), 4))
    values=[]
    for name, selected in selections.items():
        value,n,nt,nc = policy_value(yt,wt,selected,cost)
        true_value = sum((t-cost) for t,s in zip(truth,selected) if s)/len(truth)
        interval = "not applicable"
        if nt > 1 and nc > 1:
            treated_y = [v for v, w, selected_row in zip(yt, wt, selected) if selected_row and w == 1]
            control_y = [v for v, w, selected_row in zip(yt, wt, selected) if selected_row and w == 0]
            def sample_variance(values):
                mean = sum(values) / len(values)
                return sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            se = n / len(yt) * sqrt(sample_variance(treated_y) / nt + sample_variance(control_y) / nc)
            interval = f"[{value - 1.96*se:.4f}, {value + 1.96*se:.4f}]"
        print(f"{name}: net value/person={value:.4f}, approximate 95% interval={interval}, selected={n}, treated={nt}, control={nc}, synthetic truth={true_value:.4f}")
        values.append(value)
    assert policy_value([1.,2.],[0.,1.],[False,False],.5)[0] == 0.
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(list(selections), values, color=["#00897b", "#0288d1", "#78909c"])
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(ylabel="Estimated incremental net outcome per person", title="Uplift: independent randomized test set")
    fig.tight_layout(); fig.savefig(out / "uplift-evaluation.svg"); plt.close(fig)


if __name__ == "__main__":
    survival()
    uplift()
