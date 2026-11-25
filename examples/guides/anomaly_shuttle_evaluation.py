"""UCI Statlog Shuttle anomaly proxy, using the provided train/test split.

Place shuttle.trn and shuttle.tst in datasets/external/shuttle/.
Run: uv run python examples/guides/anomaly_shuttle_evaluation.py
Class 1 is the reference class; other classes are an anomaly proxy.
"""
from itertools import accumulate
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mltreelib import AnomalyTree, Metrics, plots

base = Path("datasets/external/shuttle")
def load(path):
    with path.open() as f:
        rows = [[float(v) for v in line.split()] for line in f if line.strip()]
    return [r[:-1] for r in rows], [int(r[-1]) for r in rows]
train_x, train_y = load(base / "shuttle.trn")
test_x, test_y = load(base / "shuttle.tst")
normal = [row for row, label in zip(train_x, train_y) if label == 1]
rng = Random(42)
rng.shuffle(normal)
reference = normal[:5000]
truth = [int(label != 1) for label in test_y]
model = AnomalyTree(kind="eif", max_depth=10, extension_level=1, seed=42)
model.fit(reference, col_names=[f"sensor_{i+1}" for i in range(9)])
scores = model.score_samples(test_x)
budget = round(.05 * len(test_x))
rank = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
caught = sum(truth[i] for i in rank[:budget])
print("shuttle original train/test:", len(train_x), len(test_x))
print("reference normal train subset:", len(reference))
print("test anomaly proxy count:", sum(truth), "prevalence:", round(sum(truth)/len(truth), 4))
print("review budget:", budget, "caught:", caught, "precision:", round(caught/budget, 4), "recall:", round(caught/sum(truth), 4))
hits = list(accumulate(truth[i] for i in rank))
budgets = [round(fraction * len(test_x)) for fraction in (.01, .05, .10)]
for k in budgets:
    print("queue size, precision, recall:", k, round(hits[k - 1] / k, 4),
          round(hits[k - 1] / sum(truth), 4))
print("test average precision:", round(Metrics().pr_curve(truth, scores)["average_precision"], 4))

out = Path("docs/assets/guides"); out.mkdir(parents=True, exist_ok=True)
fig, (ax, queue_ax) = plt.subplots(1, 2, figsize=(10, 4))
plots.plot_pr(truth, scores, ax=ax)
prevalence = sum(truth) / len(truth)
ax.axhline(prevalence, linestyle="--", color="#c86729", label="Test prevalence")
ax.set_title("Held-out anomaly proxy")
ax.legend()
queue_ax.plot([k / len(truth) * 100 for k in budgets],
              [hits[k - 1] / k for k in budgets], marker="o", label="EIF ranking")
queue_ax.axhline(prevalence, linestyle="--", color="#c86729", label="Random review")
queue_ax.set(xlabel="Share of test rows reviewed (%)", ylabel="Precision at budget",
             ylim=(0, 1.05), title="Workload and precision")
queue_ax.legend()
fig.tight_layout()
fig.savefig(out / "anomaly-shuttle.svg")
plt.close(fig)
