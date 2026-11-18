"""UCI Statlog Shuttle anomaly proxy, using the provided train/test split.

Place shuttle.trn and shuttle.tst in datasets/external/shuttle/.
Run: uv run python examples/anomaly_shuttle_evaluation.py
Class 1 is the reference class; other classes are an anomaly proxy.
"""
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
print("test average precision:", round(Metrics().pr_curve(truth, scores)["average_precision"], 4))

out = Path("docs/assets/guides"); out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6,4))
plots.plot_pr(truth, scores, ax=ax)
ax.axhline(sum(truth)/len(truth), linestyle="--", color="#f4511e", label="test prevalence")
ax.set_title("UCI Shuttle: held-out anomaly proxy")
ax.legend(); fig.tight_layout(); fig.savefig(out / "anomaly-shuttle.svg"); plt.close(fig)
