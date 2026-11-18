"""Synthetic anomaly review workflow through AnomalyTree.

Run: uv run python examples/anomaly_workflow.py
Matplotlib renders the figure; planted anomalies are known only in this simulation.
"""
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mltreelib import AnomalyTree

rng = Random(2042)
reference = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(2000)]
normal = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(1000)]
planted = [[rng.gauss(3.3, .6), rng.gauss(3.3, .6)] for _ in range(100)]
review = normal + planted
truth = [0] * len(normal) + [1] * len(planted)
model = AnomalyTree(kind="eif", max_depth=8, extension_level=1, seed=42)
model.fit(reference, col_names=["measure_a", "measure_b"])
scores = model.score_samples(review)
# Fixed review capacity, chosen before looking at the synthetic labels.
budget = 100
ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
selected = ranked[:budget]
caught = sum(truth[i] for i in selected)
print("reviewed:", budget, "of", len(review))
print("planted anomalies among reviewed:", caught)
print("precision at budget:", round(caught / budget, 3))
print("recall of planted anomalies:", round(caught / sum(truth), 3))
print("training contamination threshold:", round(model.threshold, 3))
print("flagged at that threshold:", sum(model.predict(review)))

out = Path("docs/assets/guides")
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter([r[0] for r in review], [r[1] for r in review], c=truth,
                cmap="coolwarm", s=22, alpha=.65)
axes[0].scatter([review[i][0] for i in selected], [review[i][1] for i in selected],
                facecolors="none", edgecolors="black", s=80, label="selected")
axes[0].set(xlabel="Measure A", ylabel="Measure B", title="Review set and selected cases")
axes[0].legend()
axes[1].bar(range(len(scores)), [scores[i] for i in ranked],
            color=["#f4511e" if truth[i] else "#00897b" for i in ranked])
axes[1].axvline(budget-.5, color="black", linestyle="--", label="review budget")
axes[1].set(xlabel="Rank, highest score first", ylabel="Anomaly score", title="Ranked review queue")
axes[1].legend(); fig.tight_layout(); fig.savefig(out / "anomaly-workflow.svg"); plt.close(fig)
