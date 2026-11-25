"""A reproducible clustering workflow using the unified ClusterTree API.

Run: uv run python examples/guides/clustering_workflow.py
The figures need matplotlib. All data is synthetic.
"""
from collections import Counter
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mltreelib import ClusterTree

rng = Random(2026)
def sample(n):
    rows = []
    for _ in range(n):
        center = -2.0 if rng.random() < .5 else 2.0
        rows.append([rng.gauss(center, .4), rng.gauss(center, .4)])
    return rows

train, review, second_train = sample(2000), sample(1000), sample(2000)
model = ClusterTree(kind="pct", max_depth=2, min_rows=12)
model.fit(train, col_names=["measure_a", "measure_b"])
labels = model.predict(review)
print("training clusters:", dict(Counter(model.labels_)))
print("held-out clusters:", dict(Counter(labels)))
print("rules:", model.rules())

# Compare whether pairs share a cluster; cluster numbers can change between fits.
second = ClusterTree(kind="pct", max_depth=2, min_rows=12)
second.fit(second_train, col_names=["measure_a", "measure_b"])
other = second.predict(review)
pairs = [(i, j) for i in range(len(review)) for j in range(i + 1, len(review))]
agreement = sum((labels[i] == labels[j]) == (other[i] == other[j]) for i, j in pairs) / len(pairs)
print("pairwise agreement across fits:", round(agreement, 3))

out = Path("docs/assets/guides")
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter([r[0] for r in review], [r[1] for r in review], c=labels, cmap="viridis", s=26)
axes[0].set(xlabel="Measure A", ylabel="Measure B", title="Held-out segment assignments")
counts = Counter(labels)
axes[1].bar([str(k) for k in sorted(counts)], [counts[k] for k in sorted(counts)], color="#168378")
axes[1].set(xlabel="Cluster ID", ylabel="Held-out rows", title="Segment sizes")
fig.tight_layout(); fig.savefig(out / "clustering-workflow.svg"); plt.close(fig)
