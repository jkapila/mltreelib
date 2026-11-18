# Clustering: a practical pathway

Find groups with similar measured characteristics. Decide first what a useful segment must support, such as profiling or a service strategy.

**Path:** Define the decision → Prepare data → Fit a baseline → Evaluate → Inspect → Act and monitor.

## 1\. Define the data and split

The synthetic example uses two numeric measurements. Scaling and distance affect similarity. Remove identifiers and outcomes that are unavailable when assigning future rows. Fit preprocessing on training data only.

## 2\. Fit and inspect a small example

This complete example uses the package and Python standard library. It uses synthetic data and a fixed seed. Run it from a Python environment with `mltreelib` installed.

```
from random import Random
from mltreelib import ClusterTree

rng = Random(43)
x = [[rng.gauss(center, .4), rng.gauss(center, .4)] for center in (-2, 2) for _ in range(1000)]
model = ClusterTree(kind="pct", max_depth=2, min_rows=10)
model.fit(x, col_names=["measure_a", "measure_b"])
labels = model.predict(x)
print("clustering labels:", sorted(set(labels)))
print("clustering rules:", model.rules())
```

## 3\. See the result

The figure comes from [`examples/modeling_pathways.py`](<https://github.com/jkapila/mltreelib/blob/main/examples/modeling_pathways.py>). Rendering the figures requires matplotlib. The example is a teaching illustration, not a benchmark.

## 4\. Evaluate before deciding

`ClusterTree(kind="pct")` creates a predictive clustering tree using the features as its unsupervised targets here. It can split a natural group into multiple leaves. Compare segment sizes, profiles and stability across samples and settings. Do not treat these synthetic labels as ground truth.

## 5\. Interpret and monitor

`rules()` reports cluster conditions, prototypes and sizes for this PCT example. Review small groups and check whether their rules are useful to domain experts. PCT prediction is shown here; check each engine's new-row behavior before deployment.

## 6\. Check a held-out segmentation

The [clustering workflow](<https://github.com/jkapila/mltreelib/blob/main/examples/clustering_workflow.py>) fits 2,000 synthetic records, assigns 1,000 independent review records and fits a second tree on another independent sample. It reports rule support, segment sizes and pairwise agreement between assignments from the two fits.

```
uv run python examples/clustering_workflow.py
```

For the fixed seed, the held-out segment sizes are 237, 273, 243 and 247. Pairwise agreement between the two fits is 0.750. The measure checks whether pairs are together or apart in both fits, so it does not require matching cluster IDs. It is a stability diagnostic, not evidence that the segments are useful. Also inspect the proportion of pairs in each category, small segments, segment profiles, and sensitivity to `max_depth` and `min_rows`. Use a relevant downstream business measure when available, without using the test outcomes to choose the final model.

## 7\. Choose an engine through `ClusterTree`

The following observations are from a small numeric, two-group synthetic check. They establish the interface paths exercised, not comparative model quality.

| `kind` | Training and prediction output checked | `rules()` in that check |
| --- | --- | --- |
| `pct` | Training labels and new-row labels | Conditions, prototypes and support |
| `cltree` | Labels can include `None` for unassigned rows | Bounding boxes and support |
| `hdbscan` | Training and new-row labels | Empty list |
| `urf` | Training and new-row labels | Empty list |

The sample check returned 8 PCT leaves, 3 CLTree groups plus unassigned points, 2 HDBSCAN groups, and a highly imbalanced URF assignment. These results depend on the data and settings. They do not rank the engines. Check whether a segment is stable and useful before assigning an action to it.

### Missing values and categories: product gaps

The wrapper currently converts `None` to `0.0` for `cltree`, `hdbscan` and `urf` during fitting and prediction. This is implicit imputation and conflicts with the package's missing-data goal. Track it as a product gap. Do not assume those engines distinguish missing from a real zero. They also reject raw strings in the checked inputs.

PCT accepted raw categorical labels and missing numeric entries in a separate check. Its returned rules displayed category codes rather than the original labels. In that check, an unseen category and a missing category went to different leaves. Category-label rendering and unseen-category routing need further investigation before those rules become a business handoff.

For density and proximity methods, check scaling, distance choice and the meaning of an unassigned or noise label. For all kinds, test the actual new-row behavior and input schema that your workflow uses.

## Real-data example

The [bank churn walkthrough](<https://jkapila.github.io/mltreelib/guides/real-data/index.md>) runs this workflow on a 10,000-row TabArena dataset with independent test rows.

## Next steps and references

[Clustering comparison](<https://jkapila.github.io/mltreelib/examples/clustering-comparison/index.md>) · [API reference](<https://jkapila.github.io/mltreelib/reference/api/cluster/index.md>)

Continue with [data preparation](<https://jkapila.github.io/mltreelib/guides/data/index.md>), [evaluation and monitoring](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>), or [model extensions](<https://jkapila.github.io/mltreelib/guides/advanced-modeling/index.md>). The [practical modeling pathways](<https://jkapila.github.io/mltreelib/guides/index.md>) show the other tasks. For broader coverage, see the [documentation audit](<https://github.com/jkapila/mltreelib/tree/main/references/research>).
