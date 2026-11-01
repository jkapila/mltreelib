# Hierarchical Leaf Retrain & Micro-Model Routing

Tree partitions carve heterogeneous feature spaces into homogeneous sub-populations. While standard model trees fit models using the same features and targets at split and leaf time, complex empirical settings often require **hierarchical model routing**:

1. A **macro-tree** splits on aggregated or cohort-level features (e.g. city, make, model year, fleet tier) to segment cohorts.
2. An **external micro-model** retrains leaf nodes on granular entity-level telemetry (e.g. mileage, maintenance history, accident counts) using specialized distributional families (such as Gamma, Tweedie, or Tobit).
3. Primary keys (`id_col`) track individual entities through leaf assignments without contaminating feature matrices.

Script: [`examples/hierarchical_leaf_retrain.py`](https://github.com/jkapila/mltreelib/blob/main/examples/hierarchical_leaf_retrain.py) (`uv run python examples/hierarchical_leaf_retrain.py`).

---

## 1. Internal vs. External Retraining Modes

The `retrain()` method supports two operational modes:

- `mode="internal"`: Mutates the tree's existing leaf models in-place, keeping the tree structure intact.
- `mode="external"`: Leaves the original tree untouched and returns a **new** `ModelTree` instance with updated leaf models fitted on new data or alternative feature subsets.

```python
from mltreelib import ModelTree

# Step 1: Macro-level segmentation tree
macro_tree = ModelTree(max_depth=2, min_rows=2, leaf="mean")
macro_tree.fit(
    X_macro,
    y_baseline,
    col_names=["city_tier", "fleet_type"]
)

# Step 2: Retrain leaves externally with granular telemetry
micro_tree = macro_tree.retrain(
    X=df_telemetry,
    y=y_fine,
    mode="external",
    leaf="linear",
    leaf_family="gamma",
    leaf_features=["mileage", "service_count"],
    control={"l2": 1e-2, "max_iter": 25}
)
```

---

## 2. Granular Leaf Feature Selection (`leaf_features`)

You can decouple the split features from the leaf regression features using `leaf_features` (or `retrain_cols`):

- **Splitting Features**: Used by internal tree nodes to route samples to the correct leaf (e.g. `city_tier`, `fleet_type`). These columns must be present in the retraining dataset.
- **Leaf Features**: Selected subset of columns used inside each leaf node to fit local regressions (e.g. `mileage`, `service_count`).

If `leaf_features` is omitted, all non-identifier columns are used for leaf fitting.

---

## 3. Entity Tracking with `leaf_samples(X, id_col=...)`

To route entities and group downstream analyses by leaf assignments, pass `id_col` to `leaf_samples()`:

```python
# Group vehicle identification numbers (VINs) by leaf assignment
vin_groups = micro_tree.leaf_samples(df_telemetry, id_col="vin")

for leaf_id, vins in vin_groups.items():
    print(f"Leaf {leaf_id} manages {len(vins)} vehicles: {vins[:3]}...")
```

If `id_col` is `None` (default), `leaf_samples()` returns integer row indices `0..N-1`.

---

## 4. Family Control Knobs

Algorithm controls for leaf fitting are passed via a flexible `control` dictionary:

```python
control = {
    "l2": 1e-3,           # L2 ridge regularization penalty
    "max_iter": 30,       # Maximum Fisher scoring / IRLS iterations
    "tol": 1e-6,          # Convergence tolerance on parameter updates
    "extra": {            # Family-specific optional hyper-parameters
        "power": 1.5,     # Tweedie compound variance power p
        "censor_lower": 0 # Tobit left-censoring threshold
    }
}

retrained = macro_tree.retrain(
    X=df,
    y=y,
    mode="internal",
    leaf="linear",
    leaf_family="tweedie",
    control=control
)
```
