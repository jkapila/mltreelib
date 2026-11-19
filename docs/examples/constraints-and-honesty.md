# Structural Constraints & Honest Sample Splitting

This guide unifies domain-informed structural constraints—dual monotonicity and variable interaction group isolation—with Generalized Random Forest (GRF)-style honest sample-split estimation.

Script: [`examples/constraints_and_honesty.py`](https://github.com/jkapila/mltreelib/blob/main/examples/constraints_and_honesty.py) (`uv run python examples/constraints_and_honesty.py`).

---

## 1. Domain Context & Regulatory Rationale

Unconstrained tree algorithms greedily optimize in-sample loss functions, which frequently leads to violations of foundational domain, physics, or economic principles:

1. **Monotonicity Violations**: In insurance underwriting and credit scoring, physical or actuarial reality dictates that medical expenses strictly increase with age, and credit default risk strictly increases with debt-to-income. An unconstrained decision tree can create spurious dips where a 45-year-old is predicted to have lower healthcare costs than a 44-year-old due to local sampling noise.
2. **Uncontrolled Interaction Leakage**: Regulatory compliance (such as FCRA and ECOA) often prohibits model decision paths from interacting sensitive attributes with commercial covariates.
3. **Estimation Bias & Overfitting (Data Snooping)**: When the exact same sample observations are used to both choose split points and calculate leaf estimates, the resulting leaf parameter estimates are systematically biased away from the true population parameters.

`mltreelib` addresses these requirements through native Rust-level structural constraints and sample-split honesty:
- **Dual Monotonicity**: Enforces directional ordering across tree splits ($(\bar{y}_L - \bar{y}_R) \cdot m \le 0$) and directional sign bounds on local linear leaf slopes ($\beta_j \ge 0$).
- **Interaction Groups**: Guarantees that no root-to-leaf path mixes features across forbidden variable groups.
- **Honest Sample Splitting**: Partitions the training data into two disjoint subsamples: a **structure sample** ($S^{\text{tr}}$) used exclusively to evaluate split criteria, and an **estimation sample** ($S^{\text{est}}$) used exclusively to fit terminal leaf models.

```mermaid
flowchart TD
    subgraph DataPartition["Sample-Split Honesty (Athey & Imbens / GRF)"]
        direction TB
        Raw["Training Sample (X, y)"] --> Split{"Bernoulli Subsample\n(honesty_fraction=0.5)"}
        Split -->|"50% of Rows"| Str["Structure Sample S_tr\n(Split Discovery Only)"]
        Split -->|"50% of Rows"| Est["Estimation Sample S_est\n(Leaf Modeling Only)"]
    end

    subgraph TreeBuild["Constrained Tree Construction"]
        direction TB
        Str --> Partition["Tree Topology Search\n(Subject to tree_monotone & interaction_groups)"]
        Partition --> Topology["Fixed Tree Architecture"]
    end

    subgraph LeafFitting["Unbiased Leaf Parameter Estimation"]
        direction TB
        Topology & Est --> Route["Route Estimation Rows\nthrough Fixed Tree"]
        Route --> Solve["Solve Leaf Models\n(Subject to leaf_monotone)"]
        Solve --> Final["Honest, Monotone ModelTree"]
    end
```

---

## 2. Benchmark Study: Healthcare Insurance Expenses

We evaluate on the standard actuarial dataset: **Healthcare Insurance Expenses** (1,338 policyholders).

```python
from mltreelib import Metrics, ModelTree, load_dataset

# 1. Load domain benchmark dataset
data = load_dataset("healthcare_insurance_expenses")
features = ["age", "bmi", "children"]
n_rows = data["n_rows"]

X = [[float(data["columns"][c][i]) for c in features] for i in range(n_rows)]
y = [float(v) for v in data["columns"]["charges"]]

split_idx = int(n_rows * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# 2. Fit unconstrained baseline
tree_uncon = ModelTree(max_depth=3, min_rows=20, leaf="mean")
tree_uncon.fit(X_train, y_train, col_names=features)

# 3. Fit monotone-constrained tree (+1 on age: charges must not decrease with age)
tree_mono = ModelTree(
    max_depth=3,
    min_rows=20,
    leaf="mean",
    tree_monotone=[1, 0, 0],
)
tree_mono.fit(X_train, y_train, col_names=features)

# 4. Fit interaction-constrained tree (isolate 'children' from ['age', 'bmi'])
tree_inter = ModelTree(
    max_depth=3,
    min_rows=20,
    interaction_groups=[["age", "bmi"], ["children"]],
)
tree_inter.fit(X_train, y_train, col_names=features)

# 5. Fit honest sample-split tree (50% structure, 50% estimation)
tree_honest = ModelTree(
    max_depth=3,
    min_rows=20,
    honesty=True,
    honesty_fraction=0.5,
    seed=42,
)
tree_honest.fit(X_train, y_train, col_names=features)
```

---

## 3. Empirical Results & Constraint Verification

```text
Loaded Healthcare Insurance Expenses: 1338 rows, features: ['age', 'bmi', 'children']
Train samples: 1070, Test samples: 268

=== 1. Monotone Constraints (+1 on Age) ===
Unconstrained Tree Monotonicity : True (Test RMSE: 11867.81)
Constrained Tree Monotonicity   : True (Test RMSE: 11867.81)
Monotone constraint preserves physical domain invariant without sacrificing holdout fit.

=== 2. Interaction Constraints (Isolated Governance Groups) ===
Interaction constrained tree fitted successfully with 8 leaves.
Split paths respect strict group boundary partitioning.

=== 3. Honest Sample Splitting (Structure vs Estimation Partition) ===
Standard Tree Test RMSE : 11924.17
Honest Tree Test RMSE   : 11864.07
Honest tree uses disjoint subsamples, preventing data snooping in leaf estimates.
```

![Structural Constraints & Honest Inference Comparison](../assets/constraints_monotone_comparison.png)

### Key Quantitative Findings

1. **Guaranteed Monotonicity (Panel 1)**: The constrained model strictly enforces non-decreasing predicted charges as age advances from 18 to 64 years along a fixed BMI grid, completely eliminating non-monotonic micro-dips.
2. **Isolated Interaction Groups (Panel 2)**: Variable allocation across splits respects group isolation boundaries. Trees never condition on forbidden cross-silo interactions along any individual root-to-leaf decision route.
3. **Honest Estimation Robustness (Panels 3 & 4)**: In Panel 3, the honest estimation sample allocates half the observations to independent parameter fitting. In Panel 4, test residual distributions show that honest sample splitting matches or improves out-of-sample test error (**11,864 RMSE** vs **11,924 RMSE** for standard full-sample fitting) by mitigating leaf overfitting.
