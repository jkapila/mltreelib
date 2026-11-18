# Review a fitted tree before handing it off

Use this checklist after you run a [modeling pathway](../guides/index.md). The example uses the concrete strength tree to show what to deliver with a fitted regression model.

Download [OpenML concrete strength 46917](https://www.openml.org/d/46917) to `datasets/concrete_compressive_strength.parquet`. Run this code from the repository root after `uv sync`.

```python
from pathlib import Path

from examples.real_tree_studies import load, rows
from mltreelib import ModelTree

df = load("concrete_compressive_strength", 46917).sample(frac=1, random_state=2026)
target = "ConcreteCompressiveStrength"
features = df.columns.drop(target).tolist()
train, test = df.iloc[:824], df.iloc[824:]
tree = ModelTree(leaf="linear", max_depth=2, min_rows=40, prune="none")
tree.fit(rows(train, features), train[target].tolist(), col_names=features)
pred = tree.predict(rows(test, features))

review = test[features + [target]].assign(predicted_mpa=pred)
review["error_mpa"] = review["predicted_mpa"] - review[target]
largest_error = review["error_mpa"].abs().nlargest(5).index
print(review.loc[largest_error].to_string(index=False))
print(tree.split_paths())
print([(leaf["id"], leaf["n"]) for leaf in tree.leaves()])
Path("concrete-tree.dot").write_text(tree.to_dot(), encoding="utf-8")
```

If Graphviz is installed, render that file with `dot -Tsvg concrete-tree.dot -o concrete-tree.svg`.

![Concrete model tree showing named split conditions and local linear leaves](../assets/guides/real-concrete-tree.svg)

Use the diagram to read the structure. [Check boundary routing](../guides/serving-contract.md#check-split-boundaries) before transcribing the displayed conditions.

## Prepare the handoff

1. **Inputs.** Record feature order, units, allowed categories and the prediction time. Keep a sample with a missing value.
2. **Evaluation.** Report errors for the intended decision. Include the baseline, split method and examples of large errors.
3. **Structure.** Export the DOT diagram and keep the structured splits and leaves. Check the support behind the smallest leaves.
4. **Deployment.** Compare scores on a fixed set of reference rows after moving the model into the serving environment. Include rows near a split and ones with absent fields.

The [concrete strength article](../blog/modernizing-quinlan-classics-in-rust.md) compares the tree with constant leaves. The [serving contract](../guides/serving-contract.md) lists the checks needed when you move from an example to a service. For the original churn exercise, see the [secondary case on the dataset page](../guides/real-data.md#secondary-example-bank-churn).
