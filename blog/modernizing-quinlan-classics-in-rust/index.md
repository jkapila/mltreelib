# Cubist models and C5.0 rules: what can a model tree expose?

A constant-leaf decision tree gives every observation in a leaf the same prediction. A model tree can fit a local regression inside that leaf. Rule extraction offers another view: a set of conditions and associated predictions that a reviewer can inspect.

Quinlan's C4.5, C5.0, M5 and Cubist lineages are useful reference points for this work. `mltreelib` explores tree splitting, linear leaves, rule extraction and neighbor adjustments in a Rust core with Python bindings. These are implementation choices with their own contracts; similarity to a named algorithm does not establish exact equivalence to another package.

## How do linear leaves change decision tree regression?

A tree first routes a row into a partition. A linear leaf then computes a prediction from the features available to that leaf. This can represent a local slope with fewer partitions than a constant leaf, but leaf predictions can extrapolate and neighboring leaves need not join smoothly.

The [regression pathway](<https://jkapila.github.io/mltreelib/guides/regression/index.md>) compares mean and linear leaves on synthetic missing-data examples. It also fits the selected configuration on a held-out split of the real diamonds dataset. The comparison is a demonstration, not proof that linear leaves always win.

## What do extracted rules tell a reader?

Rules can make feature conditions easier to review than a large tree. They can also overlap, depend on pruning choices or use a default when no rule applies. Check the extracted rule set against its own predictions, leaf support and category mapping before using it as a business policy.

See [rules, summaries and neighbor adjustments](<https://jkapila.github.io/mltreelib/examples/rules-summary-knn/index.md>) for API examples, and [tree inspection](<https://jkapila.github.io/mltreelib/examples/inspection/index.md>) for named splits and leaves. The [prediction and schema guide](<https://jkapila.github.io/mltreelib/guides/serving-contract/index.md>) records the handoff checks that still need work.

## Does Rust make the model production-ready?

Rust owns the core computation, while Python exposes the interfaces. Runtime safety and performance are valuable goals, but production readiness also requires correct missing and unseen-category behavior, stable persistence, independent evaluation and reliable explanations. The current audit tracks those gaps rather than treating an implementation language as proof.
