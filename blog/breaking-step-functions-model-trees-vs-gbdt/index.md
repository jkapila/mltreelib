# Model trees versus step functions for decision tree regression

A constant-leaf regression tree approximates a slope with steps. A linear-leaf model tree can represent a slope inside a partition. That difference can matter when the response varies continuously, but it does not make one model family universally better.

The plot is an illustration from the repository's comparison examples. Compare models on the actual dataset, split and budget that matter for the decision. A single curve does not establish performance across physical systems.

## When can a linear leaf help?

A linear leaf can fit a local trend without adding another split for every small change. It also introduces coefficients that need support and regularization. Missing predictors, correlated features and data outside the training range can make those coefficients unstable or predictions implausible. Adjacent leaves can have discontinuities.

In the [regression pathway](<https://jkapila.github.io/mltreelib/guides/regression/index.md>), four configurations are selected on synthetic validation data with missing inputs. Linear leaves at depth 2 achieve the lowest validation RMSE in that particular run. The example then fits the same configuration to 6,000 real diamond records and evaluates 2,000 held-out records. Its test RMSE is 1,381.867 versus 3,982.039 for a training-mean baseline. This is not a comparison against tuned boosting models.

## Can a model tree extrapolate?

A linear leaf can produce a changing prediction outside the observed feature range. That flexibility is not proof that the extrapolation follows physical laws. Check plausible ranges and test on an appropriate out-of-time or out-of-range holdout. Regularization, constraints and the model's configured prediction behavior all matter.

## How should I compare a model tree with gradient boosting?

Use the same training and test splits, preprocessing rules, tuning budget and evaluation metric. Report runtime, completion failures and uncertainty alongside prediction error. Inspect errors where decisions matter. The [benchmark overview](<https://jkapila.github.io/mltreelib/benchmarks/elo/index.md>) describes the repository's comparisons, while the [evaluation guide](<https://jkapila.github.io/mltreelib/guides/evaluation/index.md>) covers baselines and held-out reporting.
