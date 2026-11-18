# Uplift modeling with causal trees: who benefits from an intervention?

A customer who is likely to visit your site is not necessarily a customer who will visit **because** of an email. If the email costs money or annoys customers, the difference matters.

Uplift modeling asks a different question from response prediction: **how much does the outcome change under treatment?** A causal tree groups observations to estimate how that change varies across customers. Randomized treatment assignment, or defensible observational assumptions, is needed to interpret those differences causally.

## What does an uplift model estimate?

For pre-treatment features \\(X=x\\), the conditional average treatment effect is \\(E\[Y(1)-Y(0)\\mid X=x\]\\). In a randomized trial, treated and control outcomes within a sufficiently supported group can estimate that contrast. We do not observe both potential outcomes for one person, so an individual prediction is not an observed individual effect.

`ModelTree(gain="causal")` accepts outcomes and treatment indicators as a tuple. Honesty separates rows used for tree structure from those used for leaf estimation. It does not remove confounding or guarantee that a discovered segment will generalize. See [Athey and Imbens (2016)](<https://www.pnas.org/doi/10.1073/pnas.1510489113>).

## Does targeting beat treating everyone?

We tested the workflow on the historical [Hillstrom randomized email trial](<https://www.tensorflow.org/datasets/catalog/hillstrom>). The example compares the women's email with no email and measures a website visit within two weeks. It uses 25,615 training, 8,539 validation and 8,539 test records. The tree uses pre-treatment attributes only.

Validation selected a policy that treats every test customer. Its estimated incremental visit rate was **0.04517 per eligible customer**, with an approximate 95% interval of **0.03108 to 0.05927**. Treat-all has the same value because it is the same policy in this run. Treat-none has zero incremental visits by definition.

That result supports a difference between the two randomized arms in this split. It does **not** show that the targeting model improves on treating everyone. The example assumes zero campaign cost because the dataset does not provide it. Visits are not purchases or profit, and a historical experiment is not a substitute for a new trial.

## How should you evaluate uplift?

1. Define the intervention, outcome, cost and eligible population before fitting.
2. Keep treatment and outcomes out of pre-treatment features. Split training, policy selection and test data.
3. Compare the fixed policy with treat-all and treat-none on an independent randomized holdout. Report treated and control counts in targeted groups.
4. Report uncertainty and campaign costs. Re-test the proposed policy in the setting where it will run.

The [uplift pathway](<https://jkapila.github.io/mltreelib/guides/uplift/index.md>) contains the synthetic and real evaluation, data source, split, limitations and commands. Its synthetic example also reports PEHE, which requires known individual effects and is not directly observable in an ordinary real trial.
