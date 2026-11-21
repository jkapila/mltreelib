# Naive Bayes and Bayesian Classifier Leaves for `mltreelib`

**Date:** 2026-11-21  
**Scope:** Naive Bayes, mixed-type Bayes, one-dependence and k-dependence classifiers, Bayesian network classifiers, tree-augmented Naive Bayes, and recent extensions suitable as terminal leaves.  
**Decision informed:** Which probabilistic classifiers should enter the package as leaf families and what implementation order maximizes package value.  
**Confidence:** High for classical algorithms and architectural fit. Medium for recent neural and empirical-Bayes methods because evidence is recent and often preprint-level.

## 1. Executive Summary

Naive Bayes and Bayesian network classifiers are strong leaf candidates. They are fast, naturally probabilistic, robust to missing categorical values, and expose class-conditional distributions or dependency graphs.

The package currently has no Naive Bayes or Bayesian classifier implementation. `LeafKind::Prob`, `Odds`, and `LogOdds` are zero-dimensional binary summaries. A probabilistic classifier leaf requires per-class state, so it needs a distinct fitted model type rather than a scalar leaf statistic.

### Recommended ladder

1. **Categorical Naive Bayes leaf.** Counts, priors, and Laplace or m-estimate smoothing.
2. **Gaussian Naive Bayes leaf.** Per-class means and variances for numeric predictors.
3. **Mixed-type Naive Bayes.** Independent per-feature likelihoods with feature-specific distributions.
4. **AODE-style one-dependence leaf.** Average several one-parent classifiers without expensive structure search.
5. **TAN-style leaf.** One additional parent per feature, learned from conditional mutual information.
6. **k-dependence Bayes.** Optional larger-leaf mode.
7. **NeuralKDB and other representation-learned classifiers.** Research-only.

The historical NB-tree hybrid by Kohavi gives direct precedent: a decision tree splits data, then Naive Bayes models leaves.

## 2. Classical Naive Bayes

Naive Bayes assumes conditional independence:

\[
P(Y=y\mid X=x)\propto P(Y=y)\prod_j P(X_j=x_j\mid Y=y)
\]

### 2.1 Categorical Naive Bayes

For each class and feature value, estimate a smoothed conditional probability:

\[
\hat P(x_j=v\mid y)=\frac{n_{y,j,v}+\alpha}{n_{y,j}+\alpha V_j}
\]

where \(V_j\) is the number of observed levels.

Strengths:

- Additive counts.
- Natural categorical support.
- Missing feature values can be skipped from the product likelihood.
- Fast prediction.

Weaknesses:

- Strict independence assumption.
- Sparse class-feature counts.
- Category ordering is irrelevant, which is correct.

### 2.2 Gaussian Naive Bayes

For numeric feature \(j\):

\[
X_j\mid Y=y\sim\mathcal N(\mu_{y,j},\sigma_{y,j}^2)
\]

Strengths:

- One-pass mean and variance accumulation.
- Reuses Gram-style sufficient statistics.
- Natural continuous support.

Weaknesses:

- Class-specific variance is unstable with few rows.
- Gaussian assumption may be poor.
- Must floor variance and handle zero-variance features.

### 2.3 Mixed-type Naive Bayes

Each feature contributes its own likelihood:

- Categorical: smoothed category frequency.
- Binary: Bernoulli or Beta-binomial.
- Count: Poisson or negative-binomial.
- Positive continuous: Gamma or log-normal.
- Numeric: Gaussian.

This design matches `mltreelib`'s typed-frame philosophy better than global numeric coercion.

## 3. Smoothing

### 3.1 Laplace and m-estimate

Laplace smoothing adds one pseudo-count per category. m-estimate smoothing uses a user or global prior weight.

Package default should be m-estimate:

\[
\hat P(x_j=v\mid y)=\frac{n_{y,j,v}+mP(v\mid y)}{n_{y,j}+m}
\]

where \(P(v\mid y)\) is the prior category probability. This is less aggressive than adding one count to every cell.

### 3.2 Hierarchical empirical Bayes

Recent work proposes hierarchical empirical-Bayes smoothing for Naive Bayes and AODE extensions. It uses data-driven prior strength and reports calibration improvements in preprint experiments.

Source: “Hierarchical Empirical-Bayes Naive Bayes: Minimax Smoothing and Calibration with AODE Extension,” [arXiv:2608.11162](https://arxiv.org/abs/2608.11162).

Package decision:

- Implement m-estimate first.
- Add empirical-Bayes smoothing as experimental mode.
- Never silently replace default smoothing.

## 4. NB-Tree Hybrid

Kohavi's 1996 *Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid* is direct prior art for the package's architecture.

Procedure:

1. Grow a decision tree using standard splits.
2. At each leaf, fit a Naive Bayes model.
3. Predict using the leaf's local class posterior.

This can outperform both a global NB model and a constant-leaf tree because local class-conditional distributions vary across partitions.

Source: [KDD 1996 paper](https://cdn.aaai.org/KDD/1996/KDD96-033.pdf).

**Package value:** very high. This should become a benchmark when implementing probabilistic leaves.

## 5. One-Dependence Bayes

One-dependence classifiers allow each feature one additional predictor parent besides the class.

### 5.1 TAN

Tree-Augmented Naive Bayes constructs a maximum-weight spanning tree over features. Edge weights use conditional mutual information given the class.

Source: Friedman, Geiger, and Goldschmidt, “Bayesian Network Classifiers,” *Machine Learning* 29, 1997. [DOI](https://doi.org/10.1023/A:1007465528199).

Strength:

- Captures one important feature dependency per node.
- Tree structure limits parameter growth.

Weakness:

- Mutual-information estimation inside small leaves is noisy.
- Structure search cost grows with feature count.

### 5.2 AODE

AODE averages one-dependence estimators in which each feature serves once as the shared parent of all others.

Source: Webb, Boughton, and Wang, “Not So Naive Bayes: Aggregating One-Dependence Estimators,” *Machine Learning* 58, 2005. [DOI](https://doi.org/10.1007/s10994-005-0021-1).

Strength:

- No structure search.
- Usually stronger than TAN.
- Count-based implementation is straightforward.

Weakness:

- \(O(d^2)\) counts and prediction work.
- Parameter count still grows quickly.

### 5.3 WAODE

WAODE weights one-dependence estimators by their local accuracy. It offers a moderate complexity increase over AODE.

### 5.4 Package ordering

AODE should precede TAN for default leaves because AODE requires no structure search and is easier to regularize. TAN is a large-leaf or research mode.

## 6. k-Dependence Bayes

KDB allows each feature up to \(k\) parent predictors in addition to the class.

- **KDB:** original fixed dependence order.
- **SKDB:** selects dependence order using support coverage.
- **K-free dependence:** adaptive parent count.
- **MNKDB:** class-specific dependency networks.
- **KDF:** ensemble across dependence orders.

Strength:

- Better dependence modeling.
- Still fast relative to full Bayesian network search.

Weakness:

- Parameter explosion and sparse conditional tables.
- Small leaves are especially unsuitable.

Package decision:

- Optional mode for large leaves only.
- Minimum support and smoothing required.
- Report selected parents and support coverage.

## 7. Tree-Augmented Naive Bayes

TAN can also refer to a Bayesian network whose structure is constrained to the split tree itself. Features at one tree branch receive conditional distributions from related branch nodes.

This differs from using TAN as a leaf classifier:

- Split-side TAN uses tree ancestry as dependency structure.
- Leaf-side TAN learns a separate feature dependency tree.

Both are valid research directions. Split-side TAN may fit `mltreelib`'s divide-and-conquer theme more closely and should be considered with split criteria, not leaf fitting.

## 8. Generalized Naive Bayes

Generalized Naive Bayes relaxes full independence by allowing selected feature dependencies. Recent continuous-feature work derives structure from pairwise copulas and proposes greedy matroid-based structure learning.

Source: Papp, Szilágyi, and Kovács, “On Generalized Naive Bayes with Continuous Features,” [arXiv:2609.23819](https://arxiv.org/abs/2609.23819).

Strength:

- Direct extension of classical NB.
- Copula-based view separates marginals from dependence.
- Supports greedy structure discovery.

Weakness:

- Very recent preprint.
- Continuous pairwise dependence estimation inside small leaves is demanding.
- Structure search can overfit.

Package decision:

- Research phase after AODE.
- Reuse feature graphs and copula utilities if implemented elsewhere.

## 9. Bayesian Network Classifiers

A Bayesian network classifier represents:

\[
P(X,Y)=\prod_i P(X_i\mid Pa_i,Y)
\]

with a class root or class variable integrated into the network.

Classical families:

- Naive Bayes.
- TAN.
- AODE.
- KDB.
- Flexible Naive Bayes.
- FKDB.
- Bayesian network classifiers with optimized class-variable parameters.

Strengths:

- Missing-data support can be explicit.
- Incremental count updates are natural.
- Dependency graphs explain predictions.

Weaknesses:

- Structure learning is expensive.
- Dense conditional tables explode.
- Leaf sample sizes quickly become insufficient.

Package decision:

- Do not implement general Bayesian network structure learning in core leaves.
- Implement restricted structures whose complexity is controlled: NB, AODE, optional TAN/KDB.

## 10. Neural and Representation-Learned Bayes

### 10.1 NeuralKDB

NeuralKDB learns distributional representations of feature values from co-occurrence patterns and parameterizes higher-order conditional probabilities without directly storing every high-order table.

Source: Zhang et al., “Towards the Next-generation Bayesian Network Classifiers,” [arXiv:2508.11145](https://arxiv.org/abs/2508.11145).

Reported behavior:

- Better accuracy than conventional BNCs, KDB, TAN, NB, and random forest on 60 UCI datasets.
- Parameter count does not grow directly with dependence order in the same way as tabular KDB.
- Requires gradient training.

Package decision:

- Not a core leaf.
- Useful if `mltreelib` later adds a small neural leaf infrastructure.

### 10.2 Smart Bayes

*Discriminative Classification with Generative Features* bridges generative and discriminative learning by creating likelihood-ratio features and fitting a logistic model.

Source: [arXiv:2512.01097](https://arxiv.org/abs/2512.01097).

Package decision:

- Reject for first implementation.
- Existing softmax/logistic leaves already provide the discriminative half.

## 11. Discriminant Analysis Connection

Naive Bayes is the diagonal and conditionally independent endpoint of discriminant-analysis regularization.

Le and Hastie's SQDA creates a continuous path:

- QDA at zero regularization.
- Naive Bayes at infinite regularization.
- Sparse selected interactions in between.

This makes discriminant analysis and Bayes research naturally connected. The package can expose:

- Gaussian NB as a special case of diagonal LDA.
- AODE as a structured extension beyond independence.
- SQDA or RDA as covariance-aware alternatives.

They should remain distinct estimators, but shared covariance and sufficient-statistic utilities should avoid duplication.

## 12. Leaf Architecture

### 12.1 State

A probabilistic classifier leaf should persist:

- Class prior.
- Per-class per-feature sufficient statistics.
- Smoothing policy and strength.
- Feature type and encoding map.
- Class means and variances for numeric components.
- Parent sets for AODE/TAN/KDB.
- Support coverage for conditional tables.
- Optional feature-selection mask.

### 12.2 Fit behavior

- Zero rows: error.
- One class: return that class probability.
- Insufficient rows for requested structure: fallback to simpler NB and report fallback.
- Missing categorical value: skip its likelihood term.
- Missing numeric value: skip term or use an explicit missing component.
- Unseen categorical level: use class prior or a documented smoothing fallback.
- Zero variance numeric feature: deterministic component.

### 12.3 Prediction

Return class probabilities, class labels, and optionally per-feature log-likelihood contributions for explanation.

## 13. Relationship to Tree Growth

Naive Bayes leaves fit after tree partition selection. They do not change split logic in the first implementation.

Future possibilities:

- Model-based split criteria using class-conditional likelihood.
- Split-side TAN structures.
- Monotone constraints for categorical NB.
- Missingness-aware split gains.

These are separate from adding NB leaves.

## 14. Implementation Order

### Phase 1 — Core NB

1. Categorical NB leaf.
2. Gaussian NB leaf.
3. Mixed-type NB leaf.
4. Laplace and m-estimate smoothing.
5. Missing-value skip semantics.
6. NB-tree benchmark against constant and linear leaves.

### Phase 2 — Dependence

1. AODE leaf.
2. WAODE weighting.
3. Optional TAN leaf with minimum leaf-size guard.
4. Optional KDB leaf.

### Phase 3 — Research

1. Generalized NB using copula structure.
2. Hierarchical empirical-Bayes smoothing.
3. SQDA integration.
4. NeuralKDB behind experimental neural leaf infrastructure.

## 15. Evaluation Protocol

Report:

- Accuracy, macro F1, log loss, Brier score, and calibration error.
- Prediction latency.
- Training time and memory.
- Missingness and unseen-level behavior.
- Performance by leaf size and feature count.
- Comparison with softmax/logistic, LDA/QDA, KNN, and boosted trees.
- Comparison with global NB to verify local-leaf benefit.
- Structure size and support coverage for AODE/TAN/KDB.
- Determinism and fallback frequency.

Use synthetic data with known class priors, marginal distributions, and dependencies. Include categorical, numeric, mixed, imbalanced, and missing-heavy cases.

## 16. Risks

1. **Conditional independence failure.** NB can be badly biased when features interact.
2. **Zero-frequency probabilities.** Smoothing must be explicit and consistent.
3. **Unequal penalty scales.** Existing CCD and GLM L1 scaling differ across solvers; NB uses its own count likelihood.
4. **Small-leaf sparsity.** AODE and TAN overfit small leaves.
5. **Categorical cardinality.** One-hot or count tables can grow rapidly.
6. **Missing-value ambiguity.** Skip versus missing-component semantics must differ by model.
7. **Calibration overstatement.** Classifier accuracy does not imply calibrated probabilities.
8. **Preprint overreach.** NeuralKDB and generalized NB remain research-grade.
9. **Duplicate concepts.** Gaussian NB and diagonal LDA share mathematics but remain distinct public leaves.

## 17. Primary Sources

1. Domingos, P. and Pazzani, M. (1997). “On the Optimality of the Simple Bayesian Classifier under Zero-One Loss.” *Machine Learning* 29. [DOI](https://doi.org/10.1023/A:1007413511361).
2. Kohavi, R. (1996). “Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid.” *KDD*. [Paper](https://cdn.aaai.org/KDD/1996/KDD96-033.pdf).
3. Friedman, N., Geiger, D., and Goldschmidt, M. (1997). “Bayesian Network Classifiers.” *Machine Learning* 29. [DOI](https://doi.org/10.1023/A:1007465528199).
4. Webb, G. I., Boughton, J. R., and Wang, Z. (2005). “Not So Naive Bayes: Aggregating One-Dependence Estimators.” *Machine Learning* 58. [DOI](https://doi.org/10.1007/s10994-005-0021-1).
5. Jiang, H. et al. (2009). “Hidden Naive Bayes.” *IEEE TKDE* 21(10). [DOI](https://doi.org/10.1109/TKDE.2009.106).
6. Sahami, M. (1996). “Learning Limited Dependence Bayesian Classifiers.” *KDD*. [DOI](https://doi.org/10.1145/207604.207613).
7. Le, Y. and Hastie, T. (2016). “Sparse Quadratic Discriminant Analysis and Community Bayes.” [arXiv:1407.4543](https://arxiv.org/abs/1407.4543).
8. Zhang, H. et al. (2025). “Towards the Next-generation Bayesian Network Classifiers.” [arXiv:2508.11145](https://arxiv.org/abs/2508.11145).
9. Papp, Á., Szilágyi, B., and Kovács, E. A. (2026). “On Generalized Naive Bayes with Continuous Features.” [arXiv:2609.23819](https://arxiv.org/abs/2609.23819).
10. Sugahara, S., Kato, K., and Ueno, M. (2026). “Learning Bayesian Network Classifiers to Minimize the Class Variable Parameters.” *JMLR* 27. [Paper](https://www.ai.lab.uec.ac.jp/wp-content/uploads/2026/09/24-1901_corrected_clean.pdf).
