# Association Rule Mining Landscape for `mltreelib`

**Date:** 2026-11-21  
**Scope:** Frequent itemsets, association rules, associative classification, sequential patterns, neurosymbolic rule learning, and their fit as terminal leaves in `mltreelib` trees.  
**Decision informed:** Whether association mining belongs in the tree leaf layer, which capability should enter the package, and which research directions deserve tracking.  
**Confidence:** High for classical algorithms and architectural placement. Medium for recent neural and probabilistic methods because evidence is recent and often preprint-level.

## 1. Executive Summary

Association rule mining (ARM) is a major classical data-mining family, but it does **not** fit the ordinary `mltreelib` leaf contract.

A terminal leaf must predict one response value for one row. Association mining produces a set of logical implications:

\[
X \rightarrow Y
\]

with support, confidence, lift, leverage, conviction, significance, and coverage. Tree root-to-leaf paths are already conjunctions of rules. Putting another rule set inside every leaf would duplicate the tree's explanatory surface and change leaf output shape.

### Recommendation

Build a separate `mltreelib-rules` capability, then connect it to trees through shared rule representations and diagnostics.

First wave:

1. **FP-growth** for frequent itemsets from row transactions.
2. **Closed and maximal itemset extraction** to control redundancy.
3. **Rule generation and quality metrics** including support, confidence, lift, leverage, conviction, Zhang's metric, and Fisher exact significance.
4. **Top-k and constrained mining** to prevent rule explosion.
5. **Rule reduction and subsumption** so users receive a compact, readable set.

Second wave:

- **CBA-style associative classifier** built from the same miner.
- **Numerical ARM** with interval-valued antecedents and consequents.
- **Hierarchical and ontology-aware item construction**, matching `mltreelib` categorical and grouped-data thesis.
- Optional **neurosymbolic or tabular-foundation-model rule extraction** as experimental backends.

Sequential pattern mining should remain a separate sequence-oriented module. It changes the input unit from row transactions to ordered sequences.

## 2. Why Association Mining Is Valuable to `mltreelib`

The package already emphasizes categorical, grouped, hierarchical, and structured signal. Association mining exposes that structure directly.

Useful outputs include:

- Frequent feature bundles.
- Interpretable `if-then` statements.
- Feature redundancy and interaction evidence.
- Categorical anomaly explanations.
- Cohort and basket patterns.
- Cross-feature dependencies missed by tree paths.
- Rules for downstream associative classifiers.

A useful example:

```text
region=West → device=desktop ∧ channel=email
support: 0.084
confidence: 0.71
lift: 3.2
leverage: 0.058
conviction: 2.9
```

These statements can explain rows, features, and segments. They must never be described as causal without additional assumptions and evidence.

## 3. Classical Algorithm Landscape

### 3.1 Apriori

Apriori repeatedly scans transactions and generates candidate itemsets. Its anti-monotonicity pruning avoids extending itemsets whose subsets were infrequent.

Strength:

- Simple and transparent.

Weakness:

- Repeated candidate generation and scans.
- Can generate large candidate sets.

Best use:

- Educational baseline and correctness oracle.

### 3.2 FP-growth

FP-growth compresses transactions into a frequent-pattern tree and mines recursively through conditional FP-trees. It avoids generating the full Apriori candidate set.

Strength:

- Usually faster than Apriori on dense transaction data.
- Natural fit for repeated row encoding and compact Rust implementation.

Weakness:

- Tree construction can still be expensive for very wide item universes.
- Rule explosion remains if many itemsets pass threshold.

Best use:

- Default frequent-itemset engine for `mltreelib-rules`.

### 3.3 Eclat

Eclat represents each item by the transaction IDs containing it. Candidate support is computed through vertical bitset intersections.

Strength:

- Often effective when many items share transaction sets.
- Bitset intersections map cleanly to Rust.

Weakness:

- Item representation can be memory-heavy for sparse, wide data.

Best use:

- Second engine and independent correctness oracle for FP-growth.

### 3.4 Closed and maximal itemsets

A closed itemset has no frequent superset with equal support. A maximal itemset has no frequent strict superset.

Mining closed or maximal itemsets removes many redundant rules without losing their support and coverage properties.

Best use:

- Default output mode for human-facing rule discovery.

### 3.5 Rule quality metrics

For rule `X → Y`:

\[
\operatorname{support}(X\cup Y)=P(X\cup Y)
\]

\[
\operatorname{confidence}(X\rightarrow Y)=P(Y\mid X)
\]

\[
\operatorname{lift}(X\rightarrow Y)=\frac{P(Y\mid X)}{P(Y)}
\]

\[
\operatorname{leverage}(X\rightarrow Y)=P(X,Y)-P(X)P(Y)
\]

\[
\operatorname{conviction}(X\rightarrow Y)
=\frac{1-P(Y\mid X)}{1-P(Y)}
\]

Also useful:

- **Zhang's metric:** association strength in `[-1, 1]`.
- **Fisher exact or G-test:** statistical significance.
- **Coverage:** fraction of rows covered by at least one rule.
- **Redundancy:** overlap among rules.
- **Objective interest and rule comprehensibility.**

Skewed data makes confidence alone misleading. A statistically significant rule with 1% support may be useless operationally. Reports should show absolute support, relative support, confidence, lift, significance, and coverage together.

## 4. Tabular ARM

Tabular rows can become transactions. Each feature contributes one item, usually the observed feature-value pair. This preserves the row as the transactional unit.

Example:

```text
row 1 = {age=adult, region=West, device=desktop}
row 2 = {age=adult, region=East, device=mobile}
```

### Challenges

1. **Rule explosion.** Many feature values create many candidate itemsets.
2. **Numeric discretization.** Numerical ARM needs interval semantics and binning decisions.
3. **One-hot width.** High-cardinality features expand the item universe.
4. **Missing values.** A missing feature is neither a valid category nor a normal absent value.
5. **Hierarchy.** A raw category and its parent category can create redundant rules.
6. **Low-data regime.** Frequent patterns can be statistically fragile.

### Recommended item model

An item should carry:

- Feature identity.
- Value or interval identity.
- Optional hierarchy path.
- Optional role: numeric, categorical, group, or derived.
- Missingness semantics.

Items from the same feature must not co-occur in one row's transaction. This invariant must be enforced during transaction construction and rule extraction.

## 5. Recent and Strong Research Directions

### 5.1 Aerial+ neurosymbolic ARM

Karabulut, Groth, and Degeler's *Neurosymbolic Association Rule Mining from Tabular Data* uses an under-complete denoising autoencoder. The model reconstructs one-hot categorical rows from corrupted inputs. Controlled probes mark an antecedent and inspect reconstructed probabilities for consequents.

Reported advantages:

- Much smaller rule sets than exhaustive methods.
- Full or near-full data coverage.
- Useful downstream behavior when rules feed rule-list classifiers such as CORELS.

Limitations:

- Neural dependency and training cost.
- Rule-extraction complexity grows polynomially with feature count.
- The paper notes weakness in small-data regimes.

Source: [arXiv:2504.19354](https://arxiv.org/abs/2504.19354), [PMLR publication](https://proceedings.mlr.press/v284/karabulut25a.html).

**Package fit:** experimental backend only. Classical FP-growth remains default.

### 5.2 Tabular foundation models as rule learners

*Tabular Foundation Models Can Learn Association Rules* introduces a model-agnostic probing framework. A conditional probabilistic model supplies antecedent plausibility and consequent probabilities. The TabProbe instantiation queries a tabular foundation model once per target feature.

Reported advantages:

- Concise rules.
- Better low-data behavior than the tested neurosymbolic baseline.
- Compatibility with downstream associative and rule-list classifiers.

Limitations:

- Repeated per-feature queries.
- External heavy model dependency.
- Current evidence covers modest data sizes.

Source: [arXiv:2602.14622](https://arxiv.org/abs/2602.14622).

**Package fit:** research comparison, not a core dependency.

### 5.3 Probabilistic and reinforced ARM

*Probabilistic and Reinforced Mining of Association Rules* introduces Gaussian-process, Bayesian, and multi-armed-bandit variants.

Value:

- More principled uncertainty than threshold-only support and confidence.
- Active or reinforced rule search under budgets.

**Package fit:** research track. Core metrics should remain classical and deterministic.

### 5.4 PyAerial

PyAerial targets association mining from tabular data and reports large speed gains over common Python ARM libraries as feature count grows. It is useful as a performance reference for transaction construction and frequent-pattern implementation.

Source: *Data Mining and Knowledge Discovery* article, [ScienceDirect record](https://www.sciencedirect.com/science/article/pii/S2352711025003073).

### 5.5 Skyline-enhanced numeric ARM

Recent numeric ARM work uses skyline concepts to discover high-quality interval rules without enumerating every candidate. This direction may fit `mltreelib`'s numerical and mixed-type leaf explanations better than pure categorical mining.

Source: [Springer article](https://link.springer.com/article/10.1007/s42979-026-05145-8).

## 6. Associative Classification

CBA, CMAR, and CPAR turn frequent rules into classifiers.

- **CBA:** orders rules by confidence and uses a covering algorithm.
- **CMAR:** uses chi-squared significance and database coverage.
- **CORELS:** searches for certifiably optimal rule lists over categorical data.

These classifiers produce ordered rule lists with default classes, not scalar leaf predictions. They belong in a separate classifier module, not the tree leaf layer.

However, `mltreelib` can use the same miner for:

- Tree-derived rule export.
- Associative classifier training.
- Leaf explanation.
- Anomaly explanation.
- Feature-group discovery.

One miner, several consumers.

## 7. Sequential Pattern Mining

Sequential mining operates on ordered events or sequences, not independent rows.

Core methods:

- GSP for Apriori-style sequence mining.
- PrefixSpan for projected pattern mining.
- Closed, maximal, and constraint-guided sequential patterns.
- Targeted SPM, where mining targets user-selected events.

Source examples:

- TaSPM: [arXiv:2202.13202](https://arxiv.org/abs/2202.13202).
- Repetitive sequential pattern survey: [IEEE TKDE](https://www.computer.org/csdl/proceedings-article/ickg/2025/668900a216/2eqaIP9Vk1a).

**Package fit:** separate `mltreelib-seq` module only when sequence input becomes a first-class package target. It does not fit a tabular leaf.

## 8. Leaf Placement Decision

### Rejected: association rule set as a terminal leaf

Reasons:

1. **Wrong output shape.** A leaf predicts one response; ARM returns many rules.
2. **Duplicate explanation.** Root-to-leaf paths already express conjunctions.
3. **Repeated mining.** Every leaf would mine its local rows independently.
4. **Poor scaling.** Rule explosion would multiply by node count.
5. **Weak objective coupling.** Split criteria and ARM quality metrics would need artificial reconciliation.

### Useful tree integration

Instead:

- Mine rules over tree partitions after fitting.
- Use association strength to explain anomalies within leaves.
- Use association rules to report feature-group interactions.
- Use rules as split diagnostics, not split criteria initially.
- Feed mined rules into a separate associative classifier.
- Optionally use hierarchical items to expose category-parent relationships.

## 9. Proposed Package Architecture

```text
mltreelib-rules/
  transaction.rs       typed row-to-transaction construction
  item.rs              feature-value and interval items
  fp_growth.rs         horizontal frequent-itemset mining
  eclat.rs             vertical frequent-itemset mining
  closed.rs            closed and maximal itemsets
  rules.rs             rule generation
  quality.rs           support, confidence, lift, leverage, conviction, significance
  reduction.rs         redundancy, subsumption, top-k selection
  hierarchy.rs         parent and child item relationships
  classifier.rs        later CBA-style associative classifier
```

Public result should expose:

- Items and antecedents/consequents.
- Quality metrics.
- Confidence intervals or significance where valid.
- Coverage and redundancy.
- Source item hierarchy.
- Deterministic ordering.

## 10. Implementation Priorities

### Tier 1 — Production

1. Typed transaction construction with hierarchy and missingness rules.
2. FP-growth.
3. Eclat as correctness and performance alternative.
4. Closed and maximal itemsets.
5. Rule metrics and significance.
6. Top-k, constrained, and redundancy-aware selection.
7. Tree integration through shared rule diagnostics.

### Tier 2 — Research and extension

1. Numerical interval ARM.
2. CBA-style associative classifier.
3. Ontology or hierarchy-aware mining.
4. Probabilistic Bayesian ARM.
5. Optional Aerial+-style neurosymbolic backend.

### Tier 3 — Separate package

1. Sequential pattern mining.
2. Foundation-model rule extraction.
3. Differentiable rule learning and differentiable ILP.

## 11. Evaluation Protocol

Report:

- Execution time by transactions, items, and average items per transaction.
- Number of frequent, closed, and maximal itemsets.
- Number of rules before and after reduction.
- Average support, confidence, lift, leverage, and significance.
- Full data coverage.
- Redundancy among rules.
- Rule length and comprehensibility.
- Downstream predictive accuracy when rules feed a classifier.
- Deterministic output for equal seed and inputs.
- Memory use.

Use both transaction benchmarks and generated tabular datasets with controlled feature interactions, hierarchy, missingness, and class labels.

## 12. Risks

1. **Calling association causal.** Association is not causation.
2. **Confidence-only ranking.** Skewed data can make high-confidence low-support rules misleading.
3. **Discretization artifacts.** Numeric rules depend on bins.
4. **Hierarchy redundancy.** Parent and child items can generate duplicate-looking rules.
5. **Missing-value encoding.** Missing must not become an ordinary category.
6. **Rule explosion.** Raw exhaustive output is not user-facing output.
7. **Multiple testing.** Many mined rules require significance and false-discovery control.
8. **Dependency creep.** Neural and foundation-model backends must remain optional.

## 13. Primary Sources

1. Agrawal, R. and Srikant, A. (1994). “Fast Algorithms for Mining Association Rules.” *VLDB*.
2. Han, J., Pei, J., and Yin, Y. (2000). “Mining Frequent Patterns without Candidate Generation.” *SIGMOD*. [DOI](https://doi.org/10.1145/342009.335032).
3. Zaki, M. J. (2000). “Fast Vertical Mining for Frequent Closed Patterns.” *TKDE*. [DOI](https://doi.org/10.1109/69.892598).
4. Liu, B., Hsu, W., and Ma, Y. (1998). “Integrating Classification and Association Rule Mining.” *KDD*. [Paper](https://cdn.aaai.org/KDD/1998/KDD98-012.pdf).
5. Angelino, N. et al. (2018). “Learning Certifiably Optimal Rule Lists for Categorical Data.” *JMLR*. [Paper](https://www.jmlr.org/papers/volume18/17-716/17-716.pdf).
6. Karabulut, E., Groth, P., and Degeler, V. (2025). “Neurosymbolic Association Rule Mining from Tabular Data.” [arXiv:2504.19354](https://arxiv.org/abs/2504.19354).
7. Karabulut, E. et al. (2026). “Tabular Foundation Models Can Learn Association Rules.” [arXiv:2602.14622](https://arxiv.org/abs/2602.14622).
8. “Probabilistic and Reinforced Mining of Association Rules.” [arXiv:2506.18155](https://arxiv.org/abs/2506.18155).
9. Luna, A. et al. (2019). “A Survey of Association Rule Mining.” [DOI](https://doi.org/10.1016/j.ins.2019.03.008).
10. Kaushik, S. et al. (2023). “Numerical Association Rule Mining: A Systematic Review.” [arXiv:2307.00662](https://arxiv.org/abs/2307.00662).
