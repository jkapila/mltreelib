# Spec 24: Association Rules Capability and Context Tree Integration

Status: proposed

## Problem Statement

`mltreelib` has a Context Anomaly Tree based on Context Tree Weighting and Probabilistic Suffix Trees. It stores conditional symbol counts by context and scores sequence transitions through information surprise. TrieKNN similarly stores categorical prefix paths with local numerical neighborhoods.

The package does not have a general association-rule capability. Users cannot mine frequent itemsets, generate association rules, quantify support, confidence, lift, leverage, conviction, significance, coverage, or redundancy, or consume those rules in an associative classifier.

Association mining must remain separate from the ordinary terminal-leaf contract. A leaf predicts one response value for one row. Association mining returns many logical implications. Root-to-leaf paths already provide conjunctions, so mining another rule set inside every leaf would duplicate the tree's explanation and multiply rule explosion by node count.

Context trees create a useful specialized opportunity. Their suffix nodes already contain conditional transition counts. These counts naturally support contextual association rules such as `history=[A,B] → next=C`. Such rules are sequential and conditional. They differ from ordinary cross-sectional FP-growth rules but share quality metrics and rule representation.

The package therefore needs two related but distinct capabilities:

1. A general cross-sectional association-rules module for tabular rows and explicit transactions.
2. A Context Tree integration that extracts contextual or sequential association rules directly from suffix-tree counts.

## Solution

Build a Rust-first association-rules capability with one shared rule representation and two mining frontends:

- **Tabular/transaction ARM:** FP-growth, Eclat, closed and maximal itemsets, rule generation, quality metrics, significance, top-k selection, and reduction.
- **Contextual ARM:** extract and rank rules from Context Tree Weighting or Context Anomaly Tree conditional counts without rebuilding a separate transaction miner.

Keep contextual rules semantically distinct from cross-sectional rules. A contextual rule conditions on an ordered suffix and predicts the next symbol. Its support is contextual occurrence, and its confidence is conditional on matching context.

Build a later CBA-style associative classifier on the same cross-sectional rule miner. Do not place rule sets inside ordinary model-tree leaves.

## User Stories

### General Association-Rules Capability

1. As a data scientist, I want to mine frequent itemsets from rows, so that I can discover recurring feature combinations.
2. As a data scientist, I want FP-growth, so that frequent-pattern mining is fast and memory-conscious.
3. As a data scientist, I want Eclat, so that I have a vertical alternative and an independent correctness oracle.
4. As a data scientist, I want closed itemsets, so that redundant frequent supersets do not flood the output.
5. As a data scientist, I want maximal itemsets, so that I can inspect compact non-extendable patterns.
6. As a data scientist, I want association rules generated from frequent itemsets, so that I can inspect antecedents and consequents.
7. As a data scientist, I want support and confidence for every rule, so that frequency and conditional reliability remain visible.
8. As a data scientist, I want lift, leverage, and conviction, so that I can detect association beyond independence.
9. As a data scientist, I want Zhang's metric, so that I can compare positive and negative association strength on one scale.
10. As a statistician, I want Fisher exact or G-test significance, so that mined rules from many candidates receive statistical control.
11. As a data scientist, I want top-k selection by a configurable quality measure, so that large result sets remain usable.
12. As a data scientist, I want constrained itemsets, so that mining can focus on selected features or values.
13. As a data scientist, I want duplicate, subsumed, and redundant rule removal, so that the final set is compact.
14. As a data scientist, I want data coverage reported, so that rule usefulness is not judged only by average confidence.
15. As a data scientist, I want numerical intervals supported, so that continuous predictors can produce meaningful interval rules.
16. As a data scientist, I want categorical hierarchy respected, so that parent and child categories do not create redundant rules.
17. As a user, I want missing values excluded from transactions rather than treated as a category, so that missingness does not become a spurious association.
18. As a user, I want deterministic output ordering, so that benchmarks and snapshots remain stable.
19. As a Rust user, I want direct access to the miner, so that I can build custom rule workflows.
20. As a Python user, I want direct access to the miner, so that I can use `mltreelib` without a tree.

### Contextual Association Rules

21. As a sequence analyst, I want rules such as `history=[A,B] → next=C`, so that context-conditioned transitions become explicit.
22. As a sequence analyst, I want contextual support and confidence, so that rule quality uses familiar measures.
23. As a sequence analyst, I want lift relative to the marginal next-symbol probability, so that a contextual rule is compared with the unconditional alternative.
24. As a sequence analyst, I want context length reported, so that short and long histories remain distinguishable.
25. As a sequence analyst, I want rules aggregated over equivalent context suffixes, so that repeated structural patterns remain readable.
26. As a sequence analyst, I want context-tree branch posterior or relevance scores included when available, so that uncertain branches can be distinguished from strong ones.
27. As an anomaly analyst, I want high-information-surprise transitions connected to their contextual rules, so that anomaly explanations become probabilistic and ordered.
28. As an anomaly analyst, I want improbable conditional transitions identified as candidate associations, so that rare-but-real behavior is not confused with globally impossible behavior.
29. As a user, I want contextual rules to include smoothing counts, so that low-count rules are not presented as certain.
30. As a user, I want contextual rules to remain distinct from cross-sectional rules, so that their support semantics cannot be confused.

### Associative Classifiers

31. As a data scientist, I want a CBA-style classifier, so that frequent rules can form a predictive model.
32. As a data scientist, I want CMAR-style significance and coverage weighting, so that classifier rules use more than confidence.
33. As a data scientist, I want a CORELS-style rule-list mode, so that short ordered lists can be optimized for categorical data.
34. As a data scientist, I want associative classifiers to share the same mined rules, so that rule generation and classification do not diverge.
35. As a data scientist, I want classifier output to expose its default class and ordered rules, so that predictions remain explainable.

### Tree and Rule Integration

36. As a tree user, I want the model tree to export root-to-leaf paths as rules, so that its existing partition logic becomes reusable rule evidence.
37. As a tree user, I want tree-derived rules and mined association rules to share one representation, so that analysis tools work across both sources.
38. As a tree user, I want mined rules to annotate leaves or splits when requested, so that local feature interactions can be inspected.
39. As a tree user, I want mined rules not to replace the tree's splits automatically, so that mining cannot silently change model semantics.
40. As a tree user, I want contextual rules extracted from the same suffix counts used for anomaly scoring, so that explanations and scores remain consistent.

## Implementation Decisions

### Decision 1: Separate Capability, Shared Rule Model

Association mining is a separate Rust module. It does not implement the ordinary terminal-leaf prediction contract.

It shares a common rule representation with tree-derived and contextual rules. A rule contains:

- Antecedent items and optional ordered context.
- Consequent item or symbol.
- Support and contextual support.
- Confidence.
- Lift, leverage, and conviction where defined.
- Zhang's metric where defined.
- Significance or confidence interval where valid.
- Source: transaction, tree path, or context tree.
- Metadata such as hierarchy, item roles, counts, and smoothing.

### Decision 2: Typed Items

An item identifies:

- Feature or field.
- Value, category, interval, or symbol.
- Optional hierarchy path.
- Role: numeric, categorical, binary, group, derived, or context.
- Missingness status.

A row transaction contains at most one item from each ordinary feature. A context transaction can contain ordered context items and one predicted next symbol.

### Decision 3: Tabular Transaction Construction

Tabular ARM requires explicit transaction semantics.

Each categorical or binary feature contributes its observed value. Each numeric feature uses either:

- A user-supplied interval specification.
- A deterministic quantile, equal-frequency, or domain binning policy.

Missing numeric and categorical values do not create ordinary items. A row with no valid items cannot support a rule transaction.

The transaction builder reports counts of rows dropped, discretized, and partially missing.

### Decision 4: FP-Growth and Eclat

FP-growth is the default horizontal miner. Eclat is the default vertical alternative and independent correctness oracle.

Both implement the same frequent-itemset interface. They report runtime, itemset count, memory estimate, and deterministic ordering.

The implementation is pure Rust. No Python ARM dependency becomes a runtime requirement.

### Decision 5: Closed and Maximal Itemsets

The rules module supports:

- All frequent itemsets.
- Frequent closed itemsets.
- Frequent maximal itemsets.

Closed itemsets are the default user-facing reduction mode because they preserve support and coverage while removing frequent supersets with equal support.

### Decision 6: Rule Generation and Quality

Rule generation accepts minimum support and minimum confidence plus optional item constraints and maximum antecedent or consequent length.

Every rule reports metrics in a shared typed structure. Metrics that do not apply are absent rather than fabricated.

Significance uses Fisher exact test for binary contingency tables or a documented G-test alternative. Multiple-testing control can report false-discovery rate or Bonferroni-adjusted values.

### Decision 7: Reduction and Ranking

Reduction removes:

- Duplicate rules.
- Rules with identical antecedent and consequent semantics.
- Subsumed rules.
- Redundant consequents under the same antecedent according to configured dominance.

Top-k selection supports support, confidence, lift, leverage, conviction, Zhang metric, significance, comprehensibility, and a composite objective.

The final result must report pre-reduction and post-reduction rule counts.

### Decision 8: Numerical and Hierarchical Rules

Numeric ARM uses intervals rather than pretending a real number is a category. Default discretization is deterministic and recorded.

Hierarchy-aware mining treats parent and child items as related. Rules that differ only by hierarchy level are marked and can be reduced or ranked separately.

### Decision 9: Contextual Rules from Context Trees

The Context Anomaly Tree and future full CTW expose their conditional counts through a read-only contextual-rule extractor.

For context $c$ and next symbol $y$:

\[
P(y\mid c)=\frac{N(c,y)+\beta}{N(c)+\beta|\Sigma|}
\]

Contextual support counts rows or transitions whose context contains $c$ and whose next symbol is $y$. Contextual confidence is $P(y\mid c)$. Contextual lift compares $P(y\mid c)$ with the marginal $P(y)$.

Rules are aggregated by exact context suffix. Longer contexts remain separate because they encode different mechanisms. A later aggregation mode may group suffixes by shared counts or tree branch identity.

The extractor reads existing counts. It does not mine a second independent itemset tree.

### Decision 10: Contextual Rule Semantics

Contextual and cross-sectional rules use the same display representation but distinct source and semantics.

- Cross-sectional support: row occurrence of antecedent and consequent.
- Contextual support: transition occurrence under a suffix context.
- Contextual confidence: next-symbol probability given context.
- Contextual lift: conditional probability divided by marginal next-symbol probability.
- Contextual rules can be ordered or sequential; cross-sectional rules are unordered.

Documentation and result types must prevent accidental comparison as if supports had identical meaning.

### Decision 11: Context Tree Anomaly Explanations

The anomaly scorer can return the best matching contextual rule or the rule set contributing to the observed surprise. A rule is eligible when its context matches the observed suffix and its predicted symbol matches or differs from the observed next symbol according to the requested explanation mode.

Two explanation modes are supported:

- `likely`: the strongest high-confidence context for the observed next symbol.
- `surprising`: the context with the lowest smoothed probability for the observed next symbol.

The explanation includes counts, smoothing, context length, probability, surprise, and rule rank. It never claims causality.

### Decision 12: Associative Classifier

The cross-sectional rule miner feeds a later associative classifier.

CBA-style classification:

1. Generate and rank rules by quality.
2. Select high-confidence, high-coverage rules in order.
3. Cover training rows.
4. Assign a default class.
5. Predict by first matching rule, with documented tie-breaking.

CMAR-style significance and coverage weighting can replace pure confidence. CORELS-style optimization remains optional and uses the external solver boundary.

### Decision 13: Tree-Derived Rule Export

The model tree exports root-to-leaf paths as rules. Each path item identifies a split condition and threshold or category partition. A path's consequence can be its leaf prediction, class, or distribution.

Tree-derived rules use the shared rule representation but a distinct source. They are not included in transaction mining by default.

### Decision 14: Optional Leaf Annotation

Mined cross-sectional rules can annotate leaves or splits when explicitly requested. Annotation must not alter fit or prediction.

A future association-aware split criterion may use rule quality, but it remains separate work with its own benchmark and cannot silently change default growth.

### Decision 15: Direct Python and Rust Access

The Python surface exposes:

- A transaction builder.
- `FPGrowth`, `Eclat`, and itemset-selection operations.
- Rule generation and reduction.
- Rule quality queries.
- A contextual-rule extractor for context trees.
- Later associative-classifier fitting.

The Rust surface exposes equivalent typed operations.

### Decision 16: Determinism and Performance

Equal input, thresholds, options, and seed produce identical itemset and rule ordering.

The implementation uses flat integer identifiers, compact transaction arrays, and reusable buffers. It avoids Python objects in the hot mining loop.

Large result sets are streamed or chunked where practical. Exhaustive generation must never hide memory use.

### Decision 17: Relationship to Context Tree Weighting

The current `ContextAnomalyTree` stores KT-smoothed conditional counts and already computes context-conditional probabilities. The contextual-rule extractor consumes that state.

Full CTW, Bayesian Context Trees, and Context Tree Switching from Spec 13 can expose richer branch posteriors and predictive probabilities. The contextual-rule interface is designed to consume those richer values without changing the rule schema.

### Decision 18: Sequential Pattern Mining Boundary

General sequential pattern mining uses ordered event sequences and distinct algorithms such as PrefixSpan. It is not part of this specification.

Contextual rules are local conditional implications extracted from an existing context tree. A future sequence-rules capability can reuse the rule representation but must keep sequential support and ordering semantics separate.

### Decision 19: Singular Task Facade and Common Report

The primary Python surface is one task facade, `mltreelib.rules.RuleModel`, with a method switch. Normal users change `kind` and shared controls rather than import separate algorithm classes.

The facade exposes stable methods such as FP-growth, Eclat, closed-itemset mining, rule generation, reduction, and associative classification. Experimental neural, probabilistic, and foundation-model mining methods live under `mltreelib.experimental.rules`.

Every rule result includes itemsets, rules, quality metrics, coverage, redundancy, optional classifier, and one common report with objective, learner, controller, and diagnostics sections. The objective section states support, confidence, lift, leverage, conviction, significance, and coverage definitions. The learner section states mining algorithm, item representation, hierarchy and discretization policy, and rule source. The controller section records thresholds, constraints, top-k selection, reduction policy, seed, and budgets. The diagnostics section records itemset and rule counts before and after reduction, data coverage, runtime, memory, determinism, and warnings.

Standard error and p-value fields appear only when valid for the selected quality metric and test. The report must not imply causal interpretation.

## Testing Decisions

- **Transaction seam:** Verify typed row construction, numeric binning, hierarchy, missing exclusion, and dropped-row diagnostics.
- **Miner equivalence seam:** On small transactions, FP-growth and Eclat must produce the same frequent itemsets.
- **Closed-itemset seam:** Verify closed and maximal itemset sets against brute-force enumeration on small data.
- **Rule seam:** Verify support, confidence, lift, leverage, conviction, Zhang metric, and significance against fixed reference calculations.
- **Reduction seam:** Verify duplicate, subsumed, redundant, and top-k outputs and report pre- and post-reduction counts.
- **Numeric seam:** Verify interval boundaries, deterministic discretization, and support under interval membership.
- **Hierarchy seam:** Verify parent-child rule distinction and configured hierarchy reduction.
- **Contextual-rule seam:** For known transition counts, verify contextual probability, support, confidence, lift, count, smoothing, and context length.
- **Context consistency seam:** Rules extracted from the context tree must use the same smoothed probabilities as anomaly scoring.
- **Context anomaly explanation seam:** `likely` and `surprising` modes must identify matching and low-probability contexts on deterministic examples.
- **Semantic distinction seam:** Cross-sectional and contextual rule reports must state their source and support definition.
- **Tree-export seam:** Exported root-to-leaf paths reproduce training membership exactly.
- **Annotation seam:** Optional rule annotation cannot change predictions or fitted tree state.
- **Associative-classifier seam:** Verify rule ordering, coverage, default class, prediction, and tie-breaking against a small reference implementation.
- **Determinism seam:** Repeated mining produces identical itemset and rule ordering.
- **Performance smoke seam:** Mine a fixed small transaction set under a tiny budget and verify result schema and memory diagnostics.
- **Prior art:** Reuse patterns from existing categorical encoder, context anomaly, trie, snapshot, and benchmark tests.

## Out of Scope

- General sequential pattern mining, GSP, and PrefixSpan.
- Neural association-rule backends as mandatory dependencies.
- Tabular foundation models as a runtime requirement.
- Differential privacy or federated association mining.
- Automatic causal interpretation of mined rules.
- Replacing model-tree splits with mined association rules.
- Installing association rule sets as terminal leaf models.
- Commercial-scale ARM platforms with GUI rule editors.
- GPU mining in the first release.

## Further Notes

- Association mining remains separate because its output shape differs fundamentally from a leaf prediction.
- Contextual ARM is a strong integration because Context Tree counts already contain the needed conditional statistics.
- The rule representation should be shared across transaction ARM, context ARM, and tree-path export even though support semantics remain distinct.
- Aerial+ and Tabular Foundation Model rule learning are experimental references, not default dependencies.
- Context-tree integration should be read-only initially. Mining cannot alter anomaly scores or fitted state.
- Research evidence is summarized in `references/research/2026-11-21-association-rule-mining-landscape.md`.
- Context-tree foundations and full CTW scope remain in Spec 13 and `references/research/2026-11-01-context-trees-and-suffix-trie-siblings.md`.
