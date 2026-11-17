# Spec 13: Context Tree Weighting (CTW), Suffix Tries & Deep Retrieval

## Problem Statement

Standard decision trees partition cross-sectional feature vectors $X \in \mathbb{R}^d$ for static prediction. However, sequential streams (clickstreams, user behavior logs, text/code token sequences, sensor event logs, financial tick logs) require modeling variable-length historical context strings $s = (x_{t-D}, \dots, x_{t-1}) \in \Sigma^D$.

Existing sequence tools suffer from severe tradeoffs:
1. **Fixed-Order Markov Chains**: Underfit when context horizon $D$ is too short; encounter exponential parameter explosion $|\Sigma|^D$ and catastrophic overfitting when $D$ is large.
2. **Standard GBDTs on Lag Features**: Inefficiently evaluate lag columns independently, failing to exploit the suffix-tree hierarchy of sequential context and requiring massive manual lag engineering.
3. **RNN / Transformer Sequence Models**: Heavyweight, uncalibrated for Bayesian uncertainty, slow for ultra-low-latency real-time scoring, and impractical for streaming micro-prediction or local edge deployment.
4. **Extreme Classification / Retrieval**: Recommending from millions of items ($N > 10^7$) via linear dot-product scans violates real-time SLAs (<10ms).

## Solution

Build a pure-Rust, zero-dependency sequential tree engine in `mltreelib-core` bridging information-theoretic context trees with industrial hierarchical retrieval:

1. **Context Tree Weighting (CTW)**:
   - Implement Willems, Shtarkov, and Tjalkens (1995) exact Bayesian model averaging over all $2^{|\Sigma|^D}$ sub-trees simultaneously in $O(D)$ time and memory per symbol.
   - Utilize Krichevsky-Trofimov (KT) sequential probability estimators at every suffix node:
     $$P_e(a, b) = \frac{\prod_{i=0}^{a-1} (i + 1/2) \prod_{j=0}^{b-1} (j + 1/2)}{\prod_{k=0}^{a+b-1} (k + 1)}$$
   - Recursive weight mixing at node $s$: $P_w^s = \frac{1}{2} P_e^s + \frac{1}{2} \prod_{\sigma \in \Sigma} P_w^{s\sigma}$.
2. **Context Tree Switching (CTS)**:
   - Implement Veness et al. (2012) non-stationary extension, allowing the active generating context tree to switch arbitrarily over time with switching rate parameter $\alpha \in [0, 1]$.
3. **Bayesian Context Trees (BCT) & MAP Pruning**:
   - Implement Kontoyiannis et al. (2020, 2022) exact MAP tree discovery, computing posterior probabilities for individual context branches $P(s \in T \mid \mathbf{x})$ without MCMC sampling.
4. **Parsimonious Context Trees (PCT)**:
   - Implement Eggeling et al. (2015) parameter sharing across statistically indistinguishable context leaves, avoiding leaf parameter starvation in sparse alphabet regimes.
5. **Hierarchical Deep Retrieval Trees (TDM / TDR)**:
   - Implement Tree-based Deep Match (TDM, Zhu et al. 2018) and max-heap bounded Deep Tree-based Retrieval (TDR, Liu et al. 2024) hierarchical clustering trees, supporting $O(K \log N)$ top-$K$ beam retrieval over high-cardinality item sets.

## User Stories

1. As a quantitative researcher, I want `ContextTreeWeighting(depth=16, alphabet_size=2)`, so that I can compute exact Bayesian sequential predictive distributions without choosing a single fixed Markov order.
2. As a streaming fraud detection engineer, I want `cts.update(symbol)`, so that the context tree can dynamically detect concept drift and switch between behavioral regimes in real time.
3. As a sequential recommendation modeler, I want `tree.predict_distribution(history)`, so that I receive full probability distributions over the next action symbol conditioned on variable-length user history.
4. As an econometrician, I want `bct.map_tree()`, so that I can extract the single Maximum A Posteriori minimal context tree and discover the exact causal memory depth of an empirical process.
5. As a bioinformatician analyzing DNA/protein sequences, I want `ContextTree(alphabet_size=4)` with multi-symbol KT estimators (Dirichlet-1/2 prior), so that I can model categorical genomic alphabets natively.
6. As a search ranking engineer, I want `TreeRetrievalIndex(n_items=10_000_000)` with top-down beam search, so that I can retrieve the top 100 candidate items in under 2 milliseconds.
7. As an ML engineer, I want `tdr.enforce_max_heap()`, so that hierarchical branch-and-bound pruning provably guarantees no high-scoring leaf item is erroneously pruned.
8. As a micro-prediction agent, I want `ctw.log_loss(stream)` and sequential redundancy bounds, so that I can benchmark compression ratios and Kolmogorov complexity against theoretical limits.
9. As an applied data scientist, I want `pct.cluster_contexts()`, so that rarely visited suffixes share transition statistics with related historical contexts, preventing small-sample noise.
10. As a production engineer, I want `ctw.save_binary()` and zero-copy deserialization in Rust, so that massive sequence models can be persisted and loaded instantly without Python overhead.
11. As a systems developer, I want suffix trie nodes to use memory-efficient array representations (flat arenas) rather than pointer-heavy heap structures, minimizing cache misses during sequence traversal.
12. As a tabular modeler, I want to convert continuous numeric series into discrete sequence tokens via unsupervised quantile binning, enabling CTW to model continuous sensor feeds.

## Implementation Decisions

- **Crate Module Structure**:
  - Add module `crates/mltreelib-core/src/context/`:
    - `trie.rs`: Flat memory arena storing `ContextNode` structs. Suffix indexing via packed symbol arrays.
    - `kt.rs`: Numerically stable Krichevsky-Trofimov sequential estimator using log-gamma recurrence.
    - `ctw.rs`: Core `ContextTreeWeighting` implementation with recursive Bayesian weight mixing.
    - `cts.rs`: `ContextTreeSwitching` maintaining transition probabilities and switching state mixture.
    - `bct.rs`: `BayesianContextTree` with sum-product MAP tree extraction and branch posterior evaluation.
    - `retrieval.rs`: `TreeRetrievalIndex` implementing hierarchical k-means tree construction, top-down beam search, and max-heap scoring invariants.
- **Node Memory Layout**:
  - Each `ContextNode` in the arena holds:
    - `counts: [u32; ALPHABET_SIZE]` (or dynamically sized slice for multi-symbol alphabets).
    - `log_pe: f64` (log of local KT probability).
    - `log_pw: f64` (log of weighted mixture probability).
    - `child_indices: [u32; ALPHABET_SIZE]` (arena index references, 0 = null).
- **Python Surface (`crates/mltreelib-py`)**:
  - Expose `ContextTreeWeighting`, `ContextTreeSwitching`, `BayesianContextTree`, and `TreeRetrievalIndex` directly in `mltreelib.context`.
  - Methods: `fit(sequences)`, `update(symbol)`, `predict_proba(context)`, `log_loss(sequence)`, `extract_map_tree()`, `retrieve_top_k(query_vector, k, beam_width)`.

## Testing Decisions

- **Theoretical Optimality Tests**:
  - Synthetic Markov Chain ($D=2$): Stream data from a known order-2 Markov chain. Verify CTW achieves average log-loss within $O(\frac{\log T}{T})$ of the true entropy rate.
- **Concept Drift & Switching Test**:
  - Generate Bernoulli sequence with $P(1)=0.1$ for $t \in [1, 2000]$ and $P(1)=0.9$ for $t \in [2001, 4000]$.
  - Verify `ContextTreeSwitching` rapidly shifts predictive probability within 50 symbols of change point, while stationary CTW requires hundreds of symbols.
- **MAP Tree Exact Recovery**:
  - Generate sequence from explicit tree structure: context `[0]` predicts $0.8$, context `[1, 0]` predicts $0.2$, context `[1, 1]` predicts $0.5$.
  - Verify `BayesianContextTree.extract_map_tree()` recovers the exact 3-context topology.
- **High-Cardinality Retrieval Benchmark**:
  - Cluster 100,000 synthetic 64-dimensional vectors into a depth-10 binary tree.
  - Verify `retrieve_top_k(query, k=10, beam_width=20)` achieves $\ge 95\%$ recall against brute-force exact nearest neighbor search with $>50\times$ speedup.

## Out of Scope

- Direct end-to-end backpropagation through deep user-encoder neural networks (embeddings are provided as input arrays).
- Infinite alphabet continuous context trees (continuous series must be pre-discretized into discrete tokens).

## Further Notes

- CTW has a deep historical relationship to compression: optimal CTW log-loss is identical to arithmetic coding compression length. This provides an exact connection between minimum description length (MDL) and sequential generalization.
