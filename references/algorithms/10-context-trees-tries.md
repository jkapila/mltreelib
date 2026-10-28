# 10 — Context trees and the trie family (sequence trees + the data structure)

The "ctree" you meant is the **context tree** lineage — variable-memory
models over symbol histories. Every context tree is literally a **trie of
suffixes** (the structure behind TrieKNN), which is why this file holds both.

## Context Tree Weighting (CTW) — Willems, Shtarkov, Tjalkens 1995

**Source:** the 1995 IT paper (the "celebrated CTW" of compression;
expositions: Sunehag & Hutter 2012; Kontoyiannis arXiv:2211.02676).
**Idea:** Bayesian mixture over **all** variable-order Markov models up to
depth D at once, computed in one pass: each node carries a KT (Krichevsky–
Trofimov) estimator and a mixing weight β (½ per level); predictions mix
every context tree's estimate with posterior weights. Redundancy is within a
constant of the best single context tree — no model selection needed.

```text
init: for every node θ in the depth-D suffix trie: counts (α0, α1) = (1/2, 1/2),
      β_θ = 1/2
predict x_{t+1} given history x_{t-D+1..t}:
    walk the path of the last D symbols (contexts θ_d for each depth d ≤ D)
    P̂_d = KT estimate at the depth-d node
    P̂_CTW = Σ_d (Π weights β along deeper nodes) * P̂_d     # weighted mixture
update: increment counts along the path for the observed x_{t+1}
```

**Knobs:** depth D, alphabet size, bias (β updates for biased sources).
**Note:** the sequential-prediction/compression classic; a principled way to
mix "how much history matters" — directly relevant to our temporal features.

## Bayesian Context Trees (BCT) — Kontoyiannis et al.

**Source:** arXiv:2007.14900 ("Bayesian Context Trees: Modelling and exact
inference for discrete time series", R package `BCT`); Bayesian Analysis
paper "Posterior Representations for Bayesian Context Trees"; survey
arXiv:2211.02676.
**Idea:** variable-memory Markov chains where the **context tree itself is
random**; exact Bayesian inference over tree space by sum-product recursion
up the suffix trie — posterior over contexts, model order, and predictions,
without MCMC.

```text
model: P(x_t | context = the leaf of T matching the recent past)
prior over trees T (e.g. branching probability)
likelihood: product over t of leaf probabilities (Dirichlet-multinomial)
inference:
    upward recursion over the suffix trie:
        each subtree's marginal = Σ over its children's marginals × prior
    → P(T | data) summarized as posterior context/branch probabilities
predict: posterior predictive = Σ_T P(x_{t+1} | T) P(T | data)
```

**Knobs:** tree depth, prior on branching, Dirichlet concentration.
**Note:** the gold standard for "which history matters, with uncertainty";
its posterior-over-subtrees is the Bayesian analogue of our structure
recovery.

## Parsimonious Context Trees (PCT) — Eggeling, Karalič, Koivisto

**Source:** "Algorithms for learning parsimonious context trees" (Eggeling,
Tübingen thesis/repository) + Eggeling et al., "Dealing with small data: On
the generalization of context trees" (PMLR v37, 2015); Bayesian approximate
learning: Ghani, Heard, Sanna Passino, "Approximate learning of parsimonious
Bayesian context trees" (arXiv:2407.19236; Statistics & Computing 2026).
**Idea:** context trees extended with **context-specific parameter sharing**:
different contexts may use the same conditional distribution, so the model is
parsimonious (fewer parameters than fixed-order Markov) and works on small
data. Learned by penalized likelihood (AIC/BIC/MDL-style) or, in the Bayesian
version, by approximate inference with conjugate priors.

```text
model: context tree T where each leaf is labeled with a distribution;
      several leaves may share one distribution (parsimony)
learn:
    score(T, assignment) = Σ_contexts log P(data_context | shared params)
                           - penalty(# distinct parameter sets)
    search: greedy splits/merges over contexts (or Bayesian: variational /
            MCMC over tree + sharing structure, approximate learning 2024)
predict: the shared distribution of the leaf context reached by history
```

**Knobs:** max depth, penalty (AIC/BIC/MDL), minimum context support,
branching prior (Bayesian version).
**Note:** "parsimonious" = the compression-minded answer to overfitting
contexts; the small-data generalization paper is the one to read for our
low-N experiments.

## The trie family (the data structure behind all of the above)

**Idea:** prefix tree — nodes = symbol prefixes; children indexed by next
symbol. Context trees store **suffixes of the recent past** in exactly this
shape; TrieKNN (see `references/research/jkapila-methods-map.md`) stores
categorical combinations the same way.

```text
insert(word):  node = root; for s in word: node = node.child[s] (create if absent)
               node.count += 1; node.payload = ...
search(word):  walk; return node.payload / count
prefixes:      every node is a prefix; counts along the path give
               per-prefix statistics (TrieKNN keeps class counts + leaf KNNs)
variants:
    radix/patricia trie: compress single-child chains -> O(|word|) nodes less
    suffix tree: trie of ALL suffixes of S, edges as substrings
                 (Ukkonen: O(n) build); suffix links enable linear scans
    ternary search tree: char-indexed BST hybrid for memory/dynamic keys
    context tree: trie of history suffixes up to depth D (CTW/BCT above)
```

**Notes for our program:**
- TrieKNN = categorical path through the trie + local model at the leaf —
  the same "leaf-local models" idea as model trees, on a categorical trie.
- CatBoost's cat×cat combinations are effectively trie paths over category
  sequences (the combination feature = the path code).
- For the tree build: a trie of category paths is the natural index for
  group/hierarchy features (our `hier_*` columns are paths already).

## Applied: news recommendation with context trees (Garcin, Dimitrakakis, Faltings)

**Source:** "Personalized News Recommendation with Context Trees",
**RecSys 2013**, pp. 105–112 (arXiv:1303.0665, EPFL). This is the paper the
news hunt was tracing.
**Idea:** the browsing **steps** are the sequence; a context tree (VMM)
partitions browsing histories; every node holds an **expert**; the active
path (root→leaf) mixes experts by stop-probability weights, and each expert
mixes three local models — **standard** (Dirichlet-multinomial over items
read while active), **popularity** (items in the recent-read set), and
**freshness** (breaking items about to be published). Every recommendation
is this mixture: popularity + freshness + user steps, exactly the hybrid.

```text
tree: nodes = contexts (suffixes of the read sequence, or topic
      distributions in a k-d tree); branches grow with new articles and
      are removed as articles leave the current pool (dynamic pool)
expert i at node: P_i(x | history) = p_std*P_std + p_pop*P_pop + p_fresh*P_fresh
    P_std   Dirichlet-multinomial on reads of x while expert i was active
    P_pop   total clicks of x among the recent-read popular set P
    P_fresh uniform over the |F| about-to-appear items (breaking news)
    p_tau   Dirichlet priors or Bayesian updates from the expert's accuracy
recommend(history):
    q = P_root; for k along root->leaf:  q = w_k*P_k + (1-w_k)*q
        (w_k = usefulness of expert k, w_root = 1; specific experts
         override general ones with probability w_k)
    recommend argmax_x q(x) over the candidate pool (popular + fresh)
learn(item x read):
    for each active expert k:  w_k <- w_k * P_k(x) / q_k(x)   # Bayes
    update each expert's local model counts
```

**Knobs:** sequence vs topic-distribution tree, context depth, |P| popular
window, |F| fresh window, expert weight prior.
**Note:** evaluated with an unbiased online methodology; strong accuracy +
novelty in the drifting/cold-start news domain. For our agenda this is proof
that *context-tree mixing of simple experts* beats monolithic models under
concept drift — and a ready host problem (session-sequence recommendation)
beside regression/classification in our benchmarking story.

## Why this family matters to us

Context trees answer "how much history is relevant, and how sure are we" —
the sequence analogue of our structure-recovery question. PCT's
parameter-sharing is the parsimony lesson for our leaf models; CTW's mixing
is a ready-made recipe for **model combination without selection** (echoing
Cubist committees from a Bayesian direction).
