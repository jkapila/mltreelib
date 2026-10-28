# 11 — Sequential-retrieval trees & temporal decay

The retrieval-side counterpart of the context trees in `10`: industrial
"tree-based" recommendation — hierarchical item trees for O(log N)
retrieval, session/expert hybrids, and dynamic time-decay models for concept
drift. Source material: `references/research/tree-decay-sequential-rec-survey.md`
(Jitin's literature review, 2026-10-28); the entries below ground the
surveyed papers in our usual Source · Idea · Pseudocode · Knobs · Note form.

## TDM — Tree-based Deep Match (Zhu et al., KDD 2018)

**Source:** "Learning Tree-based Deep Model for Recommender Systems",
KDD 2018, pp. 1079–1088 (Alibaba).
**Idea:** cluster the whole item corpus into a hierarchy — leaves = items,
internal nodes = generalized latent concepts — and turn retrieval into
top-down **beam search** over learned layer-wise scores: O(N) becomes
O(log N).

```text
build: hierarchical clustering of item embeddings (hierarchical K-Means /
       balanced min-cut); depth L, fanout per node
score: for each level l:  P(n^(l) | u) = sigmoid(W^(l) · [e(n^(l)) ‖ f(H_u)] + b^(l))
       e(n) = node embedding; f(H_u) = user's sequential history vector
retrieve: beam search top-down — keep best-k nodes per level, expand only
       their children; recommend argmax over surviving leaves
train: next-item classification at every level (internal nodes as
       intermediate targets)
```

**Knobs:** beam width, depth/fanout, embedding dim, tree clustering method.
**Note:** the landmark "retrieval is tree search" paper; hierarchy quality is
a first-class optimization target (see its companion, "Learning optimal tree
structures", arXiv:1904.07548, mutual-information objective).

## TDR — Deep Tree-based Retrieval (Liu et al., 2024)

**Source:** "Deep Tree-based Retrieval for Efficient Recommendation: Theory
and Method" (arXiv:2408.11345).
**Idea:** fix TDM's unsafe pruning with a **max-heap assumption**: a parent's
score must bound the maximum of its children's scores, so pruning a parent
safely discards its whole subtree. Retrieval becomes level-wise
multi-classification with sampled-softmax — the offline loss matches the
online beam-search pattern.

```text
constraint: score(parent) >= max_{c in children} score(c)   # max-heap
objective: sampled-softmax multi-classification per tree level
retrieve: heap-pruned beam search — discard subtrees whose bound falls
          below the current k-th best
```

**Knobs:** heap-bound enforcement (projection vs penalty), sampled-softmax
negatives, beam width.
**Note:** directly answers the survey's "error propagation in top-down beam
pruning" gap — the theory piece to read for non-greedy routing.

## CHAMELEON (Moreira, Jannach, da Cunha 2019)

**Source:** "Contextual Hybrid Session-based News Recommendation with
Recurrent Neural Networks", IEEE Access 7, 2019.
**Idea:** meta-architecture stacking per-request **feature meta-modules**
(each its own NN — sequential session features via RNN/GRU, plus
non-sequential context/popularity features) into a feedforward ranker —
the neural descendant of Garcin's expert mixture (file 10).

```text
for each request:
    f_seq   = RNN/GRU over the session's item sequence
    f_ctx   = NN over contextual/feature metadata
    f_misc  = NN over popularity/recency features
    score(x) = ranker([f_seq ‖ f_ctx ‖ f_misc])
```

**Note:** "dynamic features" computed per request are what let static models
track drift; the hybrid-module pattern is the recipe for our
encode-then-model story.

## Dynamic time-decay CF (Chen, Hui, Thai et al., J. Supercomputing 2020)

**Source:** "A collaborative filtering recommendation system with dynamic
time decay", The Journal of Supercomputing, 2020 (DOI 10.1007/s11227-020-03266-2).
(Attribution caveat: the survey's reference list credits "Cao/Lin/Ji,
IEEE Access 8, 2020" — unverified; see the survey's verification header.)
**Idea:** weight each interaction by recency, with the **decay rate itself
adaptive** — the survey's generalized form:

```text
W(i, Δt) = 1 / (1 + λ(u,c) · Δt^γ)          # heavy-tailed decay (γ tunes the tail)
λ(u,c)   = λ0 · exp(-σ · S_active/S_total) · (1 + Drift(u,c))
       # engagement ratio and measured concept drift modulate the drop rate:
       # frequent context switches -> λ up (forget fast); focused sessions -> λ down
CF is then computed over decayed interactions (similarity/rating estimates
weight each observation by W).
```

**Knobs:** λ0, γ (decay shape), σ (engagement sensitivity), drift detector.
**Note:** the "cognitive" framing (Ebbinghaus forgetting) is the survey's
lens; for our purposes this is the spec for **time-aware weights** — the
same ingredient as the windowed features in our feature layer, at the
interaction level.

## TDTL — Trust Decay-based Temporal Learning (Heryawan et al., IEEE Access 2025)

**Source:** "Trust Decay-Based Temporal Learning for Dynamic Recommender
Systems with Concept Drift Adaptation", IEEE Access 13, 2025 (IEEE Xplore
11050419).
**Idea (verified):** a temporal-learning framework integrating **trust-region
optimization** with decay-based temporal weighting to adapt to concept drift
without overfitting transient spikes. The trust region constrains each update
step; decay discounts stale interactions. [INFERENCE: the trust region is what
stabilizes the drift adaptation — the abstract does not detail the decay
family or the drift detector.]

**Knobs (from title/abstract):** trust-region radius, decay schedule, drift
adaptation rate.
**Note:** the newest of the line (2025, already cited); connects constrained
optimization (RL-style trust regions) to temporal recommendation — worth
reading for our *retraining-under-drift* story (cf. `tree-vs-foundation`
section 6).

## Forgetting-curve recommendation (Jin et al., CIKM 2023)

**Source:** "Pareto-based Multi-Objective Recommender System with forgetting
curve" (CIKM).
**Idea:** Ebbinghaus retention curves as per-item **temporal masks** over
history, optimized under competing objectives (accuracy vs novelty/freshness)
with Pareto methods.

## Research gaps from the survey (open tree problems)

1. **Beam-pruning error propagation** — misrouting at high nodes kills whole
   subtrees; want non-greedy/backtracking routing with log-time bounds.
2. **Structural rigidity of static trees** — streaming item pools need online
   split/merge/rebalance of the hierarchy concurrent with gradient updates
   (news, fast fashion).
3. **Popularity ≠ recency** — decay models conflate global viral spikes with
   individual freshness; want causal decay separating network effects from
   personal memory decay. (Directly relevant to our `m_is_novel`
   novelty-isolation design.)
4. **Hyperbolic tree geometry** — hierarchies are exponentially branching;
   Euclidean embeddings distort deep leaves; native Poincaré/Lorentz decay
   models are largely unexplored.

## Why this family matters to us

This is where trees meet industrial scale (retrieval) and time (drift). The
survey's gaps are genuine tree research problems — the material for our
"more tree-based problems" list; TDM/TDR give the retrieval-tree baseline,
CHAMELEON and the decay line give the temporal-hybrid baseline. Our own
angles connect: `windowed_transform`/`lag_days` features are the tabular
flavor of temporal decay, and the hierarchy-aware categorical encoding is
the tabular flavor of tree routing.
