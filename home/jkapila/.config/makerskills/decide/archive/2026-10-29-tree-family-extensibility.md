# Decision: tree family extensibility — naming, family scope, Python-vs-Rust seam

**Date:** 2026-10-29
**Decide by:** open
**Reversibility:** medium (API shape; the split loop is easy to keep, hard to un-promise)
**Stakes:** medium

## Context
ModelTree v1 is built and rpart-verified. Jitin asked: (1) is the forest named
ModelForest, (2) what single-tree variants can the family grow, (3) should the
tree become a Python class to define growth/selection behavior ("at least for
me to try") or stay embedded in Rust. He leans Python-class ("at least for me
to try").

## Questions

### First instinct (Q9, Jitin's)
"I was also thinking to make the trees a python class so that we can define
the growth and selection behaviour, atleast for me to try ... or we keep it
embedded inside rust?"

### Does this decision need to be made? (Q1)
Yes — the next builds (forest, new leaf kinds, preselect layer) all depend on
where the seam lives.

### A year from now, how will we feel? (Q3)
A Rust inner loop with a Python policy surface ages well (it is how
LightGBM/CatBoost themselves are shaped); a Python growth loop would have to
be rewritten the first time we benchmark.

### What does saying yes displace? (Q39)
Python-definable *growth* would displace the speed story we just benchmarked
against LightGBM — and per-candidate-split callbacks are ~10⁶ GIL crossings
per fit.

## Decision
**Keep growth and the split loop in Rust behind trait seams
(`SplitCriterion`, `LeafModel`); give Python the full policy surface: config
knobs, the `ModelForest`/`ModelTree` classes, and — the tinkerer's door —
`preselect` hooks as Python callables at per-column granularity (cheap:
d calls per node, not d×n). Python-callback split criteria only when a real
experiment demands the bridge.**

**Rejected alternative:** Python-subclassable growth (`class MyTree(ModelTree):
def split_score(...)`) — maximum flexibility, but slow, hard to test
deterministically, and an API we'd lock in before any experiment needs it.

**Accepted downside:** a brand-new *criterion* prototype requires a Rust impl
(≈ a day) before Python can play with it; criterion experiments at Python
speed happen only at the selection/leaf level.

Sub-calls recorded with this decision:
- **Forest naming: `ModelForest`** — yes; wraps `ModelTreeParams` so every
  leaf/split variant inherits into forests for free (mean-leaf forests =
  RandomForest flavor, linear-leaf forests = LinearForest flavor, one class).
- **Family scope:** the family is exactly three seams — split criterion,
  leaf model, predict/output policy. Next additions in order: quantile
  leaves, classification/logistic leaves (the thesis's logistic option), M5
  leaf smoothing, Cubist-style extrapolation clamp, rule-set printout.

## Rationale
The behaviors Jitin wants to steer — *selection* (which variables compete)
and *leaf behavior* — live at coarse granularity where Python callbacks are
affordable; the split inner loop is exactly where they are not. This keeps
"for me to try" real (Python-defined selection + leaf policies + orchestration
subclasses) without paying the speed and testability cost in the inner loop.

## Weak spots
none — Jitin's lean toward a Python growth class is recorded verbatim; the
call deliberately redirects it to selection/leaf-level hooks.

## Expected outcome
By 2026-11-28: at least one experiment has used a Python-defined preselect
hook without touching Rust; `ModelForest` exists with the same param surface;
no growth-path Python callback has been needed.

## Revisit
**2026-11-28** — did any criterion experiment actually need the
Python-callback bridge? If yes, the seam was wrong; if no, the granularity
call was right.

## Source
Questions adapted from [The 37signals Guide to Making Decisions](https://37signals.com/how-we-make-decisions)
