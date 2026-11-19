# TODO: docs depth pass (parked 2026-11-18, pick up later)

Correctness audit: `.scratch/docs-audit-2026-11-18.md` (README API names, version 0.4.5 vs 0.4.8, pyarrow dependency, docs/agents/ published, stub pages, changelog, contributing, citation and scope pages).

Depth problem (Jitin, 2026-11-18): guide and example pages are thin and hard to use. Code blocks use undefined `X`, `times` and `events`. Pages paste script output. They have no evaluation (C-index/Brier, held-out split), no interpretation, no pitfalls, and no comparison with lifelines, scikit-survival or EconML.

Plan:
- [ ] Score all 38 guide and example pages against the template: when to use / vs others, load real data, fit + tune, evaluate held-out, interpret, pitfalls + limits. All code must run top to bottom.
- [ ] Rewrite one flagship page (survival on FLCHAIN) as the style model. Jitin approves it.
- [ ] Next flagship pages: uplift, anomaly.
- [ ] CI job that runs the README and docs code blocks.
- [ ] Scope: tree-based models first. RDD, panel and IV are on the roadmap and not implemented.
