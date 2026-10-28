# AGENTS.md — mltreelib

Guide for AI agents (and humans) working in this repository. Keep this file
accurate when the project changes; it is the shared context for every session.

## What this is

**mltreelib**: tree-based machine learning with a **Rust core** (`mltreelib-core`)
and thin **Python** (now) / **R** (planned) frontends. The research thesis: tree
and ensemble methods are under-studied relative to tabular foundation models, and
the signal hidden in **categorical, grouped, and hierarchical structure** is what
flat learners lose. We replicate the classics in Rust first (CART → C4.5 → C5.0 →
Cubist → GritBot), measure honestly (structure recovery on generated ground truth +
the TabArena-v0.1 suite in `datasets/`), then mix and extend. End goal: a
production-grade package **and a research paper**.

Design docs and research live in `references/` (start with
`references/research/tree-vs-foundation-tabular.md`,
`references/research/jkapila-methods-map.md`,
`references/research/dataset-component-design.md`).

## Layout

- `crates/mltreelib-core/` — pure Rust: `frame` (typed columns/dataset), `schema`
  (`n_category` rule), `builder` (ingest + NA policy), `index` (categorical codes),
  `transform` (summaries, grouped transforms, GroupedScaler), `timefeat` (calendar
  parts, seasons, day lags, trended transforms at time granularity), `generator`
  (adjustable benchmark data + ground truth).
- `crates/mltreelib-py/` — PyO3 bindings, exposed as `mltreelib._core`.
- `python/mltreelib/` — Python package surface (re-exports; sklearn-shaped API).
- `tests/` — Python behavior tests (pytest).
- `datasets/` — TabArena-v0.1 benchmark suite (51 parquet files + `MANIFEST.md`;
  parquet files are NOT tracked in git, re-download from the OpenML IDs listed).
- `references/` — `research/` notes, `docs/` vendor manuals, `data/` benchmark CSVs
  (tracked in git); `sources/` archives and `repos/` clones (not tracked).
- `legacy/` — previous Python/nbdev implementation. **Read-only reference oracle**
  (e.g. `legacy/mltreelib/tree/models.py` is the Python oracle for the Rust tree
  port). Do not modify or extend it.

## Commands

```sh
uv sync                      # venv + build the Rust extension (maturin backend)
cargo test                   # Rust tests (crates/mltreelib-core/tests/)
uv run maturin develop       # rebuild the extension after Rust changes
uv run pytest tests/ -q      # Python behavior tests
uv run zensical build        # docs site (docs/ -> site/)
uv run python scripts/gen_api_docs.py   # regenerate docs/reference/api.md after API changes
```

## Rules

- **No scikit-learn dependency** (the API shape `fit`/`transform`/`predict` is
  fine, importing sklearn is not). Otherwise be pragmatic — established core
  packages over hand-rolling: `pyarrow` for parquet in the Python layer;
  arrow-rs (`arrow`/`parquet`) or Polars in Rust when frames/IO grow beyond the
  in-house `frame` module. Keep `mltreelib-core` dependency-lean, but do not
  reinvent wheels.
- **Rust core is the source of truth.** Python/R are thin shells. Semantics and
  algorithms live in `mltreelib-core` and are tested there first.
- **Missing data is first-class.** `None` **and NaN** = missing (NaN is
  normalized at ingestion); missing in → missing out, never panic. `summarize`
  reports `n_missing` by default; imputation is opt-in (`NaPolicy::Impute`),
  never implicit; time-series transforms drop incomplete rows and the Python
  layer raises a `UserWarning` with the count. Unseen categorical levels map to
  missing. Degenerate scalings produce missing, not inf.
- **Deterministic**: generators/transformations take explicit seeds; no wall-clock
  or unordered iteration in outputs.
- **Tests are behavior, not wiring.** Every new capability lands with Rust tests
  (core) and pytest coverage of the user-visible API. Run both suites before
  committing. No test may pin incidental implementation details.
- **Commits**: small, imperative subject + short body listing the behavior added
  and the test counts. Binary data (`datasets/*.parquet`, wheels, models) is not
  committed.
- **Research compounds.** Every research finding, survey or external analysis
  is saved to `references/research/` as a dated, cited markdown note (companion to the
  chat answer, never instead of it). New session? Check `references/research/` first.
- **Roadmap lives in README.md** ("Roadmap / TODO"). Update its checkboxes when
  items land; add future work there rather than building speculative features.
  Currently parked for later on purpose:
  featuretools-style automated features (keep the tool surface in one package).

## Research context

- Benchmarks: TabArena leaderboard CSVs in `references/data/` and analysis in
  `references/research/`; datasets in
  `datasets/` (OpenML suite 457). BeyondArena needs the `data-foundry` package —
  intentionally not vendored.
- Comparison discipline: give trees the same tuning/ensembling budget as any
  challenger; track completion/missing-run rates, not just accuracy.
- The generator's `truth` manifest reports realized variance shares per component;
  use it to score **structure recovery**, not only accuracy.

## Using skills

Prefer the available agent skills for artifact work: research skills for sourcing,
`hallmark` for designed pages, `docx`/`pptx`/`pdf` for documents, `xlsx` for
spreadsheets — rather than hand-rolling formats. Code, tests, and package
structure follow the rules above regardless.
