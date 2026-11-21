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
- `references/` — `research/` notes, `algorithms/` pseudocode set (the learn-everything
  reference), `docs/` vendor manuals, `data/` benchmark CSVs
  (tracked in git); `sources/` archives (+ `sources/extracted/`) and `repos/` clones
  (not tracked).
- `legacy/` — previous Python/nbdev implementation. **Read-only reference oracle**
  (e.g. `legacy/mltreelib/tree/models.py` is the Python oracle for the Rust tree
  port). Do not modify or extend it.

## Commands

```sh
uv sync                                    # venv + build the Rust extension (maturin backend)
cargo test                                 # Rust tests (crates/mltreelib-core/tests/)
uv run maturin develop --release           # rebuild the extension with release optimizations
uv run pytest tests/ -q                    # Python behavior tests
uv run python <script.py>                  # run scripts/benchmarks via the venv Python
uv run python -u examples/elo_score_all.py # run unbuffered Elo arena benchmarks
uv run zensical build                      # docs site (docs/ -> site/)
uv run python examples/gen_api_doc.py      # regenerate docs/reference/api/*.md after API changes
```

> **Execution Note**: Use `uv run python ...` for repository scripts to load the editable package and its dependencies. Build the extension with `--release` for benchmarks.



## Writing Modes

### Mode A — Publication / Human Voice
Use when drafting articles, blog posts, social copy, intros, or any text
meant for a human audience to enjoy reading.

**Rhythm**
- Vary sentence length deliberately (short, long, medium).
- Vary paragraph length. Let one be a single line. Let the next be dense.
- Scan only the first two words of each sentence in a row. If "The",
  "It is", or "This" repeats more than once in four sentences, rewrite.
  Mix in a question, a subordinate-clause opener, or a bare fragment.

**Voice & Register**
- Use contractions (it's, don't, you'll) and conversational transitions.
- Shift register at least once per section: a technical sentence,
  then a wry aside, then back.
- Inject one clear point of view per paragraph.
- Drop in a parenthetical or one-line tangent and come back.
  Let a sentence redirect itself: "Actually, no — it's more like…"

**Structure**
- Open with a specific detail, anecdote, or opinion — not a definition.
- Break the triplet: use two or four items in a list, not three.
  If a list has three, merge or split it.
- Go one level deeper on one point instead of three flat parallels:
  claim → reasoning → limitation, or rule → conditional exception.
- End on a new observation, question, or concrete next step.
  Never restate the intro in a summary paragraph.

**Precision**
- Use specific numbers over round ones: "4 years and 3 months" not
  "about five years." "37 of 40 users" not "most users."
- Kill hedging stacks: delete "very, really, basically, extremely,
  quite, somewhat." One qualifier max per sentence, usually zero.

**Cuts**
- Remove: "Furthermore", "In conclusion", "It is worth noting",
  "It's important to", "delve", "leverage", "landscape", "game-changer".
- Prefer plain, concrete nouns over abstract compounds.
- Read-aloud test: if a sentence sounds like it was generated, rewrite it.

**Fabrication guard**
- Where a personal story, sensory detail, or specific number belongs,
  insert `[YOUR STORY HERE]` instead of inventing one.
  The human fills those in.


### Mode B — Technical / Agent-Facing (80% ASD-STE100)
Use for explanations, summaries, code reviews, instructions, error messages,
tool descriptions, inter-agent text, and any text another system or a
non-native reader will parse.

**Sentence rules**
- Instructions / procedures: max 20 words per sentence.
- Descriptions / explanations: max 25 words per sentence.
- One instruction per sentence. Split compound actions.
- One topic per paragraph; max 6 sentences per paragraph.

**Grammar rules**
- Active voice. Name the actor.
- Simple tenses only: present, past, future, infinitive, imperative.
  No perfect tenses ("has done" → "did").
- No contractions.
- No semicolons.
- Condition before command: "If the flag is set, run X."

**Word rules**
- One word, one meaning, one part of speech.
  - "close" = shut (verb). Use "near" for the adjective.
  - "make sure" not "ensure".
  - "check" not "verify / confirm / validate".
  - "use" not "utilize".
- No hedging: replace "should / would / may / might" with
  "must" (requirement) or "can" (option).
- Multi-word nouns: max 3 words.
- Keep articles (a, an, the) and "that". No telegraph-style fragments.
- Pick one verb for one action and reuse it; do not rotate synonyms.

**Scope**
- Keep product names, code identifiers, and domain terms verbatim.
  Do not "simplify" them.
- When the two modes conflict, the task context decides which applies.
  When unsure, default to Mode B.

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
- **One multi-core job at a time.** Never run several multi-core algorithms
  concurrently (our rayon scans, LightGBM, CatBoost, …): oversubscribed cores
  make every run slower. Serialize scorers and fits in drivers and background
  jobs.
- **Logging (future work).** Overall stdlib `logging` across scorers and
  examples is still owed; the pattern is established in
  `examples/elo_round.py` (`setup_logging()`: `FileHandler` +
  `StreamHandler`, timestamped `INFO`). Elo runs use it already — extend the
  same shape rather than inventing new print-to-file plumbing.
- **Codex is a standing buddy.** Use the `codex` CLI freely — ask it for
  design advice, have it code or review parts in parallel, and test its
  output. Don't hesitate; it's part of the workflow.
- **Model cost and delegation.** The primary assistant coordinates the user-facing
  work and reviews the integrated result. Delegate investigation, implementation,
  diagnostics, and test execution to the lowest-cost capable subagent. Use GPT-5.6
  Luna for routine scoped work when available. Upgrade for complexity, risk, or
  failed checks. The primary assistant owns completeness and final evidence.
- **Tests are behavior, not wiring.** Every new capability lands with Rust tests
  (core) and pytest coverage of the user-visible API. Run both suites before
  committing. No test may pin incidental implementation details.
- **Commits**: small, imperative subject + short body listing the behavior added
  and the test counts. Binary data (`datasets/*.parquet`, wheels, models) is not
  committed.
- For small commits, check the staged file list once and use the smallest
  relevant verification. Reuse passing checks when their inputs are unchanged.
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

## Task Observer Protocol

- Apply this protocol when the Task Observer skill and its local workspace are installed.
- **Workspace**: `$HOME/.claude`.
- **Query open observations at session start**:
  ```sh
  find "$HOME/.claude/skill-observations/observation-log" -maxdepth 1 -name '*.md'
  ```
- **Record new observations**:
  ```sh
  bash "$HOME/.agents/skills/task-observer/scripts/new-observation.sh" <slug> "$HOME/.claude"
  ```
- **End-of-turn backstop**: After completing each task, report a one-line summary from the observation log (logged record IDs/titles, or "none logged and why").

## MCP Tools: code-review-graph

- Use `code-review-graph` MCP tools (`detect_changes_tool`, `get_impact_radius_tool`, `query_graph_tool`) to narrow scope before scanning files.
- Always verify exact source code and behavior tests before making non-trivial changes; source code wins over graph output.
