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
- **One multi-core job at a time.** Never run several multi-core algorithms
  concurrently (our rayon scans, LightGBM, CatBoost, …): oversubscribed cores
  make every run slower. Serialize scorers and fits in drivers and background
  jobs.
- **Codex is a standing buddy.** Use the `codex` CLI freely — ask it for
  design advice, have it code or review parts in parallel, and test its
  output. Don't hesitate; it's part of the workflow.
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

(task-observer activation block — template of v3.8.1)
Before the first tool call of any session — and before writing or
proposing a plan, not merely before executing one — invoke the
task-observer skill AND execute its Session Start Protocol (storage
check, frontmatter scan, review trigger). Loading the skill and running
the protocol are separate steps; a session that loads the file and stops
has activated nothing. Any turn that will involve a tool call counts; do
not classify the session as "too simple" from its opening message.
A step you did not run is reported before the answer, in one line — never
after it, where it reads as a suggestion. If the skill does not resolve on
this turn, say which probe told you so (the skill listing, or the skill
directory on disk), re-check the listing on every later turn that makes a
tool call, and run the protocol the first time it resolves — a "not
installed" verdict holds for one turn.

Select skills on the DECISION the request is about, not on the artefact it
arrived as. Name what the user is deciding, then match the installed skill
descriptions against that — a request handed over as a file to review
still needs the skill whose description names its subject.

After completing each task, list the observation records written this
session in the same turn and report a one-line summary from that listing
(ids and titles, or "none logged and why") — never from memory of
writing them. This is the activation backstop: it forces a look at the log,
so a session that silently skipped the protocol is discovered at the
first task boundary instead of never.

Loading a skill is not complete until you have queried the observation
log for OPEN observations naming it and read their bodies:
  find "/home/jkapila/.claude/skill-observations/observation-log" -maxdepth 1 \
    -name '*.md' -exec grep -l -E "^(skill|target_file):.*<skill-name>" {} +
(<skill-name> is the bare name: drop a plugin: prefix, up to the last
colon. A qualified pattern misses every entry filed under the bare name
and reports a clean zero; the bare one matches both spellings. For an
agent definition or instructions file it is the file name, which the
target_file: field carries by any path.)
(Use find, not a bare *.md glob: under zsh an unmatched glob aborts the
whole block, and on an empty log it is unmatched in every session.)
Apply their insights to the current work — meaning: let them change what
you do in THIS task. Editing the skill file, or writing the rule into any
other file a later session reads, is acting on the observation and waits
for the review. See "Log, don't act" in SKILL.md: the test is whether the
action leaves a durable change outside the observation log. Run this at
every skill load, however many skills load in one session. The
session-start scan does not cover it: that is a frontmatter sweep over
every observation at session start, this is a body-level lookup for one
skill at the moment its rules are applied.

The converse holds too: the grep, and any checkpoint line recording the
load, run only in the same batch as the Skill invocation, and take the
skill name from a load that has actually happened — never from a list of
skills you built to load. A checkpoint line written without the load is a
false record. Before writing to any file a skill reads at run time (a
state file), load that skill: a state file that describes itself is not a
substitute for the skill that owns it.

Where scripts run, the lookup and its checkpoint line are one command:
  bash "/home/jkapila/.agents/skills/task-observer/scripts/skill-load.sh" "/home/jkapila/.claude" <skill-name>
The Skill invocation and that call are ONE batch — emit them together,
one call per skill loaded, never the call in a later batch. On a first
run the log does not exist yet (step 1 creates it): the call says so and
exits 0 with no checkpoint line; re-run it once per skill loaded in that
batch after the Workspace creation command.

The task-observer workspace for this project is:
  /home/jkapila/.claude
It does not exist until the first session creates it: a failed probe whose
parent resolves and is writable is a first run — run Workspace creation,
then re-run the lookup above once per skill loaded in that batch (on the
first batch it finds no log and writes no checkpoint line; that is
expected). Every path the skill uses derives from that root and nothing
else:
  /home/jkapila/.claude/skill-observations/observation-log/   (the log)
  /home/jkapila/.claude/skill-observations/cross-cutting-principles.md
  /home/jkapila/.claude/skill-updates/                        (staging root)
  /home/jkapila/.claude/skill-updates/PENDING.md              (staging manifest)
Never resolve any of them from the current working directory — a cwd
inside an ephemeral checkout (a git worktree, a temporary clone) is torn
down and takes the log with it. Never place the workspace inside a
skills-discovery directory or any path linked into one. If this
environment mints a separate project identity per checkout, or more than
one agent works this project, the pinned path above is the single shared
location; do not derive one per session, tool or project.

The user's repositories live under:
  /mnt/d/Programs
(the root the sibling check sweeps before declaring a skill absent;
observation-log.md, "Skill families and the sibling check").

A subagent dispatched by a session that already runs this protocol does
not run it again and does not write to the log. The controller observes,
because only it sees the whole task and its review; a subagent that
notices something worth logging says so in its final report, and the
controller writes it, running the id snippet per write. A start-up rule
in a project file reaches every agent whose working folder is inside the
project — a subagent, or a session started there. When this session
starts another session (not a subagent), pass it a working folder inside
the project, or the workspace root, so the block is in context from its
first turn; a folder named only in the brief does not load it.
Before briefing a delegate under an agent definition or instructions
file, run the lookup for that file's name; its open entries shape the
brief.

A run of this block that cannot write here or get an approval answered —
a child run with no protocol-running caller, a job with a narrow tool
allowlist, a scheduled task in default permission mode — skips the write
steps, never stalls on them, and ends its final message with a
`## Task-observer notes` section (environments.md, "Sessions whose
output channel is owned by a caller"). Such runs write their output to:
  none

An observation file comes into being only through
  bash "/home/jkapila/.agents/skills/task-observer/scripts/new-observation.sh" <slug> "/home/jkapila/.claude"
which prints the path to write into. Never by copying another file's
header or counting a listing — including on the turn after a
compaction, when the skill body is out of context.
