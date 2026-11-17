# CLAUDE.md

Comprehensive agent guide: see [AGENTS.md](AGENTS.md).

## Quick Essentials
- **mltreelib**: Rust core (`crates/mltreelib-core`) with thin PyO3 shell (`crates/mltreelib-py` -> `python/mltreelib`).
- **Commands**:
  - `uv sync` — venv + build extension (maturin backend)
  - `cargo test` — Rust core tests (`crates/mltreelib-core/tests/`)
  - `uv run maturin develop` — rebuild PyO3 extension after Rust edits
  - `uv run pytest tests/ -q` — Python behavior tests
  - `uv run zensical build` — build documentation site (`docs/` -> `site/`)
  - `uv run python examples/gen_api_doc.py` — regenerate API docs
- **Key Rules**:
  - Zero `scikit-learn` dependency (sklearn-compatible API shape only).
  - Missing data first-class (`None` and `NaN` normalized at ingestion; missing in -> missing out, never panics).
  - Rust core is source of truth; Python is thin shell. Tests test behavior, not wiring.
  - `legacy/` is a read-only reference oracle. Never edit or extend it.
  - Benchmark suite: `datasets/` (TabArena-v0.1, OpenML suite 457; parquet files not tracked in git).
  - Active roadmap & TODO: `README.md`.

## Skills & Protocols
- **Task Observer**: Shared workspace at `/home/jkapila/.claude`. Check open observations at session start and record learnings via `bash "/home/jkapila/.agents/skills/task-observer/scripts/new-observation.sh" <slug> "/home/jkapila/.claude"`.
- **Knowledge Graph**: Use `code-review-graph` MCP tools (`detect_changes_tool`, `get_impact_radius_tool`, `query_graph_tool`) to narrow search before reading source code.
- **Specs & Issues**: Specs in `specs/`, issue tracker in `.scratch/`, domain glossary in `GLOSSARY.md`.
