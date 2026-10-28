# CLAUDE.md

Full project guide for agents: @AGENTS.md

Quick essentials:

- **mltreelib** = Rust-core tree-ML library (research-backed; paper planned). Rust
  source of truth in `crates/mltreelib-core`; Python = thin PyO3 shell in
  `python/mltreelib`; tests in `crates/mltreelib-core/tests/` and `tests/`.
- Commands: `uv sync`, `cargo test`, `uv run maturin develop`, `uv run pytest tests/ -q`.
- No scikit-learn dependency (API shape only). `None` = missing everywhere.
- `legacy/` is a read-only oracle of the old Python implementation; `datasets/`
  holds the TabArena-v0.1 suite (parquet not tracked in git).
- Roadmap + TODO: README.md. Future-but-parked (do not build unprompted):
  categorical encoders, featuretools-style automated features.
