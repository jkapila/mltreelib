# v0.4.11 Elo run archive

This archive preserves the score rows, recipes, attempt ledgers, input hashes,
and comparison inputs for the 56-dataset ModelTree(all) rerun and the 35-entry
Diamonds correction. Read the [run notes](../../../references/research/2026-11-21-modeltree-all-elo-rerun.md)
and [consolidated board](../../../docs/benchmarks/elo_overall.md).

Regenerate the terminal board from the recorded scores at the repository root.

```bash
uv run --no-cache --no-sync python benchmarks/runs/elo-v0.4.11-all/diamonds-peers/compare.py --table
```

The archive includes the original benchmark recipes as historical research
inputs. Training splits, logs, environment links, and preflight outputs are
local run artifacts. The [run manifest](manifest.json) records input hashes
and build provenance. Refit work needs the pinned package build, prepared
inputs, and model-specific dependencies.
