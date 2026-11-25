# Examples

Short, runnable scripts that teach one topic each. Run any of them with `uv run python <path>`.

```text
examples/
├── quickstart.py     start here
└── guides/           one script per topic; the stem matches docs/guides/<topic>.md
```

Layout rules (see also `benchmarks/README.md`):

- `examples/` teaches. Scripts finish in seconds to a minute and print what to look for.
- `benchmarks/` measures. Slow runs, arenas, and verification suites live there.
- `scripts/` is tooling: docs generation, releases, dataset fetching.
