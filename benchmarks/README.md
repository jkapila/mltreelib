# Benchmarks

Benchmarks measure; examples teach (see `examples/`). Everything here may be slow and may need datasets or R.

```text
benchmarks/
├── benchmarks.yaml   manifest: which benchmarks are published to the site
├── arena/            Elo arena: prepare, fit, score and report (and the MARS fixtures it uses)
├── suites/           one verification suite per topic; also the regression methods study
├── speed/            speed and scaling checks
├── hpo/              internal tuning runs
├── elo_*.csv         small tracked results the site reads
└── runs/ rounds/ *_data/ logs   run artifacts; most of these are not meant to be tracked
```

## Publishing a benchmark

A benchmark appears on the site only if its entry in `benchmarks.yaml` has `publish: true`. The docs build runs `scripts/docs/check_benchmarks.py`, which fails unless the entry has:

- a `page` that is in the nav, with narration (what it verifies, how to read it, its limits)
- `results` files that exist (a small CSV, or a captured output)
- a `reproduce` command
- a `title`

Every script under `suites/`, `speed/` and `hpo/` must be listed in the manifest, published or not.

## Capturing printed output for docs

```bash
uv run python scripts/docs/capture_outputs.py benchmarks/suites/regression_methods_showcase.py
uv run python scripts/docs/capture_outputs.py --check examples/quickstart.py
```

This saves stdout to `docs/snippets/outputs/<script>.txt`. Pages embed it with `--8<-- "docs/snippets/outputs/<script>.txt"` inside a `text` fence, so the printed result is what the script really prints. `--check` fails when a saved output is stale. Use it only for scripts whose output has no timings; benchmark outputs that print fit times will always differ between runs.
