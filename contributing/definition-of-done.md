# Definition of done for a code change

A change is done only when the code, its tests, **and the existing documentation and examples that show it** agree with each other and with real output. This applies to people and to agents.

## Rules

1. **Write the tests with the code.** Test behaviour, not wiring: normal case, missing values (`None` and NaN must be kept, counted and never imputed or recoded by data helpers), empty input, determinism where randomness is used, and that inputs are not modified.
2. **Update the existing docs and examples, do not add a parallel set.** Find the page or script that already teaches the topic (guide, concept, reference, example, README) and change it there. Create a new page or script only when no existing one fits, and say why in the pull request.
3. **Replace hand-rolled code the change makes obsolete.** If a helper now exists, snippets that do the same thing by hand (manual split indexing, list comprehensions that rebuild a matrix, pandas NaN conversion) switch to the helper.
4. **Show real output.** Documentation snippets run with `<!-- run -->` and carry generated output blocks. Never edit output by hand. If prose or a table disagrees with the output, change the prose, even when the result looks unflattering.
5. **Use real data.** Snippets use datasets that `load_dataset` can fetch (or seeded synthetic data when the point is a known truth). No repo-relative paths, no untracked files.
6. **Run the gates before you hand over:** `uv run python scripts/docs/run_snippets.py --check`, `scripts/docs/check_nav.py`, `scripts/docs/check_benchmarks.py`, `uv run zensical build`, `uv run pytest tests -q`, and `cargo test` when Rust changed.
7. **Report honestly.** List what changed, what you could not verify, any API gap, and any statement in the docs you could not tie to output.
8. **Never normalise missing values** in a data helper, and add no dependency: the package has none, and scikit-learn is never allowed.

## For agents

An agent that changes code is also responsible for steps 2 to 6, not only the code. A report that says "tests pass" without the docs and examples updated is incomplete.
