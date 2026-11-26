# Release cycle

The procedure is in [RELEASING.md](../RELEASING.md); this page adds the checks around it.

1. **Scope.** Group the work by spec and issue. Every user-visible change has an issue; bug fixes say `Fixes #N`.
2. **Changelog.** Move notes from `Unreleased` to the new version heading. List behaviour changes (defaults, renamed or removed options, input handling) separately, with the old value so users can pin it.
3. **Version.** Strict `MAJOR.MINOR.PATCH` digits only. Run `scripts/bump_version.py <version>`; do not edit version strings by hand.
4. **Gates.** `cargo check`, `cargo test`, `uv run maturin develop --release`, `uv run pytest tests -q`, and the docs gates in [examples-and-docs.md](examples-and-docs.md). `scripts/prepare_release.py refs/tags/v<version>` verifies the release files.
5. **Docs.** Snippet outputs are current, the sitemap and `llms` files are regenerated, and the early-development notice is still accurate.
6. **Tag and publish** per RELEASING.md after CI passes on `main`. A correction to a published package is a new version.
