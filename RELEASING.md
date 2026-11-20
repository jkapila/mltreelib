# Releasing mltreelib

Use one release pull request for the version bump and the user-facing notes. Tag its merged commit after CI passes. GitHub Actions builds the artifacts, publishes them to PyPI, and then creates the GitHub release.

## Prepare the release

1. Choose the next version. Use a patch for compatible fixes, a minor for compatible capabilities, and a major for breaking changes after 1.0. Before 1.0, use a minor bump for breaking changes and describe the migration in the notes.
2. Set the version in `Cargo.toml`, `pyproject.toml`, and `python/mltreelib/__init__.py`. Update the expected version in `tests/test_audit_gap_fixes.py`.
3. Refresh both lockfiles. Move the relevant notes from `Unreleased` to a heading such as `## v0.4.11` in [CHANGELOG.md](CHANGELOG.md). Describe user-visible changes and include upgrade steps when needed.
4. Run the checks below in order. Open the release pull request after they pass.

```sh
cargo check --workspace
uv lock
cargo test --locked
uv run maturin develop --release --locked
uv run pytest tests/ -q
uv run --python 3.12 python scripts/prepare_release.py refs/tags/v0.4.11
```

## Tag and publish

After the release PR merges and CI passes on `main`, tag the exact commit with an annotated tag. Use the version from the release PR in both commands.

```sh
git switch main
git pull --ff-only
release_commit=REPLACE_WITH_TESTED_COMMIT
git tag -a v0.4.11 -m "Release mltreelib v0.4.11" "$release_commit"
git push origin refs/tags/v0.4.11
```

Follow the [Release workflow](https://github.com/jkapila/mltreelib/actions/workflows/release.yaml). It checks the tag against the Cargo and Python versions and requires a matching changelog entry. It builds Linux, macOS, and Windows wheels and a source distribution. The `pypi` environment uses PyPI Trusted Publishing for `.github/workflows/release.yaml`.

After the workflow passes, check the version on [PyPI](https://pypi.org/project/mltreelib/#history) and [GitHub Releases](https://github.com/jkapila/mltreelib/releases). Check that a fresh virtual environment can install and import the pinned version.

## Retry or archive

- Before an upload, retry failed jobs if the tagged source is correct. If the source needs a change after the tag is public, fix it on `main` and cut a new version.
- If PyPI succeeds and the GitHub publishing job fails, retry the failed job. It uses the notes prepared from the tag.
- If a PyPI upload fails, check which files arrived before you retry. PyPI cannot reuse file names that it already accepted.
- For a build without publishing, run the workflow manually with `publish` disabled.
- For GitHub-only historical releases, use a verified historical commit and an `archive/` tag. Set `--latest=false` to preserve the current latest release.

Use a new version for corrections to a published package. PyPI records the date each file arrives, including files uploaded for older versions.
