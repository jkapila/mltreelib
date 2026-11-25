# Workflow: issues, branches, commits, reviews

## Start from an issue

- **New feature or behaviour change:** open a GitHub issue first (the package is in early development and APIs may change). Say what and why; link the spec if one exists.
- **Bug:** open an issue with a minimal reproduction, then fix it on a branch. The fix commit says `Fixes #N`.
- **Spec work:** a spec in `specs/` (see [specs.md](specs.md)) can span many issues; branch names carry the spec numbers.

## Branches and commits

- Branch from `main`: `fix/<short-slug>`, `feat/<short-slug>`, or `spec-NN[-MM]/<short-slug>`.
- One logical change per commit. Subject in the imperative with a type prefix: `fix(scope):`, `feat(scope):`, `docs:`, `test:`, `refactor:`, `chore:`. Body says what changed and why, not a diff.
- The pre-commit hooks (ruff, ty, cargo clippy) must pass. **Verify a commit by reading the state** (`git log -1 --oneline`, `git status --short`), never by filtering the commit command's output: a filtered view can hide a failed hook.
- End commit messages with the attribution trailer the session provides. Do not commit `datasets/*.parquet`, wheels, models or secrets.
- Do not push, merge or delete branches unless the maintainer asked.

## Checks before handing over

Follow [definition-of-done.md](definition-of-done.md). In short: tests for behaviour, existing docs and examples updated with real output, and all gates green.

## Working with agents

Follow [agents.md](agents.md): lowest-cost capable agent for scoped work, the mandatory brief, and review of every report before accepting it.
