# Documentation Generation

How documentation is produced in this repo, and the exact steps an agent must
follow after changing the API. Read this before touching anything under
`docs/`.

## The one rule that drives everything

**Docstrings are the source of truth for API reference; runnable example
output is the source of truth for guides.** Nothing in `docs/` is written
from memory. Every signature and parameter description comes from a live
docstring rendered through the built `_core` extension; every number in a
guide is real output pasted from a script that actually ran.

## Layout

```
docs/
  index.md, installation.md, quickstart.md   hand-written
  examples/*.md                              hand-written guides, REAL output pasted in
  reference/*.md                             hand-written guides (data, model, metrics, reports, plots)
  reference/api/*.md                         GENERATED per-class API pages (site nav serves these)
  reference/api.md                           GENERATED single-file API reference (see "The collision")
  agents/*.md                                agent-facing process docs (this file lives here)
  benchmarks/*.md                            Elo/dataset boards
site/                                        build output of docs/ (gitignored)
```

## The build

```sh
uv run zensical build        # docs/ -> site/; also validates the nav resolves
```

Run this once after any `docs/` change. It fails loudly on nav or output-path
conflicts — treat a build error as a blocker, never ship a broken site.

## API reference: two generators, one collision

There are **two** API-reference generators. Both read live docstrings from the
built extension (run `uv run maturin develop` first, or they render stale
signatures).

| Command | Output | Used by the site? |
|---|---|---|
| `uv run python examples/gen_api_doc.py` | `docs/reference/api/{index,data,model-tree,metrics,encoders,features}.md` | **YES** — this is what `mkdocs.yml` nav serves |
| `uv run python scripts/gen_api_docs.py` | `docs/reference/api.md` (single file) | **NO** — not in the nav, not in `llmstxt` |

**The collision:** `docs/reference/api.md` and `docs/reference/api/index.md`
both map to the output page `reference/api/index.html`. If both files exist on
disk, `zensical build` aborts with
`page destination 'reference/api/index.html' has multiple owners`. This is a
pre-existing repo inconsistency (the per-class pages were added in `b7582c2`
without removing the single file from the build scope), not something a
single change introduces.

**Current status (2026-10-31):** the site's API reference is the **per-class
pages**. `docs/reference/api.md` is tracked but orphaned (no nav entry, no
`llmstxt` entry) and only breaks the build. Until the collision is resolved,
regenerating the single-file reference and committing it alongside
`docs/reference/api/` will break the site.

**Recommended resolution (pick one, record it in the README/AGENTS):**
- **Consolidate on the per-class pages** — stop generating/committing
  `docs/reference/api.md`, and update `AGENTS.md` to name
  `examples/gen_api_doc.py` as the API-regen command. Simplest; the site
  already serves these.
- **Or move the single file out of `docs/`** (e.g. `references/api.md`) if a
  monolithic offline/llms reference is genuinely wanted, so it no longer
  collides with the site build.

## After changing the API — the checklist

1. **Rebuild the extension** so docstrings are current:
   `uv run maturin develop`.
2. **Regenerate the per-class API pages** (the ones the site serves):
   `uv run python examples/gen_api_doc.py`.
   Do **not** also run `scripts/gen_api_docs.py` unless the collision above
   has been resolved — it will break the build.
3. **Update the relevant guide** under `docs/reference/*.md` (e.g. a new
   `ModelTree` parameter belongs in `docs/reference/model.md`) and/or
   `docs/examples/*.md` if it changes how a feature is used. Paste **real
   output** from the example script, never an invented transcript.
4. **Regenerate `docs/reference/api.md`** only if the repo has chosen to keep
   it (see collision note) — and never commit it beside `docs/reference/api/`.
5. **Build the site** and confirm it passes: `uv run zensical build`.
6. If you added a **new example page**, wire it into `mkdocs.yml` twice:
   the `nav:` list **and** the `llmstxt:` `sections: Documentation:` list.

## Adding a new example guide

- Put the runnable script in `examples/<name>.py` (self-contained, seeded,
  prints its own output).
- Run it and capture the **real** output.
- Write `docs/examples/<name>.md` with the output pasted in as a ` ```text `
  block, plus prose explaining what it shows.
- Add the page to `mkdocs.yml` `nav:` and the `llmstxt:` list.
- `uv run zensical build` to verify.

## Rules that do not bend

- **Real output only.** Every code block showing results must come from a run
  that happened this session. No fabricated numbers, no "approximately".
- **Docstrings are generated, never hand-edited.** The files under
  `docs/reference/api/` carry a `do not edit` header. To change what they
  say, change the Rust/Python docstring and regenerate.
- **Guides are hand-written** (`docs/examples/*.md`, `docs/reference/*.md`,
  `docs/quickstart.md`) and are safe to edit directly — but their numbers
  still come from real runs.
- **The site build is the gate.** A change is not delivered until
  `uv run zensical build` passes.

## Commands (canonical)

```sh
uv run maturin develop                     # rebuild extension (current docstrings)
uv run python examples/gen_api_doc.py      # regen site API pages (docs/reference/api/)
uv run python scripts/gen_api_docs.py      # regen single-file api.md (collides — see above)
uv run python examples/<name>.py           # run an example, capture real output
uv run zensical build                      # build + validate docs site
```
