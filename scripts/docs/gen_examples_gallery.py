"""Generate docs/examples.md: one table row per runnable script in examples/.

Run: uv run python scripts/docs/gen_examples_gallery.py
The 'What it shows' column is the first line of each script's module docstring; the 'Run' column is the first `uv run` line of its Usage block (scripts that need extra packages use `uv run --with <package> python ...`).
"""

import ast
from pathlib import Path

root = Path(__file__).resolve().parents[2]
BLOB = "https://github.com/jkapila/mltreelib/blob/main/"
rows = []
for f in sorted((root / "examples").rglob("*.py")):
    doc = ast.get_docstring(ast.parse(f.read_text(encoding="utf-8"))) or ""
    first = doc.strip().split("\n")[0].strip().rstrip(".") if doc else ""
    rel = f.relative_to(root).as_posix()
    cmd = next((ln.strip() for ln in doc.splitlines() if ln.strip().startswith("uv run")), f"uv run python {rel}")
    rows.append(f"| [`{f.name}`]({BLOB}{rel}) | {first.replace('|', '/') or '-'} | `{cmd}` |")

page = f"""---
title: Examples
description: Runnable example scripts for mltreelib, one per topic.
---

# Examples

Short, runnable scripts. Each one finishes quickly and prints what to look for. For explanation, use the [Guides](guides/index.md) (do a task) or the [Concepts](concepts/core-tree-architecture.md) (how it works).

Run any script from the repository root:

```bash
uv run python examples/quickstart.py
```

| Script | What it shows | Run |
| :--- | :--- | :--- |
{chr(10).join(rows)}
"""
(root / "docs" / "examples.md").write_text(page, encoding="utf-8")
print(f"wrote docs/examples.md ({len(rows)} scripts)")
