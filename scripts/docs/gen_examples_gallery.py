"""Generate docs/examples.md: one table row per runnable script in examples/.

Run: uv run python scripts/docs/gen_examples_gallery.py
The 'What it shows' column is the first line of each script's module docstring.
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
    rows.append(f"| [`{f.name}`]({BLOB}{rel}) | {first.replace('|', '/') or '-'} |")

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

| Script | What it shows |
| :--- | :--- |
{chr(10).join(rows)}
"""
(root / "docs" / "examples.md").write_text(page, encoding="utf-8")
print(f"wrote docs/examples.md ({len(rows)} scripts)")
