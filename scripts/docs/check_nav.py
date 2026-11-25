"""Fail if a docs page is neither in the zensical.toml nav nor a `moved_to` stub."""

import re
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[2]
cfg = (root / "zensical.toml").read_text(encoding="utf-8")
nav = set(re.findall(r'"([^"]+\.md)"', cfg[cfg.index("nav = ["):]))
excluded = set(tomllib.loads(cfg)["project"].get("exclude_docs", "").split())
orphans = []
for md in sorted((root / "docs").rglob("*.md")):
    rel = md.relative_to(root / "docs").as_posix()
    if rel in nav or rel in excluded or rel.startswith("reference/api/"):
        continue
    if md.read_text(encoding="utf-8").startswith("---") and "moved_to:" in md.read_text(encoding="utf-8")[:300]:
        continue
    orphans.append(rel)
missing = sorted(n for n in nav if not n.startswith("http") and not (root / "docs" / n).exists())
for o in orphans:
    print(f"ORPHAN  {o}")
for m in missing:
    print(f"MISSING {m}")
sys.exit(1 if orphans or missing else 0)
