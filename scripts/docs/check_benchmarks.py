"""Validate benchmarks/benchmarks.yaml.

- every script under benchmarks/{suites,speed,hpo} is listed
- every `publish: true` entry has a page that is in the nav, existing results,
  a reproduce command and a title
"""

import re
import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[2]
entries = yaml.safe_load((root / "benchmarks" / "benchmarks.yaml").read_text())
nav = set(re.findall(r'"([^"]+\.md)"', (root / "zensical.toml").read_text()))
errors: list[str] = []

listed = {e["script"] for e in entries}
for grp in ("suites", "speed", "hpo"):
    for f in sorted((root / "benchmarks" / grp).glob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel not in listed and f.name != "run_all.py":
            errors.append(f"unlisted script: {rel}")
for e in entries:
    if not (root / e["script"]).exists():
        errors.append(f"{e['id']}: script missing: {e['script']}")
    if not e.get("publish"):
        continue
    for k in ("title", "page", "reproduce", "results"):
        if not e.get(k):
            errors.append(f"{e['id']}: published entry needs `{k}`")
    page = e.get("page", "")
    if page and page not in nav:
        errors.append(f"{e['id']}: page not in nav: {page}")
    for r in e.get("results") or []:
        if not (root / r).exists():
            errors.append(f"{e['id']}: results missing: {r}")
print("\n".join(errors) or f"ok: {len(entries)} entries, {sum(bool(e.get('publish')) for e in entries)} published")
sys.exit(1 if errors else 0)
