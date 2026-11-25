"""Run the opted-in Python snippets in the docs and refresh their printed output.

    uv run python scripts/docs/run_snippets.py [--check] [paths...]

Convention
- A ```python fence runs only when the line directly above it is exactly
  ``<!-- run -->``. Fences indented inside admonitions or tabs are dedented.
- The run blocks of one page share one namespace, in document order. Stdout
  is captured per block, so later blocks can use earlier variables.
- Right after a run block, a ``<!-- output -->`` marker and a ```text fence
  hold what the block printed. Each run regenerates it; a block that prints
  nothing has no output block. A hand-written ```text block directly after a
  run block is treated as its output and replaced.
- A page stops at its first failing block and is left unchanged. Other pages
  still run. Exit code is 1 if any block failed or any output is stale.
- --check writes nothing and prints ``STALE page:line`` for each output block
  that differs from a fresh run.
"""

import argparse
import contextlib
import io
import os
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN, OUT = "<!-- run -->", "<!-- output -->"
OPEN = re.compile(r"^(\s*)```python\s*$")


def default_pages():
    skip = ROOT / "docs" / "reference" / "api"
    return sorted(p for p in (ROOT / "docs").rglob("*.md")
                  if skip not in p.parents and not p.name.startswith("llms"))


def parse(lines):
    blocks, i = [], 0
    while i < len(lines):
        m = OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        ind, j = m.group(1), i + 1
        while j < len(lines) and lines[j].strip() != "```":
            j += 1
        body = [ln[len(ind):] if ln.startswith(ind) else ln.lstrip() for ln in lines[i + 1:j]]
        blocks.append({"start": i, "end": j, "ind": ind, "run": i > 0 and lines[i - 1].strip() == RUN,
                       "code": textwrap.dedent("\n".join(body))})
        i = j + 1
    return blocks


def find_output(lines, close):
    """Return (start, end) of the output block after the fence closing at `close`, or None."""
    k = close + 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    start = k
    if k < len(lines) and lines[k].strip() == OUT:
        k += 1
        while k < len(lines) and not lines[k].strip():
            k += 1
    if k >= len(lines) or lines[k].strip() != "```text":
        return None
    end = k + 1
    while end < len(lines) and lines[end].strip() != "```":
        end += 1
    return start, end + 1


def render(lines, blocks, outs):
    """Rebuild the page with fresh output blocks. Returns (new_lines, stale_line_numbers)."""
    out, stale, pos = [], [], 0
    for b in blocks:
        out += lines[pos:b["end"] + 1]
        ind, text = b["ind"], outs[b["start"]]
        region = find_output(lines, b["end"])
        want = [] if not text else ([ind + OUT, ind + "```text"]
                                    + [ind + ln if ln else "" for ln in text.split("\n")] + [ind + "```"])
        have = lines[region[0]:region[1]] if region else []
        if have != want:
            stale.append(b["start"] + 1)
        out += ([""] + want) if want else []
        pos = region[1] if region else b["end"] + 1
        while pos < len(lines) and not lines[pos].strip():
            pos += 1
        out.append("")
    out += lines[pos:]
    return out, stale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(ROOT)
    pages = [Path(p).resolve() for p in args.paths] or default_pages()
    bad = False
    for page in pages:
        try:
            rel = page.relative_to(ROOT)
        except ValueError:
            rel = page
        lines = page.read_text(encoding="utf-8").split("\n")
        blocks = [b for b in parse(lines) if b["run"]]
        ns, outs, failed = {"__name__": "__main__"}, {}, False
        for b in blocks:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(b["code"], f"{rel}:{b['start'] + 1}", "exec"), ns)  # noqa: S102
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {rel}:{b['start'] + 1}: {type(exc).__name__}: {exc}")
                failed = bad = True
                break
            outs[b["start"]] = buf.getvalue().rstrip("\n")
        if failed:
            continue
        new, stale = render(lines, blocks, outs)
        if new != lines and not stale and blocks:
            stale = [blocks[0]["start"] + 1]  # spacing-only drift
        if args.check:
            for line in stale:
                print(f"STALE {rel}:{line}")
            bad = bad or bool(stale)
        elif new != lines:
            page.write_text("\n".join(new), encoding="utf-8")
            print(f"updated {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
