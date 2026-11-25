"""Run example scripts and save their stdout for embedding in docs.

    uv run python scripts/docs/capture_outputs.py examples/quickstart.py ...
    uv run python scripts/docs/capture_outputs.py --check examples/quickstart.py

Output goes to docs/snippets/outputs/<script-stem>.txt. Pages embed it with
`--8<-- "docs/snippets/outputs/<stem>.txt"` inside a ```text fence, so the
printed result is always what the script really prints. --check fails if a
saved output is stale. Lines with wall-clock timings are not stable, so the
scripts you capture should not print them (or print them to stderr).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "snippets" / "outputs"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    scripts = [a for a in argv if not a.startswith("--")]
    stale = 0
    for s in scripts:
        path = ROOT / s
        res = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if res.returncode:
            print(f"FAIL {s}\n{res.stderr[-800:]}")
            return 1
        out = res.stdout.rstrip() + "\n"
        target = OUT / f"{path.stem}.txt"
        if check:
            if not target.exists() or target.read_text() != out:
                print(f"STALE {target.relative_to(ROOT)}")
                stale += 1
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            target.write_text(out)
            print(f"wrote {target.relative_to(ROOT)} ({len(out.splitlines())} lines)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
