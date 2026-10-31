"""Generate docs/reference/api/*.md from the live mltreelib API.

Forwards to examples/gen_api_doc.py (which generates per-class pages served by nav).
"""

from __future__ import annotations

from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "examples" / "gen_api_doc.py"
    runpy.run_path(str(target), run_name="__main__")
