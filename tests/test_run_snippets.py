import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_snippets", ROOT / "scripts/docs/run_snippets.py")
run_snippets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_snippets)

PAGE = """# Fake

<!-- run -->
```python
x = 21
print(x * 2)
```

Then a paragraph.

<!-- run -->
```python
print(x + 1)
```

Trailing text.
"""


def test_output_inserted_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = tmp_path / "fake.md"
    page.write_text(PAGE, encoding="utf-8")

    assert run_snippets.main([str(page)]) == 0
    once = page.read_text(encoding="utf-8")
    assert "<!-- output -->\n```text\n42\n```" in once
    assert "<!-- output -->\n```text\n22\n```" in once
    assert "Then a paragraph." in once

    assert run_snippets.main([str(page)]) == 0
    assert page.read_text(encoding="utf-8") == once
    assert run_snippets.main(["--check", str(page)]) == 0
