"""Check the tagged version and print its changelog entry."""

import re
import sys
from pathlib import Path

import tomllib

if __name__ == "__main__":
    ref = sys.argv[1]
    match = re.fullmatch(r"refs/tags/v(\d+\.\d+\.\d+)", ref)
    if match is None:
        sys.exit("A publishing run requires an existing vMAJOR.MINOR.PATCH tag.")
    version = match[1]

    root = Path(__file__).resolve().parents[1]
    python_version = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "version"
    ]
    rust_version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"][
        "package"
    ]["version"]
    package_source = (root / "python/mltreelib/__init__.py").read_text()
    if version != python_version or version != rust_version:
        sys.exit(f"Tag v{version} must match Cargo.toml and pyproject.toml.")
    if f'__version__ = "{version}"' not in package_source.splitlines():
        sys.exit(f"Tag v{version} must match mltreelib.__version__.")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.split(r"^## (.+)$", changelog, flags=re.MULTILINE)
    entries = [
        body.strip()
        for heading, body in zip(sections[1::2], sections[2::2])
        if heading.strip() == f"v{version}"
    ]
    if len(entries) != 1 or not entries[0]:
        sys.exit(f"Add one nonempty '## v{version}' entry to CHANGELOG.md.")
    print(entries[0])
