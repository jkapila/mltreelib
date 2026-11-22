"""Check the tagged version across all files and print its changelog entry."""

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def verify_release(root: Path, version: str) -> str:
    """Verify all version-bearing files match the expected release version and return changelog entry."""
    errors = []

    # 1. pyproject.toml
    pyproject_file = root / "pyproject.toml"
    if pyproject_file.exists():
        pyproject_ver = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))["project"]["version"]
        if pyproject_ver != version:
            errors.append(f"pyproject.toml version '{pyproject_ver}' != '{version}'")
    else:
        errors.append("pyproject.toml not found")

    # 2. Cargo.toml
    cargo_file = root / "Cargo.toml"
    if cargo_file.exists():
        cargo_ver = tomllib.loads(cargo_file.read_text(encoding="utf-8"))["workspace"]["package"]["version"]
        if cargo_ver != version:
            errors.append(f"Cargo.toml version '{cargo_ver}' != '{version}'")
    else:
        errors.append("Cargo.toml not found")

    # 3. python/mltreelib/__init__.py
    init_file = root / "python/mltreelib/__init__.py"
    if init_file.exists():
        init_src = init_file.read_text(encoding="utf-8")
        if f'__version__ = "{version}"' not in init_src.splitlines():
            errors.append(f"python/mltreelib/__init__.py missing '__version__ = \"{version}\"'")
    else:
        errors.append("python/mltreelib/__init__.py not found")

    # 4. tests/test_audit_gap_fixes.py
    test_file = root / "tests/test_audit_gap_fixes.py"
    if test_file.exists():
        test_src = test_file.read_text(encoding="utf-8")
        if f'assert mltreelib.__version__ == "{version}"' not in test_src:
            errors.append(f"tests/test_audit_gap_fixes.py missing 'assert mltreelib.__version__ == \"{version}\"'")

    # 5. zensical.toml
    zensical_file = root / "zensical.toml"
    if zensical_file.exists():
        zensical_src = zensical_file.read_text(encoding="utf-8")
        if f'site_name = "mltreelib v{version}"' not in zensical_src:
            errors.append(f"zensical.toml missing 'site_name = \"mltreelib v{version}\"'")

    # 6. docs/index.md
    docs_file = root / "docs/index.md"
    if docs_file.exists():
        docs_src = docs_file.read_text(encoding="utf-8")
        if f"releases/tag/v{version}" not in docs_src:
            errors.append(f"docs/index.md missing release tag link 'releases/tag/v{version}'")

    # 7. README.md
    readme_file = root / "README.md"
    if readme_file.exists():
        readme_src = readme_file.read_text(encoding="utf-8")
        if f"From a v{version} checkout" not in readme_src:
            errors.append(f"README.md missing 'From a v{version} checkout'")
        if f"mltreelib_{version}.tar.gz" not in readme_src:
            errors.append(f"README.md missing 'mltreelib_{version}.tar.gz'")

    # 8. r/build-r-package.sh
    r_build_file = root / "r/build-r-package.sh"
    if r_build_file.exists():
        r_build_src = r_build_file.read_text(encoding="utf-8")
        if f'version = "{version}"' not in r_build_src:
            errors.append(f"r/build-r-package.sh missing 'version = \"{version}\"'")

    # 9. r/mltreelib/DESCRIPTION
    r_desc_file = root / "r/mltreelib/DESCRIPTION"
    if r_desc_file.exists():
        r_desc_src = r_desc_file.read_text(encoding="utf-8")
        if f"Version: {version}" not in r_desc_src:
            errors.append(f"r/mltreelib/DESCRIPTION missing 'Version: {version}'")

    # 10. r/mltreelib/tools/rust/Cargo.toml
    r_cargo_file = root / "r/mltreelib/tools/rust/Cargo.toml"
    if r_cargo_file.exists():
        r_cargo_src = r_cargo_file.read_text(encoding="utf-8")
        if f'version = "{version}"' not in r_cargo_src:
            errors.append(f"r/mltreelib/tools/rust/Cargo.toml missing 'version = \"{version}\"'")

    # 11. CHANGELOG.md
    changelog_file = root / "CHANGELOG.md"
    if not changelog_file.exists():
        errors.append("CHANGELOG.md not found")
        entry = ""
    else:
        changelog = changelog_file.read_text(encoding="utf-8")
        sections = re.split(r"^## (.+)$", changelog, flags=re.MULTILINE)
        entries = [
            body.strip()
            for heading, body in zip(sections[1::2], sections[2::2])
            if heading.strip() == f"v{version}"
        ]
        if len(entries) != 1 or not entries[0]:
            errors.append(f"CHANGELOG.md missing one nonempty '## v{version}' entry")
            entry = ""
        else:
            entry = entries[0]

    if errors:
        err_msg = (
            f"Release verification failed for v{version}:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + f"\n\nRun 'python scripts/bump_version.py {version}' to synchronize files."
        )
        sys.exit(err_msg)

    return entry


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/prepare_release.py refs/tags/v<VERSION>")

    ref = sys.argv[1]
    match = re.fullmatch(r"refs/tags/v(\d+\.\d+\.\d+)", ref)
    if match is None:
        # Also allow plain vX.Y.Z
        match = re.fullmatch(r"v?(\d+\.\d+\.\d+)", ref)
        if match is None:
            sys.exit("A publishing run requires a valid vMAJOR.MINOR.PATCH tag.")
    version = match[1]

    root = Path(__file__).resolve().parents[1]
    changelog_entry = verify_release(root, version)
    print(changelog_entry)
