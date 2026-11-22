"""Bump version across all configuration, metadata, code, test, and documentation files.

Usage:
    python scripts/bump_version.py <NEW_VERSION> [--dry-run]

Example:
    python scripts/bump_version.py 0.4.15
"""

import argparse
import re
import sys
from pathlib import Path

import tomllib


def parse_version(raw_version: str) -> str:
    """Normalize and validate semantic version."""
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+)", raw_version.strip())
    if not match:
        sys.exit(f"Invalid version format: '{raw_version}'. Expected MAJOR.MINOR.PATCH (e.g. 0.4.15).")
    return match.group(1)


def get_current_version(root: Path) -> str:
    """Read current version from pyproject.toml."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        sys.exit("pyproject.toml not found at repository root.")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


def replace_in_file(path: Path, pattern: str, replacement: str, dry_run: bool = False) -> bool:
    """Perform regex replacement in a file and save if changed."""
    if not path.exists():
        print(f"Warning: File {path} not found; skipped.")
        return False

    content = path.read_text(encoding="utf-8")
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        print(f"Warning: Pattern '{pattern}' not matched in {path}.")
        return False

    if not dry_run and new_content != content:
        path.write_text(new_content, encoding="utf-8")
    return True


def bump_version(new_version: str, root: Path, dry_run: bool = False) -> None:
    """Update version across all tracked files."""
    old_version = get_current_version(root)
    escaped_old = re.escape(old_version)
    print(f"{'[DRY RUN] ' if dry_run else ''}Bumping version: v{old_version} -> v{new_version}")

    targets = [
        # (relative_path, regex_pattern, replacement)
        (
            Path("pyproject.toml"),
            rf'(?m)^version\s*=\s*"{escaped_old}"',
            f'version = "{new_version}"',
        ),
        (
            Path("Cargo.toml"),
            rf'(?m)^version\s*=\s*"{escaped_old}"',
            f'version = "{new_version}"',
        ),
        (
            Path("python/mltreelib/__init__.py"),
            rf'(?m)^__version__\s*=\s*"{escaped_old}"',
            f'__version__ = "{new_version}"',
        ),
        (
            Path("tests/test_audit_gap_fixes.py"),
            rf'assert mltreelib\.__version__ == "{escaped_old}"',
            f'assert mltreelib.__version__ == "{new_version}"',
        ),
        (
            Path("zensical.toml"),
            rf'site_name = "mltreelib v{escaped_old}"',
            f'site_name = "mltreelib v{new_version}"',
        ),
        (
            Path("docs/index.md"),
            rf'\[v{escaped_old} release notes\]\(https://github\.com/jkapila/mltreelib/releases/tag/v{escaped_old}\)',
            f'[v{new_version} release notes](https://github.com/jkapila/mltreelib/releases/tag/v{new_version})',
        ),
        (
            Path("README.md"),
            rf'From a v{escaped_old} checkout',
            f'From a v{new_version} checkout',
        ),
        (
            Path("README.md"),
            rf'dist/mltreelib_{escaped_old}\.tar\.gz',
            f'dist/mltreelib_{new_version}.tar.gz',
        ),
        (
            Path("r/build-r-package.sh"),
            rf'(?m)^version\s*=\s*"{escaped_old}"',
            f'version = "{new_version}"',
        ),
        (
            Path("r/mltreelib/DESCRIPTION"),
            rf'(?m)^Version:\s*{escaped_old}',
            f'Version: {new_version}',
        ),
        (
            Path("r/mltreelib/tools/rust/Cargo.toml"),
            rf'(?m)^version\s*=\s*"{escaped_old}"',
            f'version = "{new_version}"',
        ),
    ]

    updated = []
    for rel_path, pattern, repl in targets:
        full_path = root / rel_path
        if replace_in_file(full_path, pattern, repl, dry_run=dry_run):
            updated.append(rel_path)

    # Check CHANGELOG.md
    changelog_path = root / "CHANGELOG.md"
    if changelog_path.exists():
        changelog = changelog_path.read_text(encoding="utf-8")
        if f"## v{new_version}" not in changelog:
            print(f"Notice: CHANGELOG.md does not yet contain '## v{new_version}'.")
            if not dry_run:
                # Add placeholder under ## Unreleased
                unreleased_marker = "## Unreleased\n"
                if unreleased_marker in changelog:
                    placeholder = f"## Unreleased\n\n## v{new_version}\n\n### Summary\n\n- TBD\n"
                    changelog_path.write_text(changelog.replace(unreleased_marker, placeholder, 1), encoding="utf-8")
                    print(f"Added template entry '## v{new_version}' to CHANGELOG.md.")
                else:
                    print("Could not find '## Unreleased' in CHANGELOG.md; please add section manually.")

    print(f"\n{'Would update' if dry_run else 'Successfully updated'} {len(set(updated))} files.")
    print("\nNext steps:")
    print("  1. Update CHANGELOG.md with release notes.")
    print("  2. Run 'cargo check --workspace' and 'cargo test -p mltreelib-core'.")
    print("  3. Run 'uv lock' to update lockfile.")
    print(f"  4. Verify with 'uv run python scripts/prepare_release.py refs/tags/v{new_version}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump mltreelib version across all files.")
    parser.add_argument("version", help="New version (e.g. 0.4.15 or v0.4.15)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    new_ver = parse_version(args.version)
    bump_version(new_ver, root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
