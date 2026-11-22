"""Tests for release and version synchronization scripts."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bump_version import parse_version
from scripts.prepare_release import verify_release


def test_parse_version_valid():
    """Valid semver strings parse properly."""
    assert parse_version("0.4.15") == "0.4.15"
    assert parse_version("v0.4.15") == "0.4.15"
    assert parse_version("1.0.0") == "1.0.0"


def test_parse_version_invalid():
    """Invalid version strings raise SystemExit."""
    with pytest.raises(SystemExit):
        parse_version("0.4.14a")
    with pytest.raises(SystemExit):
        parse_version("invalid")


def test_verify_release_on_current_repo():
    """verify_release passes cleanly on the current repo version."""
    root = Path(__file__).resolve().parents[1]
    changelog_entry = verify_release(root, "0.4.14")
    assert changelog_entry
    assert "Standalone optimization solvers" in changelog_entry


def test_verify_release_catches_mismatch():
    """verify_release raises SystemExit when version does not match."""
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(SystemExit) as exc:
        verify_release(root, "0.4.99")
    assert "Release verification failed for v0.4.99" in str(exc.value)
