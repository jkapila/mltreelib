"""Offline tests for the on-demand CSV sources (FLCHAIN, Hillstrom) in the dataset registry."""

import hashlib
import os
import urllib.request

import pytest
from mltreelib import datasets

CSV_NAMES = ("flchain_rdatasets", "hillstrom_email")


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "repo_datasets"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    monkeypatch.setattr(datasets, "_DATA_DIR", data_dir)
    monkeypatch.setattr(datasets, "_CACHE_DIR", cache_dir)
    return data_dir, cache_dir


def _no_network(monkeypatch):
    def fail(url, filename=None, *args, **kwargs):
        raise AssertionError(f"unexpected download from {url}")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail)


def test_csv_sources_are_registered_with_checksums():
    for name in CSV_NAMES:
        spec = datasets.DATASET_REGISTRY[name]
        assert isinstance(spec, str) and spec.startswith("http") and spec.endswith(".csv")
        digest = datasets.DATASET_SHA256[name]
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert "flchain_rdatasets" in datasets.list_available_datasets()
    assert "hillstrom_email" in datasets.list_available_datasets()


def test_dataset_path_uses_cached_csv_without_download(isolated_dirs, monkeypatch):
    _, cache_dir = isolated_dirs
    cache_dir.mkdir(parents=True)
    cached = cache_dir / "hillstrom_email.csv"
    cached.write_text("recency,visit\n1,0\n")
    _no_network(monkeypatch)
    assert datasets.dataset_path("hillstrom_email") == cached


def test_download_writes_csv_via_part_file_and_verifies_sha256(isolated_dirs, monkeypatch):
    _, cache_dir = isolated_dirs
    payload = b"recency,visit\n1,0\n2,1\n"
    monkeypatch.setitem(datasets.DATASET_SHA256, "hillstrom_email", hashlib.sha256(payload).hexdigest())
    seen = {}

    def fake_retrieve(url, filename=None, *args, **kwargs):
        seen["url"] = url
        seen["part"] = str(filename)
        with open(filename, "wb") as handle:
            handle.write(payload)
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    path = datasets.dataset_path("hillstrom_email")
    assert path == cache_dir / "hillstrom_email.csv"
    assert path.read_bytes() == payload
    assert seen["url"] == datasets.DATASET_REGISTRY["hillstrom_email"]
    assert seen["part"].endswith("hillstrom_email.csv.part")
    assert not any(cache_dir.glob("*.part"))


def test_checksum_mismatch_keeps_nothing(isolated_dirs, monkeypatch):
    _, cache_dir = isolated_dirs

    def fake_retrieve(url, filename=None, *args, **kwargs):
        with open(filename, "wb") as handle:
            handle.write(b"not the published file")
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        datasets.download_dataset("flchain_rdatasets")
    assert not (cache_dir / "flchain_rdatasets.csv").exists()
    assert not any(cache_dir.glob("*.part"))


def test_load_dataset_reads_csv_with_missing_values(isolated_dirs, monkeypatch):
    pytest.importorskip("pyarrow")
    data_dir, _ = isolated_dirs
    (data_dir / "flchain_rdatasets.csv").write_text(
        "rownames,age,sex,creatinine,chapter\n1,97,F,1.7,\n2,50,M,NA,Circulatory\n"
    )
    _no_network(monkeypatch)
    ds = datasets.load_dataset("flchain_rdatasets")
    assert ds["n_rows"] == 2
    assert ds["columns"]["age"] == [97, 50]
    assert ds["dtypes"]["sex"] == "categorical"
    assert ds["columns"]["creatinine"] == [1.7, None]  # NA is missing
    assert ds["columns"]["chapter"] == [None, "Circulatory"]  # empty string is missing
    assert ds["na_counts"]["creatinine"] == 1


@pytest.mark.skipif(
    os.environ.get("MLTREELIB_NETWORK_TESTS") != "1",
    reason="live download test; set MLTREELIB_NETWORK_TESTS=1 to run",
)
def test_live_hillstrom_download_matches_registered_checksum(tmp_path):
    path = datasets.download_dataset("hillstrom_email", target_dir=tmp_path)
    assert path.name == "hillstrom_email.csv"
    assert datasets._sha256(path) == datasets.DATASET_SHA256["hillstrom_email"]
