"""Offline tests for the on-demand CSV sources (FLCHAIN, Hillstrom) in the dataset registry."""

import hashlib
import os
import urllib.request

import pytest
from mltreelib import datasets

CSV_NAMES = (
    "veteran",
    "pbc",
    "flchain",
    "flchain_rdatasets",
    "colon",
    "rott2",
    "gbsg2",
    "mgus2",
    "nsw",
    "jobs_ii",
    "hillstrom_email",
)


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
        meta = datasets.dataset_metadata(name)
        assert meta["source"] and meta["licence"] and meta["checksum_date"]
    assert "flchain_rdatasets" in datasets.list_available_datasets()
    assert "hillstrom_email" in datasets.list_available_datasets()
    assert "mgus2" in datasets.list_available_datasets()
    assert "nsw" in datasets.list_available_datasets()
    assert "jobs_ii" in datasets.list_available_datasets()


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


def test_offline_mode_forbids_download_but_allows_cache(isolated_dirs, monkeypatch):
    _, cache_dir = isolated_dirs
    monkeypatch.setenv("MLTREELIB_OFFLINE", "1")
    _no_network(monkeypatch)
    with pytest.raises(RuntimeError, match="MLTREELIB_OFFLINE"):
        datasets.dataset_path("mgus2")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "mgus2.csv"
    cached.write_text("rownames,id,age,sex,dxyr,hgb,creat,mspike,ptime,pstat,futime,death\n1,1,70,M,1990,12,1.1,1.0,100,1,200,1\n")
    assert datasets.dataset_path("mgus2") == cached


def test_checksum_mismatch_escape_hatch_warns(isolated_dirs, monkeypatch):
    _, cache_dir = isolated_dirs
    monkeypatch.setenv("MLTREELIB_ALLOW_CHECKSUM_MISMATCH", "1")

    def fake_retrieve(url, filename=None, *args, **kwargs):
        with open(filename, "wb") as handle:
            handle.write(b"updated upstream file")
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    with pytest.warns(UserWarning, match="sha256 mismatch"):
        path = datasets.download_dataset("veteran")
    assert path == cache_dir / "veteran.csv"
    assert path.exists()


def test_normalised_survival_and_uplift_loaders(isolated_dirs, monkeypatch):
    pytest.importorskip("pyarrow")
    data_dir, _ = isolated_dirs
    _no_network(monkeypatch)

    (data_dir / "mgus2.csv").write_text(
        "rownames,id,age,sex,dxyr,hgb,creat,mspike,ptime,pstat,futime,death\n"
        "1,1,70,M,1990,12,1.1,1.0,50,1,200,1\n"
        "2,2,65,F,1991,13,0.9,0.5,300,0,300,1\n"
        "3,3,60,M,1992,14,1.0,0.2,400,0,400,0\n"
    )
    surv = datasets.load_survival_dataset("mgus2")
    assert surv["task"] == "competing_risks"
    assert surv["time"] == [50.0, 300.0, 400.0]
    assert surv["cause"] == [1, 2, 0]
    assert surv["event"] == [1, 1, 0]
    assert "ptime" not in surv["features"] and "death" not in surv["features"]

    (data_dir / "colon.csv").write_text(
        "rownames,id,study,rx,sex,age,obstruct,perfor,adhere,nodes,status,differ,extent,surg,node4,time,etype\n"
        "1,10,1,Obs,1,55,0,0,0,2,1,2,3,0,0,120,1\n"
        "2,10,1,Obs,1,55,0,0,0,2,1,2,3,0,0,180,2\n"
    )
    colon = datasets.load_survival_dataset("colon")
    assert colon["group"] == [10, 10]
    assert colon["cause"] == [1, 2]

    (data_dir / "jobs_ii.csv").write_text(
        "rownames,treat,econ_hard,depress1,sex,age,occp,marital,nonwhite,educ,income,job_seek,depress2,work1,comply,control,job_dich,job_disc\n"
        "1,1,3,1.9,1,34,prof,married,white,gradwk,50k+,4.8,1.7,psyemp,1,treat,1,4\n"
        "2,1,2,2.1,0,40,prof,single,white,ba,30k,3.5,2.2,unemp,0,treat,0,2\n"
        "3,0,1,1.5,1,29,cler,single,white,hs,20k,4.0,1.4,unemp,0,ctrl,0,3\n"
    )
    up = datasets.load_uplift_dataset("jobs_ii", max_rows=2, seed=7)
    assert up["n_rows"] == 2
    assert up["randomized"] is True
    assert up["assignment"] == up["treatment"]
    assert len(up["receipt"]) == 2
    assert "comply" not in up["features"] and "work1" not in up["features"]
