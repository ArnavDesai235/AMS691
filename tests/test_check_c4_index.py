"""check_c4_index on the committed dev index, plus its guards."""
import asyncio
import json
from pathlib import Path

import pytest

import acl_rag.check_c4_index as audit_mod

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "dev_cf0.3_s42"
INDEX = ROOT / "rag_storage" / "dev_cf0.3_s42"
FRAGMENTS = INDEX / "c4_fragments.sqlite3"

pytestmark = pytest.mark.skipif(not FRAGMENTS.is_file(), reason="committed dev index not present")


def run(index=INDEX, **kw):
    return asyncio.run(audit_mod.audit(CORPUS, index, FRAGMENTS, **kw))


def test_dev_index_clean():
    r = run(config_strategy="auto")
    assert r["cases_checked"] > 0
    assert r["returned_cases"] + r["dropped_cases"] == r["cases_checked"]
    assert r["restricted_code_remaining_cases"] == 0
    assert r["issues"] == []
    assert "fragment selection only" in r["note"]


def test_no_note_for_join():
    assert "note" not in run(config_strategy="join")


def test_zero_cases_fails_unless_allowed(monkeypatch):
    real = audit_mod.check

    def no_exposures(*a, **kw):
        rep = real(*a, **kw)
        rep["canaries"]["rows"] = []
        return rep

    monkeypatch.setattr(audit_mod, "check", no_exposures)
    assert [i["reason"] for i in run()["issues"]] == ["no_cases_checked"]
    assert run(allow_empty=True)["issues"] == []


def test_fragment_store_mismatch(tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    for f in INDEX.iterdir():
        if f.is_file() and f.suffix in (".json", ".graphml"):
            (index / f.name).symlink_to(f)
    (index / "ingest_report.json").unlink()
    report = json.loads((INDEX / "ingest_report.json").read_text())
    report["c4_provenance"]["fingerprint"] = "0" * 64
    (index / "ingest_report.json").write_text(json.dumps(report))
    reasons = [i["reason"] for i in run(index=index)["issues"]]
    assert reasons == ["fragment_store_mismatch"]
