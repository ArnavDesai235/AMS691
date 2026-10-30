"""Re-running ingest retries FAILED docs and never adds dup-* rows (real LightRAG 1.5.7).

Reproduces dev job 80486: an extraction timeout leaves a doc FAILED, and a plain
re-insert neither retries it nor leaves doc_status clean.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acl_rag import rag_factory
from acl_rag.ingest import ingest


class CharacterCodec:
    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def test_rerun_retries_failed_doc_without_duplicates(tmp_path, monkeypatch):
    import lightrag
    from lightrag.utils import EmbeddingFunc, Tokenizer

    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    facts = {"DOC_A": "Project Atlas is run by Team North.",
             "DOC_B": "Project Atlas launches in April with Team North.",
             "DOC_C": "Team North reports to Project Atlas leadership."}
    docs = [{"doc_id": d, "acl_users": ["u1"]} for d in facts]
    for d, f in facts.items():
        (corpus / "docs" / f"{d}.txt").write_text(f"{d}\n{f}")
    (corpus / "manifest.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (corpus / "oracle_splits.json").write_text(json.dumps(
        {"access_sets": {"AS0": {}}, "user_to_access_set": {"u1": "AS0"}}))
    (corpus / "facts.json").write_text(json.dumps({"entities": [
        {"entity_id": "E1", "canonical_name": "Project Atlas", "cls": "x", "hub": False},
        {"entity_id": "E2", "canonical_name": "Team North", "cls": "x", "hub": False}]}))

    broken = {"DOC_B"}
    calls = []

    async def llm(prompt, system_prompt=None, **kwargs):
        if "Description List:" in prompt:
            return " ".join(json.loads(l)["Description"] for l in prompt.splitlines()
                            if l.startswith('{"Description":'))
        for d, f in facts.items():
            if d in prompt:
                calls.append(d)
                if d in broken:
                    raise TimeoutError(f"{d}: Worker execution timeout after 480s")
                return json.dumps({"entities": [
                    {"name": "Project Atlas", "type": "PROJECT", "description": f},
                    {"name": "Team North", "type": "ORGANIZATION", "description": f}],
                    "relationships": [{"source": "Project Atlas", "target": "Team North",
                                       "description": f, "keywords": "project"}]})
        raise AssertionError(f"Unexpected model call: {prompt[:100]}")

    async def embed(texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

    monkeypatch.setattr(rag_factory, "_llm_and_embedding", lambda _: (
        llm, EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=embed)))
    real_rag = lightrag.LightRAG
    # LightRAG's JSON stores share in-process state per workspace; isolate from other tests.
    monkeypatch.setattr(lightrag, "LightRAG", lambda **kw: real_rag(
        **kw, workspace="ingest_resume_test", tokenizer=Tokenizer("test-characters", CharacterCodec()),
        entity_extraction_use_json=True, entity_extract_max_gleaning=0))
    cfg = {"corpus": corpus, "working_dir": tmp_path / "index",
           "results_dir": tmp_path / "results", "backend": {"llm": "test", "llm_model": "stub"},
           "lightrag": {"chunk_token_size": 1200, "chunk_overlap_token_size": 0},
           "c4": {"description_strategy": "join"}}
    status_file = cfg["working_dir"] / "ingest_resume_test" / "kv_store_doc_status.json"

    first = asyncio.run(ingest(cfg))
    assert [f["doc_id"] for f in first["failed_docs"]] == ["DOC_B"]

    # What the overnight re-runs did: a plain re-insert of every id adds dup-* rows.
    async def plain_reinsert():
        rag = await rag_factory.build_rag(cfg)
        try:
            ids = list(facts)
            await rag.ainsert([(corpus / "docs" / f"{d}.txt").read_text() for d in ids],
                              ids=ids, file_paths=ids)
        finally:
            await rag.finalize_storages()

    asyncio.run(plain_reinsert())
    assert any(k.startswith("dup-") for k in json.loads(status_file.read_text()))

    broken.clear()
    calls.clear()
    second = asyncio.run(ingest(cfg))
    assert second["resume"]["removed_duplicate_rows"] == 3
    assert second["failed_docs"] == []
    assert second["resume"]["retried_failed"] == ["DOC_B"]
    assert second["resume"]["already_processed"] == 2
    assert calls == ["DOC_B"]  # processed docs are not re-extracted
    rows = json.loads(status_file.read_text())
    assert sorted(rows) == ["DOC_A", "DOC_B", "DOC_C"]  # no dup-* rows
    assert {r["status"] for r in rows.values()} == {"processed"}
    node_src = second["entities"][0]["n_source_chunks"]
    assert node_src == 3

    # A third run on a complete index is a no-op.
    third = asyncio.run(ingest(cfg))
    assert third["resume"]["inserted"] == 0 and third["failed_docs"] == []
    assert sorted(json.loads(status_file.read_text())) == ["DOC_A", "DOC_B", "DOC_C"]


def test_truncated_extraction_is_reported(tmp_path, monkeypatch):
    """A token-capped extraction leaves the doc PROCESSED; ingest must still flag it."""
    import lightrag
    from lightrag.utils import EmbeddingFunc, Tokenizer, TruncatedResponse

    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    facts = {"DOC_A": "Project Atlas is run by Team North.",
             "DOC_B": "Project Atlas launches in April with Team North."}
    for d, f in facts.items():
        (corpus / "docs" / f"{d}.txt").write_text(f"{d}\n{f}")
    (corpus / "manifest.jsonl").write_text("\n".join(
        json.dumps({"doc_id": d, "acl_users": ["u1"]}) for d in facts))
    (corpus / "facts.json").write_text(json.dumps({"entities": [
        {"entity_id": "E1", "canonical_name": "Project Atlas", "cls": "x", "hub": False}]}))

    async def llm(prompt, system_prompt=None, **kwargs):
        if "Description List:" in prompt:
            return " ".join(json.loads(l)["Description"] for l in prompt.splitlines()
                            if l.startswith('{"Description":'))
        for d, f in facts.items():
            if d in prompt:
                out = json.dumps({"entities": [
                    {"name": "Project Atlas", "type": "PROJECT", "description": f}],
                    "relationships": []})
                return TruncatedResponse(out) if d == "DOC_B" else out
        raise AssertionError(f"Unexpected model call: {prompt[:100]}")

    async def embed(texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

    monkeypatch.setattr(rag_factory, "_llm_and_embedding", lambda _: (
        llm, EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=embed)))
    real_rag = lightrag.LightRAG
    monkeypatch.setattr(lightrag, "LightRAG", lambda **kw: real_rag(
        **kw, workspace="ingest_truncation_test", tokenizer=Tokenizer("test-characters", CharacterCodec()),
        entity_extraction_use_json=True, entity_extract_max_gleaning=0))
    cfg = {"corpus": corpus, "working_dir": tmp_path / "index",
           "results_dir": tmp_path / "results", "backend": {"llm": "test", "llm_model": "stub"},
           "lightrag": {"chunk_token_size": 1200, "chunk_overlap_token_size": 0},
           "c4": {"description_strategy": "join"}}

    report = asyncio.run(ingest(cfg))
    assert report["failed_docs"] == []  # LightRAG itself treats it as success
    assert [t["doc_id"] for t in report["truncated_docs"]] == ["DOC_B"]
    assert report["truncated_docs"][0]["events"] >= 1
