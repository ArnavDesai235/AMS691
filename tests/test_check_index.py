"""check_index on an index written from the dev corpus ground truth."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from fake_lightrag import FakeRAG, import_operate
from acl_rag.check_index import check, summary

CORPUS = ROOT / "corpus/dev_cf0.3_s42"


def write_index(wd: Path, *, rename=None, canary_node=None, drop_doc=None, leak=None):
    rag = FakeRAG(CORPUS, import_operate())
    g = nx.Graph()
    for n, d in rag.chunk_entity_relation_graph.nodes.items():
        g.add_node(rename.get(n, n) if rename else n,
                   **{k: str(v) for k, v in d.items() if k != "created_at"})
    for (s, t), d in rag.chunk_entity_relation_graph.edges.items():
        if rename:
            s, t = rename.get(s, s), rename.get(t, t)
        g.add_edge(s, t, **{k: str(v) for k, v in d.items() if k not in ("src_id", "tgt_id")})
    if canary_node:
        g.add_node(canary_node, description="a budget code", source_id="chunk-d-team_a-007")
    if leak:  # merged description of a public node carries a restricted canary
        name, code = leak
        g.nodes[name]["description"] += f"<SEP>{name} uses code {code}."
    chunks = {cid: {"full_doc_id": r["full_doc_id"], "content": ""}
              for cid, r in rag.text_chunks._data.items()} if hasattr(rag.text_chunks, "_data") else None
    if chunks is None:
        chunks = {f"chunk-{json.loads(l)['doc_id']}": {"full_doc_id": json.loads(l)["doc_id"]}
                  for l in (CORPUS / "manifest.jsonl").read_text().splitlines()}
    status = {c["full_doc_id"]: {"status": "processed"} for c in chunks.values()}
    if drop_doc:
        status[drop_doc] = {"status": "failed"}
    wd.mkdir(parents=True)
    nx.write_graphml(g, wd / "graph_chunk_entity_relation.graphml")
    (wd / "kv_store_text_chunks.json").write_text(json.dumps(chunks))
    (wd / "kv_store_doc_status.json").write_text(json.dumps(status))


def test_ground_truth_index_matches_plan(tmp_path):
    write_index(tmp_path / "ix")
    r = check(CORPUS, tmp_path / "ix")
    assert r["entities"]["found"] == r["entities"]["planned"] == 55
    assert r["docs"]["not_processed"] == []
    assert r["cross_tier"]["planned_not_mixed"] == []
    assert abs(r["cross_tier"]["realized_cf"] - 0.3) < 0.01  # 8 / 27
    cb = r["canaries"]["by_kind"]
    assert cb["structural"]["exposed_via_merge"] == 0
    assert cb["contamination"]["exposed_via_merge"] == cb["contamination"]["in_index"] > 0
    assert r["canaries"]["named_nodes"] == {}
    summary(r)


def test_detects_renames_canary_nodes_failed_docs_and_leaks(tmp_path):
    facts = json.loads((CORPUS / "facts.json").read_text())
    cross = next(e["canonical_name"] for e in facts["entities"] if e["cls"] == "cross_tier")
    public = next(e["canonical_name"] for e in facts["entities"] if e["cls"] == "public_only")
    write_index(tmp_path / "ix", rename={cross: cross + " Inc."}, canary_node="SP-9062-V",
                drop_doc="d-team_a-007", leak=(public, "SP-9062-V"))
    r = check(CORPUS, tmp_path / "ix")
    assert r["entities"]["missing"] == [{"name": cross, "near_matches": [cross + " Inc."]}]
    assert cross in r["cross_tier"]["planned_not_mixed"]
    assert r["canaries"]["named_nodes"] == {"C000": ["SP-9062-V"]}
    assert "d-team_a-007" in r["docs"]["not_processed"]
    c000 = next(x for x in r["canaries"]["rows"] if x["canary_id"] == "C000")
    assert not c000["origin_doc_processed"] and c000["exposed_via_merge"]
    assert "u_public_1" in next(x for x in c000["carriers"] if x["key"] == public)["visible_outside_acl"]
