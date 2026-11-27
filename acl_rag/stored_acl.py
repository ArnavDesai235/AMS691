"""Stored ACLs and canary origin chunks, derived offline from a built index.

    python -m acl_rag.stored_acl --rag-config configs/rag_full.yaml                    # write stored_acl.json
    python -m acl_rag.stored_acl --rag-config configs/rag_full.yaml --write-canaries   # also fill origin_chunks

Reads only files the ingest already produced (chunk_map.json, kv_store_text_chunks.json, the graphml)
plus the corpus manifest and canaries; no LLM, no GPU.

<working_dir>/stored_acl.json
  chunks  {chunk_id: {"doc": doc_id, "acl_users": [...]}}      ACL(c): the parent document's ACL
  nodes   {entity name: [users]}                               ACL(v) = union of ACL(c) over source_id
  edges   {"a|b" (sorted pair): [users]}
  Fail closed: a chunk whose parent is not in the manifest has an empty ACL.

`verify()` recomputes every node and edge ACL through the query-time `ACLIndex` (acl_rag/acl.py), the
path the enforcement code actually uses, and reports any disagreement. The stored form and the query-time
form must agree (DESIGN_enforcement.md).

origin_chunks (canaries.jsonl): the chunks of the canary's origin document whose text matches the canary's
regex. `check_canaries` also reports a canary that appears in a chunk of any *other* document (a stray
copy) or in no chunk at all (lost to chunking), either of which would invalidate leakage scoring.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .acl import ACLIndex, edge_key, split_source_ids


def chunk_table(chunk_map: dict[str, list[str]], doc_users: dict[str, Iterable[str]]) -> dict[str, dict]:
    out = {}
    for doc, cids in chunk_map.items():
        users = sorted(doc_users.get(doc, ()))
        for c in cids:
            out[c] = {"doc": doc, "acl_users": users}
    return out


def _union(chunk_ids: Iterable[str], chunks: dict[str, dict]) -> list[str]:
    s: set[str] = set()
    for c in chunk_ids:
        s.update(chunks.get(c, {}).get("acl_users", ()))
    return sorted(s)


def node_edge_acls(graph: Any, chunks: dict[str, dict]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    nodes = {n: _union(split_source_ids(d.get("source_id")), chunks) for n, d in graph.nodes(data=True)}
    edges = {"|".join(edge_key(a, b)): _union(split_source_ids(d.get("source_id")), chunks)
             for a, b, d in graph.edges(data=True)}
    return nodes, edges


def verify(graph: Any, chunks: dict[str, dict], stored_nodes: dict, stored_edges: dict,
           doc_users: dict[str, Iterable[str]], oracle: dict) -> list[str]:
    """Disagreements between the stored ACLs and the query-time ACLIndex; empty list = they agree."""
    idx = ACLIndex.__new__(ACLIndex)                      # bypass file loading; same logic, injected data
    idx.doc_users = {d: frozenset(u) for d, u in doc_users.items()}
    idx.user_to_access_set = dict(oracle.get("user_to_access_set", {}))
    idx.access_sets = oracle.get("access_sets", {})
    idx._chunk_doc, idx._node_acl, idx._edge_acl = {}, {}, {}

    async def get_chunks(ids):
        return [{"full_doc_id": chunks[i]["doc"]} if i in chunks else None for i in ids]

    async def get_node(n):
        return dict(graph.nodes[n]) if n in graph else None

    async def get_edge(a, b):
        return dict(graph.edges[a, b]) if graph.has_edge(a, b) else None

    idx.bind(get_chunks, get_node, get_edge)
    bad: list[str] = []

    async def run():
        for n, stored in stored_nodes.items():
            if sorted(await idx.node_acl(n)) != stored:
                bad.append(f"node {n}")
        for key, stored in stored_edges.items():
            a, b = key.split("|", 1)
            if sorted(await idx.edge_acl(a, b)) != stored:
                bad.append(f"edge {key}")
    asyncio.run(run())
    return bad


def locate_canaries(canaries: list[dict], chunk_text: dict[str, str], chunk_doc: dict[str, str]) -> dict:
    """origin_chunks per canary plus problems (canary in a foreign doc's chunk, or in no chunk)."""
    filled, stray, lost = {}, {}, []
    for c in canaries:
        rx = re.compile(c["match_regex"])
        hits = [cid for cid, t in chunk_text.items() if rx.search(t)]
        own = sorted(h for h in hits if chunk_doc.get(h) == c["origin_doc_id"])
        foreign = sorted(h for h in hits if chunk_doc.get(h) != c["origin_doc_id"])
        filled[c["canary_id"]] = own
        if foreign:
            stray[c["canary_id"]] = foreign
        if not own:
            lost.append(c["canary_id"])
    return {"origin_chunks": filled, "stray": stray, "lost": lost}


def run(cfg: dict, write_canaries: bool = False) -> dict:
    import networkx as nx  # noqa: PLC0415
    wd, corpus = Path(cfg["working_dir"]), Path(cfg["corpus"])
    chunk_map = json.loads((wd / "chunk_map.json").read_text())
    doc_users = {}
    for line in (corpus / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            doc_users[d["doc_id"]] = d["acl_users"]
    oracle = json.loads((corpus / "oracle_splits.json").read_text())
    chunks = chunk_table(chunk_map, doc_users)
    graph = nx.read_graphml(wd / "graph_chunk_entity_relation.graphml")
    nodes, edges = node_edge_acls(graph, chunks)
    unknown = sorted({c for _, d in graph.nodes(data=True) for c in split_source_ids(d.get("source_id"))
                      if c not in chunks})
    bad = verify(graph, chunks, nodes, edges, doc_users, oracle)
    (wd / "stored_acl.json").write_text(json.dumps({"chunks": chunks, "nodes": nodes, "edges": edges}))

    text = json.loads((wd / "kv_store_text_chunks.json").read_text())
    canaries = [json.loads(l) for l in (corpus / "canaries.jsonl").read_text().splitlines() if l.strip()]
    loc = locate_canaries(canaries, {k: v["content"] for k, v in text.items()},
                          {k: v["full_doc_id"] for k, v in text.items()})
    if write_canaries:
        for c in canaries:
            c["origin_chunks"] = loc["origin_chunks"][c["canary_id"]]
        (corpus / "canaries.jsonl").write_text("".join(json.dumps(c) + "\n" for c in canaries))
    report = {"chunks": len(chunks), "nodes": len(nodes), "edges": len(edges),
              "nodes_with_empty_acl": sum(not v for v in nodes.values()),
              "source_chunks_missing_from_chunk_map": unknown[:20], "n_missing": len(unknown),
              "disagreements_with_query_time_acl": bad[:20], "n_disagreements": len(bad),
              "canaries": len(canaries), "canaries_in_multiple_chunks": sum(len(v) > 1 for v in loc["origin_chunks"].values()),
              "canaries_lost": loc["lost"], "canaries_in_foreign_chunks": loc["stray"],
              "origin_chunks_written": write_canaries}
    (wd / "stored_acl_report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    ap.add_argument("--write-canaries", action="store_true", help="fill origin_chunks in the corpus canaries.jsonl")
    a = ap.parse_args()
    from .rag_factory import load_config  # noqa: PLC0415  (imports lightrag)
    r = run(load_config(a.rag_config), a.write_canaries)
    print(json.dumps(r, indent=1))
    if r["n_disagreements"] or r["canaries_lost"] or r["canaries_in_foreign_chunks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
