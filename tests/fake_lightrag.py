"""In-memory stand-ins for LightRAG 1.5.7 storages, built from a corpus build.

Lets the ACL tests drive LightRAG's *real* query code (lightrag.operate.kg_query)
without an LLM, embeddings or an ingest run. The graph mimics what ingest would
produce on our corpus: one chunk per document, one node per canonical entity name
whose source_id lists every chunk that mentions it and whose description joins
the fact sentences from those chunks with <SEP> (so a cross-tier node really does
carry restricted text), and one edge per entity-valued fact.

Vector search is replaced by name/word overlap with the query keywords.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

SEP = "<SEP>"


def import_operate():
    """Import lightrag.operate; fall back to a source checkout (LIGHTRAG_SRC) with
    stubs for modules the query path does not use."""
    try:
        return importlib.import_module("lightrag.operate")
    except ModuleNotFoundError:
        src = os.environ.get("LIGHTRAG_SRC")
        if not src:
            raise
        sys.path.insert(0, src)
        for _ in range(10):
            try:
                return importlib.import_module("lightrag.operate")
            except ModuleNotFoundError as e:
                stub = types.ModuleType(e.name)
                stub.__path__ = []
                stub.__getattr__ = lambda name: type(name, (), {})
                sys.modules[e.name] = stub
                for k in [k for k in sys.modules if k.startswith("lightrag")]:
                    del sys.modules[k]
        raise


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class FakeVDB:
    def __init__(self, rows: list[dict], text_key: str):
        self.rows = rows
        self.text_key = text_key
        self.cosine_better_than_threshold = 0.2
        self.calls: list[int] = []

    async def query(self, query, top_k, query_embedding=None):
        self.calls.append(top_k)
        q = _words(query)
        scored = []
        for i, r in enumerate(self.rows):
            s = len(q & _words(r[self.text_key]))
            if s:
                scored.append((-s, i, r))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [dict(r) for _, _, r in scored[:top_k]]

    async def get_vectors_by_ids(self, ids):
        return {i: [0.0] for i in ids}


class FakeKV:
    def __init__(self, rows: dict[str, dict], global_config: dict):
        self.rows = rows
        self.global_config = global_config
        self.embedding_func = None

    async def get_by_ids(self, ids):
        return [dict(self.rows[i]) if i in self.rows else None for i in ids]

    async def get_by_id(self, i):
        return dict(self.rows[i]) if i in self.rows else None

    async def upsert(self, data):
        self.rows.update(data)


class FakeGraph:
    def __init__(self, nodes: dict[str, dict], edges: dict[tuple[str, str], dict]):
        self.nodes = nodes
        self.edges = edges
        self.adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for s, t in edges:
            self.adj[s].append((s, t))
            self.adj[t].append((s, t))

    def _edge(self, s, t):
        return self.edges.get((s, t)) or self.edges.get((t, s))

    async def get_node(self, n):
        return dict(self.nodes[n]) if n in self.nodes else None

    async def get_edge(self, s, t):
        e = self._edge(s, t)
        return dict(e) if e else None

    async def get_nodes_batch(self, ids):
        return {i: dict(self.nodes[i]) for i in ids if i in self.nodes}

    async def node_degrees_batch(self, ids):
        return {i: len(self.adj.get(i, [])) for i in ids}

    async def get_nodes_edges_batch(self, ids):
        return {i: list(self.adj.get(i, [])) for i in ids}

    async def get_edges_batch(self, pairs):
        out = {}
        for p in pairs:
            e = self._edge(p["src"], p["tgt"])
            if e:
                out[(p["src"], p["tgt"])] = dict(e)
        return out

    async def edge_degrees_batch(self, pairs):
        return {p: len(self.adj.get(p[0], [])) + len(self.adj.get(p[1], [])) for p in pairs}


class WordTokenizer:
    def encode(self, s):
        return [hash(w) % 50000 for w in s.split()]

    def decode(self, toks):
        return " ".join("w" for _ in toks)


class FakeRAG:
    """Exposes the attributes acl_rag.enforce.install() expects of a LightRAG."""

    def __init__(self, corpus: Path, operate):
        from lightrag.utils import Tokenizer  # noqa: PLC0415

        corpus = Path(corpus)
        facts = json.loads((corpus / "facts.json").read_text())
        name = {e["entity_id"]: e["canonical_name"] for e in facts["entities"]}
        etype = {e["entity_id"]: e["type"] for e in facts["entities"]}
        fact = {f["fact_id"]: f for f in facts["facts"]}
        self.llm_calls: list[str] = []

        async def llm(prompt, system_prompt=None, **kw):
            self.llm_calls.append(prompt)
            codes = sorted(set(re.findall(r"\b[A-Z]{2}-\d{4}-[A-Z]\b", system_prompt or "")))
            return "Context mentions: " + (", ".join(codes) if codes else "no codes")

        self.global_config = {
            "tokenizer": Tokenizer("words", WordTokenizer()),
            "role_llm_funcs": {"query": llm, "keyword": llm},
            "llm_model_name": "fake",
            "kg_chunk_pick_method": "WEIGHT",
            "related_chunk_number": 5,
            "enable_llm_cache": True,
            "max_total_tokens": 300000,
            "max_entity_tokens": 100000,
            "max_relation_tokens": 100000,
        }
        chunks, nodes, edges = {}, {}, {}
        node_src, node_desc = defaultdict(list), defaultdict(list)
        with open(corpus / "manifest.jsonl") as fh:
            for line in fh:
                d = json.loads(line)
                cid = f"chunk-{d['doc_id']}"
                text = (corpus / "docs" / f"{d['doc_id']}.txt").read_text()
                chunks[cid] = {"content": text, "full_doc_id": d["doc_id"], "file_path": d["doc_id"],
                               "chunk_order_index": 0, "tokens": len(text.split())}
                for eid in d["entities"]:
                    node_src[eid].append(cid)
                    sents = [fact[f]["sentence"] for f in d["fact_ids"]
                             if eid in (fact[f].get("mentions") or [])]
                    if sents:
                        node_desc[eid].append(" ".join(sents))
                for fid in d["fact_ids"]:
                    f = fact[fid]
                    other = f["object"].get("entity")
                    if not other:
                        continue
                    key = (name[f["subject"]], name[other])
                    e = edges.setdefault(key, {"src_id": key[0], "tgt_id": key[1], "description": [],
                                               "keywords": f["predicate"], "weight": 1.0,
                                               "source_id": [], "file_path": d["doc_id"]})
                    if cid not in e["source_id"]:
                        e["source_id"].append(cid)
                    e["description"].append(f["sentence"])
        for eid, srcs in node_src.items():
            nodes[name[eid]] = {"entity_id": name[eid], "entity_type": etype[eid],
                                "description": SEP.join(node_desc[eid]) or name[eid],
                                "source_id": SEP.join(srcs), "file_path": SEP.join(c[6:] for c in srcs),
                                "created_at": 0}
        for e in edges.values():
            e["source_id"] = SEP.join(e["source_id"])
            e["description"] = SEP.join(dict.fromkeys(e["description"]))

        self.text_chunks = FakeKV(chunks, self.global_config)
        self.chunk_entity_relation_graph = FakeGraph(nodes, edges)
        self.entities_vdb = FakeVDB([{"entity_name": n, "content": n, "created_at": 0} for n in nodes],
                                    "content")
        self.relationships_vdb = FakeVDB([{"src_id": s, "tgt_id": t, "content": f"{s} {t}", "created_at": 0}
                                          for s, t in edges], "content")
        self.chunks_vdb = FakeVDB([{"id": c, "content": r["content"], "full_doc_id": r["full_doc_id"],
                                    "file_path": r["file_path"], "created_at": 0} for c, r in chunks.items()],
                                  "content")
        self.llm_response_cache = FakeKV({}, self.global_config)
        self.operate = operate

    async def query(self, text: str, keywords: list[str], mode: str = "mix", top_k: int = 10,
                    chunk_top_k: int = 5):
        """Run LightRAG's real kg_query with explicit keywords (no keyword LLM call)."""
        from lightrag.base import QueryParam  # noqa: PLC0415
        from lightrag.utils import convert_to_user_format  # noqa: F401,PLC0415

        param = QueryParam(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, enable_rerank=False,
                           ll_keywords=list(keywords), hl_keywords=list(keywords))
        return await self.operate.kg_query(
            text, self.chunk_entity_relation_graph, self.entities_vdb, self.relationships_vdb,
            self.text_chunks, param, self.global_config, hashing_kv=self.llm_response_cache,
            chunks_vdb=self.chunks_vdb)
