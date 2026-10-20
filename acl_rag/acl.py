"""ACL lookup for LightRAG items (chunks, entities, relations).

ACL(c) for a chunk   = acl_users of its parent document in manifest.jsonl
                       (parent found through the chunk record's ``full_doc_id``).
ACL(v) for a node/edge = union of ACL(c) over the chunk ids in its ``source_id``
                       (LightRAG joins them with ``<SEP>``).

This is the "compute it yourself for now" version: it is derived at query time
from LightRAG's own stores plus our manifest. In Week 3 the same union will be
stored on nodes/edges at merge time; the two must agree.

Fail closed: a chunk whose document is not in the manifest (or whose record is
missing) has an empty ACL, so nobody is authorized for it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

GRAPH_FIELD_SEP = "<SEP>"  # lightrag.constants.GRAPH_FIELD_SEP in 1.5.7
EMPTY: frozenset[str] = frozenset()


def split_source_ids(source_id: str | None) -> list[str]:
    """Split a node/edge ``source_id`` into chunk ids (order kept, blanks dropped)."""
    if not source_id:
        return []
    return [s.strip() for s in source_id.split(GRAPH_FIELD_SEP) if s.strip()]


def edge_key(src: str, tgt: str) -> tuple[str, str]:
    """LightRAG graphs are undirected; normalise the pair."""
    return (src, tgt) if src <= tgt else (tgt, src)


class ACLIndex:
    """Answers "may user u see item v?" for one corpus + one LightRAG instance.

    Call :meth:`bind` with the *unpatched* storage readers before enforcement is
    installed, so ACL lookups never go through the ACL filters themselves.
    """

    def __init__(self, corpus_dir: str | Path):
        corpus_dir = Path(corpus_dir)
        self.corpus_dir = corpus_dir
        self.doc_users: dict[str, frozenset[str]] = {}
        with open(corpus_dir / "manifest.jsonl") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    self.doc_users[d["doc_id"]] = frozenset(d["acl_users"])
        oracle = json.loads((corpus_dir / "oracle_splits.json").read_text())
        self.user_to_access_set: dict[str, str] = dict(oracle["user_to_access_set"])
        self.access_sets: dict[str, dict[str, Any]] = oracle["access_sets"]

        # bound later
        self._get_chunks: Callable[[list[str]], Awaitable[list[dict | None]]] | None = None
        self._get_node: Callable[[str], Awaitable[dict | None]] | None = None
        self._get_edge: Callable[[str, str], Awaitable[dict | None]] | None = None

        # memo (valid for one ingested index; call clear_cache() after re-ingest)
        self._chunk_doc: dict[str, str | None] = {}
        self._node_acl: dict[str, frozenset[str]] = {}
        self._edge_acl: dict[tuple[str, str], frozenset[str]] = {}

    # ------------------------------------------------------------- binding
    def bind(
        self,
        get_chunks: Callable[[list[str]], Awaitable[list[dict | None]]],
        get_node: Callable[[str], Awaitable[dict | None]],
        get_edge: Callable[[str, str], Awaitable[dict | None]],
    ) -> "ACLIndex":
        self._get_chunks, self._get_node, self._get_edge = get_chunks, get_node, get_edge
        return self

    def clear_cache(self) -> None:
        self._chunk_doc.clear()
        self._node_acl.clear()
        self._edge_acl.clear()

    # ------------------------------------------------------------- principals
    def access_set_of(self, user_id: str) -> str:
        try:
            return self.user_to_access_set[user_id]
        except KeyError:
            raise KeyError(f"unknown user {user_id!r}; users come from oracle_splits.json") from None

    @property
    def users(self) -> list[str]:
        return list(self.user_to_access_set)

    # ------------------------------------------------------------- chunks
    def note_chunk_doc(self, chunk_id: str, doc_id: str | None) -> None:
        """Record a chunk->doc mapping seen elsewhere (e.g. a vector-store hit)."""
        if chunk_id and doc_id and chunk_id not in self._chunk_doc:
            self._chunk_doc[chunk_id] = doc_id

    async def chunk_docs(self, chunk_ids: Iterable[str]) -> dict[str, str | None]:
        ids = list(dict.fromkeys(c for c in chunk_ids if c))
        todo = [c for c in ids if c not in self._chunk_doc]
        if todo:
            if self._get_chunks is None:
                raise RuntimeError("ACLIndex.bind() was not called")
            rows = await self._get_chunks(todo)
            for cid, row in zip(todo, rows):
                doc = (row or {}).get("full_doc_id")
                if doc is None:
                    log.warning("chunk %s has no full_doc_id; treating as unreadable", cid)
                self._chunk_doc[cid] = doc
        return {c: self._chunk_doc[c] for c in ids}

    def doc_acl(self, doc_id: str | None) -> frozenset[str]:
        if doc_id is None:
            return EMPTY
        acl = self.doc_users.get(doc_id)
        if acl is None:
            log.warning("doc %s not in manifest; treating as unreadable", doc_id)
            return EMPTY
        return acl

    async def chunk_acls(self, chunk_ids: Iterable[str]) -> dict[str, frozenset[str]]:
        docs = await self.chunk_docs(chunk_ids)
        return {c: self.doc_acl(d) for c, d in docs.items()}

    async def union_acl(self, chunk_ids: Iterable[str]) -> frozenset[str]:
        acls = await self.chunk_acls(chunk_ids)
        out: set[str] = set()
        for a in acls.values():
            out |= a
        return frozenset(out)

    # ------------------------------------------------------------- nodes / edges
    async def node_acl(self, name: str, node: dict | None = None) -> frozenset[str]:
        """ACL of an entity. ``node`` may be passed when the caller already has it."""
        if name in self._node_acl:
            return self._node_acl[name]
        if node is None or "source_id" not in node:
            if self._get_node is None:
                raise RuntimeError("ACLIndex.bind() was not called")
            node = await self._get_node(name)
        acl = await self.union_acl(split_source_ids((node or {}).get("source_id")))
        self._node_acl[name] = acl
        return acl

    async def edge_acl(self, src: str, tgt: str, edge: dict | None = None) -> frozenset[str]:
        key = edge_key(src, tgt)
        if key in self._edge_acl:
            return self._edge_acl[key]
        if edge is None or "source_id" not in edge:
            if self._get_edge is None:
                raise RuntimeError("ACLIndex.bind() was not called")
            edge = await self._get_edge(src, tgt)
            if edge is None:
                edge = await self._get_edge(tgt, src)
        acl = await self.union_acl(split_source_ids((edge or {}).get("source_id")))
        self._edge_acl[key] = acl
        return acl

    async def node_source_chunks(self, name: str) -> list[str]:
        node = await self._get_node(name) if self._get_node else None
        return split_source_ids((node or {}).get("source_id"))
