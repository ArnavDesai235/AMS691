"""Ingest a corpus build into stock LightRAG (C1 index; C2-C4 query the same index).

    python -m acl_rag.ingest --rag-config configs/rag.yaml

Each document goes in with ``ainsert(text, ids=doc_id, file_paths=doc_id)`` so
every chunk records its parent doc (``full_doc_id``) and every retrieved chunk
carries the doc id as its ``file_path``.

Writes into the index directory:
  chunk_map.json      doc_id -> [chunk ids]   (for origin_chunks / ACL checks)
  ingest_report.json  per-entity merge check: does a node exist under the exact
                      canonical name, how many source chunks it has, and the
                      source-count hub proxy (>= 8 chunks vs < 8).
  c4_fragments.sqlite3 original entity/relation descriptions and source ACLs,
                      captured before merging (unless disabled in config).

The source-count proxy is not an observation of whether an LLM summary ran;
LightRAG also applies token limits and may merge prior summaries incrementally.

Re-running on an existing index resumes it. LightRAG 1.5.7 never retries FAILED
docs on insert, and re-inserting a known doc id only adds a FAILED ``dup-*``
status row. So before inserting, this script deletes FAILED docs (and any
``dup-*`` rows) and then inserts only docs that are not PROCESSED. Extraction
results already in the LLM cache are reused.

LightRAG 1.5.7 marks a doc PROCESSED even when an extraction response hit the
output-token cap (Ollama ``done_reason == "length"``); it only records this in
``doc_status.metadata.llm_truncation``. Those docs are listed as
``truncated_docs`` and fail the run, since their graph may be missing records.

The LLM cache key covers role and model only, not num_predict / temperature /
seed: after changing those options, re-ingest into a fresh working_dir.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .acl import split_source_ids

LLM_TRUNCATION_METADATA_KEY = "llm_truncation"  # lightrag.utils, 1.5.7
from .rag_factory import build_rag, load_config

log = logging.getLogger("acl_rag.ingest")


def read_docs(corpus: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(corpus / "manifest.jsonl") as fh:
        for line in fh:
            if line.strip():
                doc_id = json.loads(line)["doc_id"]
                ids.append(doc_id)
                texts.append((corpus / "docs" / f"{doc_id}.txt").read_text())
    return ids, texts


def _status(row: dict | None) -> str:
    return str((row or {}).get("status", "")).lower().rsplit(".", 1)[-1]


def _truncation(row: dict | None) -> dict | None:
    return ((row or {}).get("metadata") or {}).get(LLM_TRUNCATION_METADATA_KEY)


async def prepare_resume(rag, ids: list[str], *, retry_truncated: bool = False) -> tuple[list[str], dict]:
    """Clear failed (and optionally truncated) state so it can be retried; return doc ids to insert.

    Truncated extraction responses are never written to the LLM cache, so a
    retry re-extracts only the truncated chunks; it is useful after raising
    backend.ollama_num_predict (at temperature 0 the same cap truncates again).
    """
    from lightrag.base import DocStatus  # noqa: PLC0415

    failed_rows = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
    ours = set(ids)
    dup = [k for k in failed_rows if k.startswith("dup-") and k not in ours]
    if dup:
        await rag.doc_status.delete(dup)
        await rag.doc_status.index_done_callback()
    statuses = dict(zip(ids, await rag.doc_status.get_by_ids(ids)))
    retry = [d for d in ids if _status(statuses[d]) == "failed"
             or (retry_truncated and _truncation(statuses[d]))]
    for doc_id in retry:
        res = await rag.adelete_by_doc_id(doc_id)
        if res.status not in ("success", "not_found"):
            raise RuntimeError(f"could not clear failed doc {doc_id}: {res.status} {res.message}")
    todo = [d for d in ids if d in retry or _status(statuses[d]) != "processed"]
    info = {"already_processed": len(ids) - len(todo), "retried_failed": retry,
            "retry_truncated": retry_truncated,
            "removed_duplicate_rows": len(dup), "inserted": len(todo)}
    return todo, info


async def ingest(cfg: dict, *, retry_truncated: bool = False) -> dict:
    corpus: Path = cfg["corpus"]
    ids, texts = read_docs(corpus)
    rag = await build_rag(cfg, capture_fragments=True)
    try:
        todo, resume = await prepare_resume(rag, ids, retry_truncated=retry_truncated)
        log.info("resume: %s", resume)
        if todo:
            text_of = dict(zip(ids, texts))
            log.info("inserting %d docs from %s into %s", len(todo), corpus, cfg["working_dir"])
            await rag.ainsert([text_of[d] for d in todo], ids=todo, file_paths=todo)

        # ---- chunk map + doc status
        chunk_map, failed, truncated = {}, [], []
        for doc_id in ids:
            st = await rag.doc_status.get_by_id(doc_id) or {}
            status = str(st.get("status", "")).lower()
            if "processed" not in status:
                failed.append({"doc_id": doc_id, "status": status, "error": st.get("error_msg")})
            trunc = _truncation(st)
            if trunc:
                truncated.append({"doc_id": doc_id, **trunc})
            chunk_map[doc_id] = list(st.get("chunks_list") or [])

        # every chunk must point back to its own doc
        bad_parent = []
        for doc_id, cids in chunk_map.items():
            rows = await rag.text_chunks.get_by_ids(cids) if cids else []
            for cid, row in zip(cids, rows):
                if (row or {}).get("full_doc_id") != doc_id:
                    bad_parent.append(cid)

        # ---- merge check on our canonical names
        facts = json.loads((corpus / "facts.json").read_text())
        # The planner's regime prediction lives in plan.json; facts.json's `hub`
        # flag is only enforced for cross-tier entities (restricted-only entities
        # can carry hub=True while being planned into the concatenated regime).
        plan_file = corpus / "plan.json"
        predicted = json.loads(plan_file.read_text()).get("predicted_regime", {}) if plan_file.is_file() else {}
        graph = rag.chunk_entity_relation_graph
        threshold = cfg.get("lightrag", {}).get("force_llm_summary_on_merge", 8)
        entities, missing = [], []
        for e in facts["entities"]:
            node = await graph.get_node(e["canonical_name"])
            n_src = len(split_source_ids((node or {}).get("source_id")))
            row = {"entity_id": e["entity_id"], "name": e["canonical_name"], "cls": e["cls"],
                   "planned_hub": e.get("hub"), "predicted_regime": predicted.get(e["entity_id"]),
                   "found": node is not None,
                   "n_source_chunks": n_src,
                   "realized_regime": "hub" if n_src >= threshold else "non_hub"}
            entities.append(row)
            if node is None:
                missing.append(e["canonical_name"])

        report = {
            "corpus": str(corpus), "n_docs": len(ids), "resume": resume, "failed_docs": failed,
            "truncated_docs": truncated,
            "n_chunks": sum(len(v) for v in chunk_map.values()),
            "chunks_with_wrong_parent": bad_parent,
            "entities_found": sum(r["found"] for r in entities), "entities_total": len(entities),
            "entities_missing": missing,
            "realized_hubs": sum(r["realized_regime"] == "hub" for r in entities),
            "hub_prediction_mismatches": [
                {"name": r["name"], "cls": r["cls"], "predicted": r["predicted_regime"],
                 "n_source_chunks": r["n_source_chunks"]}
                for r in entities if r["found"] and (r["realized_regime"] == "hub") != (
                    r["predicted_regime"] == "summarized" if r["predicted_regime"] else bool(r["planned_hub"]))],
            "entities": entities,
            "c4_provenance": rag._c4_fragment_store.stats()
                if hasattr(rag, "_c4_fragment_store") else None,
        }
        wd: Path = cfg["working_dir"]
        (wd / "chunk_map.json").write_text(json.dumps(chunk_map, indent=1))
        (wd / "ingest_report.json").write_text(json.dumps(report, indent=1))
        return report
    finally:
        await rag.finalize_storages()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    ap.add_argument("--retry-truncated", action="store_true",
                    help="delete and re-insert docs whose extraction hit the output-token cap")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    r = asyncio.run(ingest(load_config(args.rag_config), retry_truncated=args.retry_truncated))
    print(json.dumps({k: v for k, v in r.items() if k != "entities"}, indent=1))
    if r["failed_docs"] or r["truncated_docs"] or r["chunks_with_wrong_parent"]:
        raise SystemExit("ingest finished with errors (see above)")


if __name__ == "__main__":
    main()
