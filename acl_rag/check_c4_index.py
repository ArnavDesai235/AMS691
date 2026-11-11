"""Check C4's fragment filtering against known exposures in a saved index.

    python -m acl_rag.check_c4_index --rag-config configs/rag_full.yaml

Recompute the index report so the candidates match the current graph. Run the
real C4 filter in join mode, without retrieval, summarization, or generation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from .acl import ACLIndex, split_source_ids
from .c4 import C4DescriptionFilter
from .check_index import _jsonl, _load_graph, check
from .enforce import Mode, Scope
from .fragments import FragmentStore, MissingProvenanceError, NoAuthorizedFragments
from .rag_factory import fragment_path, load_config


def _flagged_cases(report):
    """Deduplicate cases that match both a description and its keywords."""
    for row in report["canaries"]["rows"]:
        for carrier in row["carriers"]:
            kind, key = carrier["kind"], carrier["key"]
            if kind == "edge":
                key = tuple(key)
            for user in carrier["visible_outside_acl"]:
                yield row["canary_id"], kind, key, user


def _recorded_fingerprint(index: Path) -> str | None:
    """Fragment-store fingerprint that ingest recorded for this index, if any."""
    path = index / "ingest_report.json"
    if not path.is_file():
        return None
    return ((json.loads(path.read_text(encoding="utf-8")).get("c4_provenance") or {})
            .get("fingerprint"))


async def audit(corpus: Path, index: Path, fragments: Path, hub_threshold: int = 8, *,
                allow_empty: bool = False, config_strategy: str | None = None) -> dict:
    report = check(corpus, index, hub_threshold)
    cases = dict.fromkeys(_flagged_cases(report))
    graph = _load_graph(index)
    chunks = json.loads((index / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    canaries = {row["canary_id"]: row for row in _jsonl(corpus / "canaries.jsonl")}
    patterns = {cid: re.compile(row["match_regex"]) for cid, row in canaries.items()}

    async def get_chunks(ids):
        return [chunks.get(cid) for cid in ids]

    async def get_node(name):
        return graph.nodes[name] if name in graph else None

    async def get_edge(src, dst):
        return graph.get_edge_data(src, dst)

    acl = ACLIndex(corpus).bind(get_chunks, get_node, get_edge)
    store = FragmentStore(fragments, corpus)
    evaluated = {}
    counts: Counter = Counter()
    issues = []
    if not cases and not allow_empty:
        # Zero exposures usually means a wrong or mismatched index, not a clean one:
        # a passing audit must have checked something.
        issues.append({"reason": "no_cases_checked",
                       "detail": "check_index found no exposed canaries; pass --allow-empty if expected"})
    recorded = _recorded_fingerprint(index)
    if recorded is not None and recorded != store.fingerprint():
        issues.append({"reason": "fragment_store_mismatch",
                       "detail": f"{fragments} does not match the fragments recorded at ingest"})
    try:
        filter_description = C4DescriptionFilter(store, strategy="join")
        for cid, kind, key, user in cases:
            pair = kind, key, user
            original = graph.nodes[key] if kind == "node" else graph.edges[key]
            if pair not in evaluated:
                scope = Scope(Mode.C4, user, acl.access_set_of(user))
                try:
                    result = await filter_description(kind, key, dict(original), scope, acl)
                    evaluated[pair] = "returned", result
                except NoAuthorizedFragments:
                    evaluated[pair] = "dropped", None
                except MissingProvenanceError as exc:
                    evaluated[pair] = "provenance_error", str(exc)
            status, result = evaluated[pair]
            counts[status] += 1
            context = {"canary_id": cid, "kind": kind, "item": key, "user": user}
            if status == "provenance_error":
                issues.append({**context, "reason": status, "detail": result})
            if status != "returned":
                continue

            sources = split_source_ids(result.get("source_id"))
            original_sources = set(split_source_ids(original.get("source_id")))
            if not sources or any(
                src not in original_sources or
                user not in acl.doc_users.get(chunks.get(src, {}).get("full_doc_id"), ())
                for src in sources
            ):
                issues.append({**context, "reason": "invalid_returned_sources"})

            fields = {f: result.get(f) or "" for f in ("description", "keywords", "entity_type")}
            if kind == "node":
                fields["name"] = key
            else:
                fields["source_name"], fields["target_name"] = key
            matches = [field for field, text in fields.items() if patterns[cid].search(text)]
            if matches:
                counts["restricted_code_remaining"] += 1
                issues.append({**context, "reason": "restricted_code_remaining", "fields": matches})
    finally:
        store.close()

    result = {
        "corpus": str(corpus),
        "index": str(index),
        "strategy": "join",
        "config_description_strategy": config_strategy,
        "distinct_codes_by_kind": dict(Counter(canaries[cid]["kind"] for cid in {c[0] for c in cases})),
        "cases_checked": len(cases),
        "distinct_item_user_pairs": len(evaluated),
        "returned_cases": counts["returned"],
        "dropped_cases": counts["dropped"],
        "provenance_error_cases": counts["provenance_error"],
        "restricted_code_remaining_cases": counts["restricted_code_remaining"],
        "issues": issues,
    }
    if config_strategy not in (None, "join"):
        result["note"] = (f"Experiments use description_strategy={config_strategy!r}: items at or above the "
                          "summary threshold get an LLM summary of these same fragments. This audit checks "
                          "fragment selection only, not those generated summaries.")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    ap.add_argument("--allow-empty", action="store_true",
                    help="pass when check_index finds no exposed canaries (default: fail)")
    args = ap.parse_args()
    cfg = load_config(args.rag_config)
    result = asyncio.run(audit(cfg["corpus"], cfg["working_dir"], fragment_path(cfg),
                               cfg.get("lightrag", {}).get("force_llm_summary_on_merge", 8),
                               allow_empty=args.allow_empty,
                               config_strategy=cfg.get("c4", {}).get("description_strategy")))
    print(json.dumps(result, indent=2))
    return int(bool(result["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
