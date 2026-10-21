"""Check a built LightRAG index against the corpus plan before running C1-C4.

    python -m acl_rag.check_index --rag-config configs/rag.yaml
    python -m acl_rag.check_index --rag-config configs/rag.yaml --json out.json

Reads the index files directly (graphml, text chunks, doc status), so it needs
no model and runs on the login node. Reports:

docs          status counts; any doc not PROCESSED
entities      planned entities found under their exact canonical name; for
              missing ones, graph nodes whose name matches case-insensitively
              or contains the name (extractor naming variants)
cross_tier    a node is *mixed* if its source chunks come from docs with more
              than one ACL. Planned cross-tier entities that did not end up
              mixed (no contamination possible), unplanned mixed nodes, and the
              realized cross-tier fraction = mixed / nodes with any restricted
              source (over planned entities, same definition as the corpus cf)
hubs          nodes with >= force_llm_summary_on_merge source chunks
canaries      for each canary: nodes and edges whose description or keywords
              match it, and whether any carrier is visible to a user outside
              the canary's ACL (carrier ACL = union of source-chunk ACLs, as
              in C3). Also nodes whose *name* matches a canary (the extractor
              turned the code into an entity)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .acl import split_source_ids
from .rag_factory import load_config

PUBLIC = "public"


def _load_graph(wd: Path):
    import networkx as nx  # noqa: PLC0415

    path = wd / "graph_chunk_entity_relation.graphml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has ingest run for this corpus?")
    return nx.read_graphml(path)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def check(corpus: Path, wd: Path, hub_threshold: int = 8) -> dict:
    manifest = {d["doc_id"]: d for d in _jsonl(corpus / "manifest.jsonl")}
    facts = json.loads((corpus / "facts.json").read_text())
    canaries = _jsonl(corpus / "canaries.jsonl")
    chunks = json.loads((wd / "kv_store_text_chunks.json").read_text())
    status = json.loads((wd / "kv_store_doc_status.json").read_text())
    g = _load_graph(wd)

    chunk_doc = {cid: (row or {}).get("full_doc_id") for cid, row in chunks.items()}

    def sources(data: dict) -> list[str]:
        return [chunk_doc.get(c) for c in split_source_ids(data.get("source_id"))]

    def labels(docs) -> set[str]:
        return {manifest[d]["acl_label"] if d in manifest else "?" for d in docs}

    def users(docs) -> set[str]:
        out: set[str] = set()
        for d in docs:
            out |= set(manifest.get(d, {}).get("acl_users", []))
        return out

    # ---- docs
    st = Counter(str(v.get("status")) for v in status.values())
    not_done = sorted(k for k, v in status.items() if v.get("status") != "processed")
    missing_docs = sorted(set(manifest) - set(status))

    # ---- nodes
    nodes = {n: dict(d) for n, d in g.nodes(data=True)}
    node_docs = {n: sources(d) for n, d in nodes.items()}
    node_labels = {n: labels(ds) for n, ds in node_docs.items()}

    planned = {e["canonical_name"]: e for e in facts["entities"]}
    lower = {n.lower(): n for n in nodes}
    entities, missing = [], []
    for name, e in planned.items():
        found = name in nodes
        row = {"name": name, "cls": e["cls"], "planned_hub": bool(e.get("hub")), "found": found}
        if found:
            row["n_source_chunks"] = len(node_docs[name])
            row["n_docs"] = len(set(node_docs[name]))
            row["acl_labels"] = sorted(node_labels[name])
            row["mixed"] = len(node_labels[name]) > 1
        else:
            near = ({lower[name.lower()]} if name.lower() in lower else set()) | {
                n for n in nodes if name.lower() in n.lower() and n != name}
            near = sorted(near)
            row["near_matches"] = near[:5]
            missing.append(name)
        entities.append(row)

    by = {r["name"]: r for r in entities}
    planned_cross = [n for n, e in planned.items() if e["cls"] == "cross_tier"]
    cross_not_mixed = [n for n in planned_cross if not by[n].get("mixed")]
    restricted_planned = [r for r in entities if r["found"] and set(r["acl_labels"]) - {PUBLIC}]
    mixed_planned = [r for r in restricted_planned if r["mixed"]]
    unplanned_mixed = sorted(
        ((n, sorted(node_labels[n]), len(node_docs[n])) for n in nodes
         if n not in planned and len(node_labels[n]) > 1),
        key=lambda x: -x[2])

    # ---- hubs
    hubs = sorted((n for n in nodes if len(node_docs[n]) >= hub_threshold),
                  key=lambda n: -len(node_docs[n]))
    planned_hubs = [n for n, e in planned.items() if e.get("hub")]

    # ---- canaries
    edges = [(s, t, dict(d)) for s, t, d in g.edges(data=True)]
    edge_docs = {(s, t): sources(d) for s, t, d in edges}
    canary_rows = []
    for c in canaries:
        rx = re.compile(c["match_regex"])
        allowed = set(c["acl_users"])
        carriers = []
        for n, d in nodes.items():
            if rx.search(d.get("description") or ""):
                outside = sorted(users(node_docs[n]) - allowed)
                carriers.append({"kind": "node", "key": n, "field": "description",
                                 "visible_outside_acl": outside})
        for s, t, d in edges:
            for fld in ("description", "keywords"):
                if rx.search(d.get(fld) or ""):
                    outside = sorted(users(edge_docs[(s, t)]) - allowed)
                    carriers.append({"kind": "edge", "key": [s, t], "field": fld,
                                     "visible_outside_acl": outside})
        canary_rows.append({
            "canary_id": c["canary_id"], "kind": c["kind"], "entity": c["entity_id"],
            "origin_doc": c["origin_doc_id"],
            "origin_doc_processed": status.get(c["origin_doc_id"], {}).get("status") == "processed",
            "in_index": bool(carriers),
            "exposed_via_merge": any(x["visible_outside_acl"] for x in carriers),
            "named_node": sorted(n for n in nodes if rx.search(n)),
            "carriers": carriers,
        })
    by_kind = {}
    for kind in ("structural", "contamination"):
        rows = [r for r in canary_rows if r["kind"] == kind]
        by_kind[kind] = {"total": len(rows), "in_index": sum(r["in_index"] for r in rows),
                         "exposed_via_merge": sum(r["exposed_via_merge"] for r in rows),
                         "origin_doc_not_processed": [r["canary_id"] for r in rows
                                                      if not r["origin_doc_processed"]]}

    return {
        "corpus": str(corpus), "index": str(wd),
        "docs": {"status": dict(st), "not_processed": not_done, "never_ingested": missing_docs},
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                  "chunks": len(chunks)},
        "entities": {"planned": len(planned), "found": len(planned) - len(missing),
                     "missing": [{"name": r["name"], "near_matches": r["near_matches"]}
                                 for r in entities if not r["found"]]},
        "cross_tier": {
            "planned": len(planned_cross), "planned_not_mixed": cross_not_mixed,
            "realized_cf": round(len(mixed_planned) / len(restricted_planned), 3)
                if restricted_planned else None,
            "mixed_planned": [r["name"] for r in mixed_planned],
            "unplanned_mixed": unplanned_mixed[:20],
        },
        "hubs": {"threshold": hub_threshold, "realized": len(hubs), "planned": len(planned_hubs),
                 "realized_names": [(n, len(node_docs[n])) for n in hubs[:20]],
                 "planned_not_realized": [n for n in planned_hubs if n not in hubs]},
        "canaries": {"by_kind": by_kind,
                     "named_nodes": {r["canary_id"]: r["named_node"] for r in canary_rows
                                     if r["named_node"]},
                     "rows": canary_rows},
        "entity_rows": entities,
    }


def summary(r: dict) -> str:
    ct, cb = r["cross_tier"], r["canaries"]["by_kind"]
    lines = [
        f"docs        {r['docs']['status']}  not processed: {r['docs']['not_processed'] or 'none'}",
        f"graph       {r['graph']['nodes']} nodes, {r['graph']['edges']} edges, {r['graph']['chunks']} chunks",
        f"entities    {r['entities']['found']}/{r['entities']['planned']} found by exact name",
    ]
    for m in r["entities"]["missing"]:
        lines.append(f"            missing {m['name']!r}  near: {m['near_matches'] or '-'}")
    lines += [
        f"cross-tier  realized cf {ct['realized_cf']}; "
        f"{len(ct['mixed_planned'])} mixed, planned cross-tier not mixed: {ct['planned_not_mixed'] or 'none'}",
        f"            unplanned mixed nodes: {[n for n, _, _ in ct['unplanned_mixed'][:8]] or 'none'}",
        f"hubs        {r['hubs']['realized']} realized (>= {r['hubs']['threshold']} chunks) vs "
        f"{r['hubs']['planned']} planned; planned not realized: {len(r['hubs']['planned_not_realized'])}",
    ]
    for kind, v in cb.items():
        lines.append(f"canaries    {kind:13} {v['in_index']}/{v['total']} in index, "
                     f"{v['exposed_via_merge']} visible outside ACL via merge"
                     + (f", origin doc not processed: {v['origin_doc_not_processed']}"
                        if v["origin_doc_not_processed"] else ""))
    lines.append(f"            canary codes as node names: {r['canaries']['named_nodes'] or 'none'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args()
    cfg = load_config(args.rag_config)
    r = check(cfg["corpus"], cfg["working_dir"],
              cfg.get("lightrag", {}).get("force_llm_summary_on_merge", 8))
    print(summary(r))
    out = Path(args.json) if args.json else cfg["working_dir"] / "index_check.json"
    out.write_text(json.dumps(r, indent=1))
    print(f"\nfull report: {out}")


if __name__ == "__main__":
    main()
