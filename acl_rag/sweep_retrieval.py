"""Retrieval-only sweep: recall@k for C1-C4 under different LightRAG query settings.

    python -m acl_rag.sweep_retrieval --rag-config configs/rag.yaml
    python -m acl_rag.sweep_retrieval --rag-config configs/rag.yaml --configs C3 C4 \
        --top-k 10 20 40 --budgets 6000/8000 3000/3000 --chunk-top-k 20 40 --keywords orig ll

Uses ``rag.aquery_data`` (same retrieval and ACL hooks as ``acl_rag.run``, no answer
generation). Keywords are taken from an earlier run's result files (``--keywords-from``,
default: the config's results_dir), so no keyword LLM call is made; C4 with
``description_strategy: auto`` may still summarize hub items it has not summarized before.

Retrieval depends only on (query, access set) -- on the query alone for C1 -- so each is
retrieved once and scored for every user in it. Recall uses the same definition as
``acl_rag.metrics``: relevant = docs holding the row's authorized facts that the user may
read; ranking = chunk order, first occurrence of each doc.

Settings (a grid):
  top_k        entities / relations per vector search
  budgets      max_entity_tokens/max_relation_tokens (max_total_tokens is kept)
  chunk_top_k  cap on the FINAL chunk list (entity-, relation- and vector-sourced chunks
               after round-robin merge), applied before the token budget truncates it
  keywords     orig: recorded high/low-level keywords
               ll:   recorded low-level keywords also used as high-level keywords

The setting equal to the config's query block is checked against the earlier run: the
summary reports the fraction of rows whose ranked docs are identical (``baseline_match``).

Output: <results_dir>/sweep/summary.json (and rows.jsonl.gz with ranked docs per row).
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import itertools
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .enforce import Mode, acting_as
from .metrics import Corpus
from .rag_factory import build_acl_rag, load_config
from .resultio import iter_rows, result_files
from .run import load_rows

log = logging.getLogger("acl_rag.sweep_retrieval")
KS = (1, 3, 5, 10, 20)


def recorded(results_dir: Path) -> tuple[dict, dict]:
    """Keywords per query text and ranked docs per (config, qid) from an earlier run."""
    keywords: dict[str, dict] = {}
    ranked: dict[tuple[str, str], list[str]] = {}
    for path in result_files(results_dir):
        for r in iter_rows(path):
            if r.get("keywords"):
                keywords.setdefault(r["query_id"], r["keywords"])
            ranked[(r["config"], r["qid"])] = _ranked_docs(r["retrieved"]["chunks"], "doc_id")
    return keywords, ranked


def _ranked_docs(chunks: list[dict], field: str) -> list[str]:
    out: list[str] = []
    for c in chunks:
        d = c.get(field)
        if d and d not in out:
            out.append(d)
    return out


def settings(args, qcfg: dict) -> list[dict]:
    grid = []
    for top_k, budget, ctk, kw in itertools.product(args.top_k, args.budgets, args.chunk_top_k, args.keywords):
        ent, rel = (int(x) for x in budget.split("/"))
        grid.append({"top_k": top_k, "max_entity_tokens": ent, "max_relation_tokens": rel,
                     "chunk_top_k": ctk, "keywords": kw})
    base = {"top_k": qcfg.get("top_k", 40), "max_entity_tokens": 6000, "max_relation_tokens": 8000,
            "chunk_top_k": qcfg.get("chunk_top_k", 20), "keywords": "orig"}
    if base not in grid:
        grid.insert(0, base)
    return [{**s, "baseline": s == base} for s in grid]


def label(s: dict) -> str:
    return (f"k{s['top_k']}_e{s['max_entity_tokens']}_r{s['max_relation_tokens']}"
            f"_c{s['chunk_top_k']}_{s['keywords']}")


def query_param(qcfg: dict, s: dict, kw: dict | None):
    from lightrag import QueryParam  # noqa: PLC0415
    hl, ll = list((kw or {}).get("high_level") or []), list((kw or {}).get("low_level") or [])
    if s["keywords"] == "ll" and ll:
        hl = ll
    return QueryParam(mode=qcfg.get("mode", "mix"), top_k=s["top_k"], chunk_top_k=s["chunk_top_k"],
                      max_entity_tokens=s["max_entity_tokens"], max_relation_tokens=s["max_relation_tokens"],
                      enable_rerank=qcfg.get("enable_rerank", False), hl_keywords=hl, ll_keywords=ll)


def score(corpus: Corpus, row: dict, ranked: list[str]) -> dict | None:
    user = row["user_id"]
    relevant = {d for fid in row["authorized_fact_ids"] for d in corpus.fact_docs.get(fid, ())
                if user in corpus.doc_users.get(d, ())}
    if not relevant:
        return None
    out = {f"recall@{k}": len(relevant & set(ranked[:k])) / len(relevant) for k in KS}
    out["recall@all"] = len(relevant & set(ranked)) / len(relevant)
    return out


async def sweep_config(cfg: dict, mode: Mode, grid: list[dict], rows: list[dict], keywords: dict,
                       baseline_ranked: dict, corpus: Corpus, concurrency: int, rows_out) -> list[dict]:
    rag, acl = await build_acl_rag(cfg, mode=mode)
    qcfg = cfg.get("query", {})
    sem = asyncio.Semaphore(concurrency)
    results = []
    try:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[(r["query_id"], "*" if mode is Mode.C1 else acl.access_set_of(r["user_id"]))].append(r)
        for s in grid:
            t0 = time.perf_counter()
            per_row: list[dict] = []
            sizes = defaultdict(list)
            same = []

            async def one(members):
                rep = members[0]
                async with sem:
                    with acting_as(acl, rep["user_id"], mode):
                        res = await rag.aquery_data(rep["text"], param=query_param(qcfg, s, keywords.get(rep["query_id"])))
                data = res.get("data") or {}
                ranked = _ranked_docs(data.get("chunks") or [], "file_path")
                sizes["entities"].append(len(data.get("entities") or []))
                sizes["relations"].append(len(data.get("relationships") or []))
                sizes["chunks"].append(len(data.get("chunks") or []))
                for r in members:
                    sc = score(corpus, r, ranked)
                    if sc is not None:
                        per_row.append({"type": r["type"], "eval_role": r["eval_role"], **sc})
                    if s["baseline"] and (mode.value, r["qid"]) in baseline_ranked:
                        same.append(baseline_ranked[(mode.value, r["qid"])] == ranked)
                    rows_out.write(json.dumps({"config": mode.value, "setting": label(s), "qid": r["qid"],
                                               "ranked_docs": ranked}) + "\n")

            await asyncio.gather(*(one(m) for m in groups.values()))
            summary = {"config": mode.value, "setting": label(s), **{k: v for k, v in s.items()},
                       "n_rows": len(per_row), "n_retrievals": len(groups),
                       "seconds": round(time.perf_counter() - t0, 1)}
            for k in [*(f"recall@{k}" for k in KS), "recall@all"]:
                summary[k] = round(statistics.fmean(x[k] for x in per_row), 4) if per_row else None
            for t in sorted({x["type"] for x in per_row}):
                sub = [x for x in per_row if x["type"] == t]
                summary[f"type_{t}_recall@10"] = round(statistics.fmean(x["recall@10"] for x in sub), 4)
                summary[f"type_{t}_recall@all"] = round(statistics.fmean(x["recall@all"] for x in sub), 4)
            summary.update({f"mean_{k}": round(statistics.fmean(v), 1) for k, v in sizes.items()})
            if s["baseline"]:
                summary["baseline_match"] = round(sum(same) / len(same), 4) if same else None
            log.info("%s %s recall@10=%s all=%s (%ss)", mode.value, label(s), summary["recall@10"],
                     summary["recall@all"], summary["seconds"])
            results.append(summary)
    finally:
        await rag.finalize_storages()
    return results


async def main_async(args) -> Path:
    cfg = load_config(args.rag_config)
    rows = load_rows(cfg["corpus"])
    if args.limit:
        rows = rows[:args.limit]
    corpus = Corpus(cfg["corpus"])
    keywords, baseline_ranked = recorded(Path(args.keywords_from) if args.keywords_from else cfg["results_dir"])
    missing = {r["query_id"] for r in rows} - keywords.keys()
    if missing:
        log.warning("%d queries have no recorded keywords; LightRAG will extract them with the LLM", len(missing))
    grid = settings(args, cfg.get("query", {}))
    out_dir = Path(args.out) if args.out else cfg["results_dir"] / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with gzip.open(out_dir / "rows.jsonl.gz", "wt") as rows_out:
        for name in args.configs:
            summaries += await sweep_config(cfg, Mode.parse(name), grid, rows, keywords, baseline_ranked,
                                            corpus, args.concurrency, rows_out)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summaries, indent=1) + "\n")
    return path


def table(summaries: list[dict]) -> str:
    cols = ["config", "setting", "recall@5", "recall@10", "recall@20", "recall@all", "mean_chunks", "seconds"]
    lines = ["\t".join(cols)]
    for s in summaries:
        lines.append("\t".join(str(s.get(c)) for c in cols) + ("\tBASELINE match=%s" % s.get("baseline_match")
                                                                 if s.get("baseline") else ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    ap.add_argument("--configs", nargs="+", default=["C1", "C2", "C3", "C4"])
    ap.add_argument("--top-k", nargs="+", type=int, default=[10, 20, 40])
    ap.add_argument("--budgets", nargs="+", default=["6000/8000", "3000/3000"],
                    help="max_entity_tokens/max_relation_tokens pairs")
    ap.add_argument("--chunk-top-k", nargs="+", type=int, default=[20, 40])
    ap.add_argument("--keywords", nargs="+", default=["orig", "ll"], choices=["orig", "ll"])
    ap.add_argument("--keywords-from", help="results dir of an earlier run (default: the config's results_dir)")
    ap.add_argument("--limit", type=int, help="only the first N rows (smoke test)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", help="output dir (default: <results_dir>/sweep)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("lightrag", "nano-vectordb"):
        logging.getLogger(name).setLevel(logging.WARNING)
    path = asyncio.run(main_async(args))
    print(table(json.loads(path.read_text())))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
