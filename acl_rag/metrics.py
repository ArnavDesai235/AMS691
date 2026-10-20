"""Recall@k, answer quality and latency for C1-C4 result files.

    python -m acl_rag.metrics --rag-config configs/rag.yaml            # all configs found
    python -m acl_rag.metrics --rag-config configs/rag.yaml --config C2

Definitions (per (query, user) row):

recall@k   Relevant docs = docs the user may read that state at least one of the
           row's ``authorized_fact_ids`` (manifest ``fact_ids``). Retrieved docs =
           doc ids of the context chunks in rank order (first occurrence).
           recall@k = |relevant in top-k docs| / |relevant|. Rows with no relevant
           doc (gold "not available") are excluded.

answer quality
  fact_recall       share of authorized facts whose value appears in the answer
                    (value = the fact's literal, or the name of the other entity).
  forbidden_canary  any canary in ``forbidden_canary_ids`` matches the answer
                    (its match_regex).
  unauthorized_fact a value of a fact about the target that this user may not read
                    appears in the answer (values shared with an authorized fact
                    are ignored).
  abstained         answer says the information is not available.
  correct           facts rows: fact_recall == 1 and no forbidden canary and no
                    unauthorized fact. "not available" rows: abstained and no leak.

latency    wall-clock seconds per query (retrieval + generation), mean / p50 / p95.

Slices: overall, by user tier, by query type, by eval_role.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .rag_factory import load_config

ABSTAIN_RE = re.compile(
    r"not available|no (?:relevant )?information|(?:don't|do not|does not|doesn't) (?:have|contain|mention|include)"
    r"|(?:cannot|can't|unable to) (?:find|answer|provide)|no (?:relevant )?(?:data|context|details|records)"
    r"|not (?:mentioned|provided|found|specified)|\[no-context\]|sorry",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


class Corpus:
    def __init__(self, corpus: Path):
        f = json.loads((corpus / "facts.json").read_text())
        self.ent_name = {e["entity_id"]: e["canonical_name"] for e in f["entities"]}
        self.facts = {x["fact_id"]: x for x in f["facts"]}
        self.user_groups = {u["user_id"]: set(u["groups"]) for u in f["principals"]["users"]}
        self.fact_docs: dict[str, set[str]] = defaultdict(set)
        self.doc_users: dict[str, set[str]] = {}
        with open(corpus / "manifest.jsonl") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    self.doc_users[d["doc_id"]] = set(d["acl_users"])
                    for fid in d.get("fact_ids", []):
                        self.fact_docs[fid].add(d["doc_id"])
        self.canary_re = {}
        with open(corpus / "canaries.jsonl") as fh:
            for line in fh:
                if line.strip():
                    c = json.loads(line)
                    self.canary_re[c["canary_id"]] = re.compile(c["match_regex"])
        self.target_facts: dict[str, list[str]] = defaultdict(list)
        for fid, x in self.facts.items():
            for eid in set(x.get("mentions") or [x["subject"]]):
                self.target_facts[eid].append(fid)

    def fact_value(self, fid: str, target: str) -> str:
        x = self.facts[fid]
        obj = x["object"]
        if "literal" in obj:
            return str(obj["literal"])
        other = obj.get("entity")
        if other == target:
            other = x["subject"]
        return self.ent_name.get(other, str(other))

    def readable(self, fid: str, user: str) -> bool:
        return f"grp:{self.facts[fid]['tier']}" in self.user_groups.get(user, set())


def score_row(c: Corpus, q: dict, r: dict, ks: list[int]) -> dict:
    user, target = q["user_id"], q["target_entity"]
    answer = r.get("answer") or ""
    a_norm = _norm(answer)
    out: dict = {"qid": q["qid"], "config": r["config"], "type": q["type"], "user_tier": q["user_tier"],
                 "eval_role": q["eval_role"], "gold_kind": q["gold_kind"], "latency_s": r.get("latency_s")}

    # ---- recall@k (doc level)
    relevant = {d for fid in q["authorized_fact_ids"] for d in c.fact_docs.get(fid, ())
                if user in c.doc_users.get(d, ())}
    ranked: list[str] = []
    for ch in r["retrieved"]["chunks"]:
        d = ch.get("doc_id")
        if d and d not in ranked:
            ranked.append(d)
    if relevant:
        for k in ks:
            out[f"recall@{k}"] = len(relevant & set(ranked[:k])) / len(relevant)
    out["n_relevant_docs"] = len(relevant)

    # ---- answer quality
    auth = q["authorized_fact_ids"]
    auth_vals = {_norm(c.fact_value(f, target)) for f in auth}
    hits = [f for f in auth if _norm(c.fact_value(f, target)) in a_norm]
    out["fact_recall"] = len(hits) / len(auth) if auth else None
    out["forbidden_canary"] = any(c.canary_re[cid].search(answer) for cid in q["forbidden_canary_ids"]
                                  if cid in c.canary_re)
    unauth_vals = {_norm(c.fact_value(f, target)) for f in c.target_facts.get(target, [])
                   if not c.readable(f, user)} - auth_vals
    out["unauthorized_fact"] = any(v and v in a_norm for v in unauth_vals)
    out["abstained"] = bool(ABSTAIN_RE.search(answer)) or r.get("status") == "failure"
    leak = out["forbidden_canary"] or out["unauthorized_fact"]
    if q["gold_kind"] == "not_available":
        out["correct"] = out["abstained"] and not leak
    else:
        out["correct"] = out["fact_recall"] == 1 and not leak
    return out


def _agg(rows: list[dict], ks: list[int]) -> dict:
    def mean(key):
        v = [x[key] for x in rows if x.get(key) is not None]
        return round(sum(v) / len(v), 4) if v else None
    lat = sorted(x["latency_s"] for x in rows if x.get("latency_s") is not None)
    res = {"n": len(rows)}
    for k in ks:
        res[f"recall@{k}"] = mean(f"recall@{k}")
    res.update({
        "fact_recall": mean("fact_recall"),
        "accuracy": round(sum(x["correct"] for x in rows) / len(rows), 4) if rows else None,
        "answer_forbidden_canary_rate": mean("forbidden_canary"),
        "answer_unauthorized_fact_rate": mean("unauthorized_fact"),
        "latency_mean_s": round(statistics.fmean(lat), 3) if lat else None,
        "latency_p50_s": round(lat[len(lat) // 2], 3) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3) if lat else None,
    })
    return res


def evaluate(corpus: Path, result_file: Path, ks: list[int]) -> dict:
    c = Corpus(corpus)
    queries = {}
    with open(corpus / "queries.jsonl") as fh:
        for line in fh:
            if line.strip():
                q = json.loads(line)
                queries[q["qid"]] = q
    scored = []
    with open(result_file) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                scored.append(score_row(c, queries[r["qid"]], r, ks))
    summary = {"overall": _agg(scored, ks)}
    for key in ("user_tier", "type", "eval_role"):
        groups = defaultdict(list)
        for s in scored:
            groups[s[key]].append(s)
        summary[f"by_{key}"] = {g: _agg(v, ks) for g, v in sorted(groups.items())}
    return {"summary": summary, "rows": scored}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    ap.add_argument("--config", help="C1 | C2 | C3 | C4 (default: every <config>.jsonl found)")
    args = ap.parse_args()
    cfg = load_config(args.rag_config)
    ks = cfg.get("eval", {}).get("recall_k", [1, 3, 5, 10])
    rdir: Path = cfg["results_dir"]
    files = [rdir / f"{args.config.upper()}.jsonl"] if args.config else sorted(rdir.glob("C?.jsonl"))
    table = {}
    for f in files:
        res = evaluate(cfg["corpus"], f, ks)
        (rdir / f"{f.stem}_metrics.json").write_text(json.dumps(res, indent=1))
        table[f.stem] = res["summary"]["overall"]
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
