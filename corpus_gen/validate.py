"""Independent validator for a generated corpus directory.

Reads ONLY the output files (docs/, manifest.jsonl, facts.json, canaries.jsonl,
queries.jsonl, oracle_splits.json, config.resolved.yaml) and re-derives every
property from the text, so generator bugs cannot hide behind generator state.

Errors (exit 1):
  files      required files exist; manifest docs exist and match sha256
  acl        acl_users == users holding any of acl_groups
  canary     each canary matches (any spelling) ONLY in its origin doc, at the recorded
             span; origin doc tier == canary tier; that doc's ACL excludes >= 1 test user;
             no canary appears in any public doc
  codes      every code-shaped string in any doc is a planted canary in that same doc
  fraction   realized cross-tier fraction == round(cf * |R|) / |R| (config)
  placement  (from text) cross-tier entities appear in docs of >= 2 ACLs; restricted-only
             names appear only in home-tier docs; public-only names only in public docs
  gold       forbidden set == canaries the user cannot read; no forbidden canary in gold;
             authorized facts readable by the user; not_available iff no authorized facts;
             positive controls carry their target canary
  oracle     each access set's split == docs it can read; "all" == every doc
Warnings (manual review): Titlecase words that are neither entity names nor common
English, i.e. possible invented proper nouns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

CODE_REGEX = re.compile(r"\b[A-Z]{2}-\d{4}-[A-Z]\b")


def _distinctive(e: dict[str, Any]) -> str:
    parts = e["canonical_name"].split()
    return parts[-1] if e["type"] == "person" else (parts[1] if e["type"] == "project" else parts[0])


def validate(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    need = ["manifest.jsonl", "facts.json", "canaries.jsonl", "config.resolved.yaml"]
    for n in need:
        if not (run_dir / n).exists():
            errors.append(f"files: missing {n}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    facts = json.loads((run_dir / "facts.json").read_text())
    manifest = [json.loads(l) for l in (run_dir / "manifest.jsonl").open()]
    canaries = [json.loads(l) for l in (run_dir / "canaries.jsonl").open()]
    users = facts["principals"]["users"]
    ents = {e["entity_id"]: e for e in facts["entities"]}
    fmap = {f["fact_id"]: f for f in facts["facts"]}
    pub = next(t["name"] for t in cfg["tiers"] if not t["includes"])
    all_users = {u["user_id"] for u in users}

    def readers(groups) -> set[str]:
        return {u["user_id"] for u in users if set(u["groups"]) & set(groups)}

    def can_read(u: dict, tier: str) -> bool:
        return f"grp:{tier}" in u["groups"]

    # ---- files + acl
    text: dict[str, str] = {}
    tier_of: dict[str, str] = {}
    for m in manifest:
        p = run_dir / "docs" / f"{m['doc_id']}.txt"
        if not p.exists():
            errors.append(f"files: {m['doc_id']} missing")
            continue
        t = p.read_text()
        text[m["doc_id"]], tier_of[m["doc_id"]] = t, m["acl_label"]
        if hashlib.sha256(t.encode()).hexdigest() != m["sha256"]:
            errors.append(f"files: {m['doc_id']} sha256 mismatch")
        if set(m["acl_users"]) != readers(m["acl_groups"]):
            errors.append(f"acl: {m['doc_id']} acl_users inconsistent with acl_groups")

    # ---- canaries
    exact_to_doc = {}
    for c in canaries:
        hits = [d for d, t in text.items() if re.search(c["match_regex"], t)]
        if hits != [c["origin_doc_id"]]:
            errors.append(f"canary: {c['canary_id']} found in {hits}, expected [{c['origin_doc_id']}]")
        d = c["origin_doc_id"]
        if d in text:
            s, e = c["origin_char_span"]
            if text[d][s:e] != c["exact"]:
                errors.append(f"canary: {c['canary_id']} span mismatch")
            if tier_of[d] != c["tier"]:
                errors.append(f"canary: {c['canary_id']} tier {c['tier']} but doc tier {tier_of[d]}")
            if readers([f"grp:{tier_of[d]}"]) >= all_users:
                errors.append(f"canary: {c['canary_id']} in a doc every user can read")
            if tier_of[d] == pub:
                errors.append(f"canary: {c['canary_id']} in public doc {d}")
        exact_to_doc[c["exact"]] = d

    # ---- stray codes
    for d, t in text.items():
        for m in CODE_REGEX.finditer(t):
            if exact_to_doc.get(m.group(0)) != d:
                errors.append(f"codes: unexpected code {m.group(0)} in {d}")

    # ---- cross-tier fraction (config vs realized)
    R = [e for e in ents.values() if e["pool"] == "restricted"]
    cf = cfg["entities"]["cross_tier_fraction"]
    n_cross = sum(e["cls"] == "cross_tier" for e in R)
    if R and n_cross != round(cf * len(R)):
        errors.append(f"fraction: {n_cross}/{len(R)} cross-tier, expected round({cf}*{len(R)})")

    # ---- placement from text
    appears: dict[str, set[str]] = {eid: set() for eid in ents}
    for d, t in text.items():
        for eid, e in ents.items():
            if re.search(rf"\b{re.escape(_distinctive(e))}\b", t, flags=re.I):
                appears[eid].add(tier_of[d])
    for eid, e in ents.items():
        tiers = appears[eid]
        if e["cls"] == "cross_tier" and len(tiers) < 2:
            errors.append(f"placement: cross-tier {e['canonical_name']} appears only in {sorted(tiers)}")
        if e["cls"] == "restricted_only" and tiers - {e["home_tier"]}:
            errors.append(f"placement: restricted-only {e['canonical_name']} appears in {sorted(tiers)}")
        if e["cls"] == "public_only" and tiers - {pub}:
            errors.append(f"placement: public-only {e['canonical_name']} appears in {sorted(tiers)}")

    # ---- gold
    qpath = run_dir / "queries.jsonl"
    n_rows = 0
    if qpath.exists():
        cmap = {c["canary_id"]: c for c in canaries}
        umap = {u["user_id"]: u for u in users}
        for line in qpath.open():
            r = json.loads(line)
            n_rows += 1
            u = umap[r["user_id"]]
            expect = sorted(c["canary_id"] for c in canaries if not can_read(u, c["tier"]))
            if r["forbidden_canary_ids"] != expect:
                errors.append(f"gold: {r['qid']} forbidden set != canaries user cannot read")
            for cid in r["forbidden_canary_ids"]:
                if re.search(cmap[cid]["match_regex"], r["gold_answer"]):
                    errors.append(f"gold: {r['qid']} contains forbidden canary {cid}")
            for fid in r["authorized_fact_ids"]:
                if not can_read(u, fmap[fid]["tier"]):
                    errors.append(f"gold: {r['qid']} authorized fact {fid} not readable by user")
            readable = [f for f in facts["facts"] if r["target_entity"] in f["mentions"] and can_read(u, f["tier"])]
            if (r["gold_kind"] == "not_available") != (not readable):
                errors.append(f"gold: {r['qid']} gold_kind inconsistent with readable facts")
            if r["eval_role"] == "positive_control" and r["gold_kind"] == "facts":
                for cid in r["authorized_canary_ids"]:
                    if cmap[cid]["entity_id"] == r["target_entity"] and cmap[cid]["exact"] not in r["gold_answer"]:
                        errors.append(f"gold: {r['qid']} positive control missing {cid}")
    else:
        warnings.append("queries.jsonl not found; gold checks skipped")

    # ---- oracle
    opath = run_dir / "oracle_splits.json"
    if opath.exists():
        o = json.loads(opath.read_text())
        for as_id, groups in facts["principals"]["access_sets"].items():
            expect = sorted(m["doc_id"] for m in manifest if set(groups) & set(m["acl_groups"]))
            if o["access_sets"][as_id]["doc_ids"] != expect:
                errors.append(f"oracle: {as_id} split mismatch")
        if o["all"]["doc_ids"] != sorted(m["doc_id"] for m in manifest):
            errors.append("oracle: 'all' split mismatch")
    else:
        warnings.append("oracle_splits.json not found; oracle checks skipped")

    # ---- review list: possible invented proper nouns
    try:
        from wordfreq import zipf_frequency
    except Exception:  # noqa: BLE001
        zipf_frequency = None
    name_toks = {t for e in ents.values() for t in e["canonical_name"].split()}
    name_toks |= set(facts["company"].split())
    lit_toks = {t for f in facts["facts"] for t in re.findall(r"[A-Za-z]+", f["object"].get("literal", ""))}
    review = Counter()
    if zipf_frequency:
        for t in text.values():
            for w in re.findall(r"(?<![\w'-])([A-Z][a-z]{2,})(?![\w-])", t):
                if w not in name_toks and w not in lit_toks and zipf_frequency(w.lower(), "en") < 2.0:
                    review[w] += 1
    if review:
        warnings.append("review: rare Titlecase words (possible invented names): "
                        + ", ".join(f"{w}({n})" for w, n in review.most_common(30)))

    return {
        "ok": not errors, "n_errors": len(errors), "errors": errors[:200], "warnings": warnings,
        "counts": {"docs": len(manifest), "canaries": len(canaries), "query_rows": n_rows,
                   "cross_tier": n_cross, "restricted_pool": len(R),
                   "cross_tier_fraction_realized": round(n_cross / len(R), 4) if R else 0.0},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a generated corpus directory.")
    ap.add_argument("run_dir", type=Path, help="e.g. corpus/dev_cf0.3_s42")
    a = ap.parse_args()
    rep = validate(a.run_dir)
    (a.run_dir / "validation_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("ok", "n_errors", "counts") if k in rep}, indent=2))
    for e in rep["errors"][:20]:
        print("ERROR", e)
    for w in rep["warnings"]:
        print("WARN ", w)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
