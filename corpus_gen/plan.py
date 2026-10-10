"""Document plan: assign facts to documents (no LLM calls).

Each document has one ACL (= one tier). Chunks inherit the doc ACL, so canary
origin chunks are unambiguous.

Two placement phases per tier keep the restricted side sweep-invariant:
  Phase A (cf-invariant): public-only facts in public docs; home-tier facts
      (restricted + canary) of every restricted-pool entity in restricted docs.
  Phase B (cf-dependent): lower-tier facts of cross-tier entities ("bridge"
      mentions) added on top of phase-A docs, with an RNG keyed per entity so
      adding a cross-tier entity never moves another entity's placements.
Canary placement happens only in phase A, so canary -> doc mapping is identical
across the sweep. Sibling-tier docs can still gain bridge facts as cf grows (and
then get re-verbalized); the CLI reports exactly which restricted docs differ.

Mention plan: every entity is stated in its planned number of docs per tier by
repeating a non-canary fact about it. A canary appears in exactly one doc.

Merge regime: LightRAG 1.5.7 LLM-summarizes a merged description only at >= 8
fragments (one per mentioning chunk). The plan records planned_docs and a
predicted regime; analysis should use the realized source_id count after ingest.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, load_config, rng_for
from .facts import Entity, Fact, FactTable, _largest_remainder, build_fact_table


@dataclass
class DocPlan:
    doc_id: str
    tier: str
    acl_groups: list[str]
    acl_users: list[str]
    doc_type: str
    target_words: int
    fact_ids: list[str] = field(default_factory=list)      # in placement order
    bridge_fact_ids: list[str] = field(default_factory=list)  # phase-B subset
    entities: list[str] = field(default_factory=list)
    canary_ids: list[str] = field(default_factory=list)


@dataclass
class Plan:
    docs: list[DocPlan]
    entity_docs: dict[str, dict[str, list[str]]]   # eid -> tier -> [doc_id]
    planned_docs: dict[str, int]                    # eid -> #docs mentioning
    predicted_regime: dict[str, str]                # eid -> concatenated|summarized
    canary_doc: dict[str, str]                      # canary_id -> doc_id
    shortfalls: list[dict[str, Any]]
    stats: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"docs": [asdict(d) for d in self.docs], "entity_docs": self.entity_docs,
                "planned_docs": self.planned_docs, "predicted_regime": self.predicted_regime,
                "canary_doc": self.canary_doc, "shortfalls": self.shortfalls, "stats": self.stats}


class _TierPlanner:
    def __init__(self, docs: list[DocPlan], facts: dict[str, Fact]):
        self.docs = docs
        self.facts = facts
        self.ents: dict[str, set[str]] = {d.doc_id: set() for d in docs}
        self.fset: dict[str, set[str]] = {d.doc_id: set() for d in docs}

    def mentions(self, eid: str) -> int:
        return sum(eid in s for s in self.ents.values())

    def put(self, d: DocPlan, f: Fact, bridge: bool) -> None:
        d.fact_ids.append(f.fact_id)
        if bridge:
            d.bridge_fact_ids.append(f.fact_id)
        if f.canary_id:
            d.canary_ids.append(f.canary_id)
        self.fset[d.doc_id].add(f.fact_id)
        for m in f.mentions:
            if m not in self.ents[d.doc_id]:
                self.ents[d.doc_id].add(m)
                d.entities.append(m)

    def best_doc(self, r, f: Fact, exclude_entity: str | None = None,
                 cohesion: float = 1.5) -> DocPlan | None:
        best, best_s = None, None
        for d in self.docs:
            if f.fact_id in self.fset[d.doc_id]:
                continue
            if exclude_entity and exclude_entity in self.ents[d.doc_id]:
                continue
            overlap = len(self.ents[d.doc_id] & set(f.mentions))
            s = len(d.fact_ids) + 0.5 * r.random() - cohesion * overlap
            if best_s is None or s < best_s:
                best, best_s = d, s
        return best

    def seed(self, r, facts: list[Fact], bridge: bool) -> None:
        """Place every fact at least once (canaries exactly once)."""
        order = facts[:]
        r.shuffle(order)
        for f in order:
            d = self.best_doc(r, f)
            if d is not None:
                self.put(d, f, bridge)

    def fill_mentions(self, r, eid: str, target: int, pool: list[Fact], bridge: bool,
                      can_add: Any) -> int:
        """Repeat non-canary facts about `eid` in new docs until `target` docs mention it."""
        reps = [f for f in pool if not f.is_canary and eid in f.mentions]
        short = 0
        guard = 0
        while self.mentions(eid) < target and guard < 4 * target + 10:
            guard += 1
            r.shuffle(reps)
            placed = False
            for f in reps:
                others = [m for m in f.mentions if m != eid]
                if not all(can_add(m) for m in others):
                    continue
                d = self.best_doc(r, f, exclude_entity=eid, cohesion=0.0)
                if d is not None:
                    self.put(d, f, bridge)
                    placed = True
                    break
            if not placed:
                short = target - self.mentions(eid)
                break
        return short


def build_plan(cfg: Config, ft: FactTable) -> Plan:
    cs, pub = cfg.company_seed, cfg.public_tier
    em: dict[str, Entity] = {e.entity_id: e for e in ft.entities}
    fm: dict[str, Fact] = {f.fact_id: f for f in ft.facts}
    readers = lambda t: [u.user_id for u in ft.users if f"grp:{t}" in u.groups]  # noqa: E731
    thr = cfg.lightrag.force_llm_summary_on_merge
    cap_nonhub = cfg.entities.mentions_low[1]

    # ---- doc shells per tier (ids stable across the sweep)
    tiers = _largest_remainder({t.name: t.doc_share for t in cfg.tiers}, cfg.docs.n_docs)
    n_per = {t: tiers.count(t) for t in cfg.tier_names}
    docs_by_tier: dict[str, list[DocPlan]] = {}
    for t in cfg.tier_names:
        r = rng_for(cs, "docs", t)
        docs_by_tier[t] = [DocPlan(f"d-{t}-{i:03d}", t, [f"grp:{t}"], readers(t),
                                   r.choice(cfg.docs.doc_types), 0)
                           for i in range(n_per[t])]

    planners = {t: _TierPlanner(docs_by_tier[t], fm) for t in cfg.tier_names}
    shortfalls: list[dict[str, Any]] = []

    def total_docs(eid: str) -> int:
        return sum(p.mentions(eid) for p in planners.values())

    def can_add_factory(tier: str):
        # Phase A only: must not read e.cls (cf-dependent) or the plan stays sweep-invariant
        # only by luck. Restricted-pool entities are capped at their planned home mentions.
        def can_add(eid: str) -> bool:
            e = em[eid]
            if e.pool == "restricted" and tier == e.home_tier:
                return planners[tier].mentions(eid) < e.restricted_mentions
            return True
        return can_add

    # ---- phase A: cf-invariant
    pub_facts = [f for f in ft.facts if f.role == "public"]
    planners[pub].seed(rng_for(cs, "planA", pub), pub_facts, bridge=False)
    for e in sorted((e for e in ft.entities if e.pool in ("open", "qc")), key=lambda e: e.entity_id):
        s = planners[pub].fill_mentions(rng_for(cs, "fillA", e.entity_id), e.entity_id,
                                        e.lower_mentions, pub_facts, False, can_add_factory(pub))
        if s:
            shortfalls.append({"entity": e.entity_id, "tier": pub, "phase": "A", "short": s})

    for t in cfg.tier_names:
        if t == pub:
            continue
        home_facts = [f for f in ft.facts if f.role in ("restricted", "canary") and f.tier == t]
        planners[t].seed(rng_for(cs, "planA", t), home_facts, bridge=False)
        for e in sorted((e for e in ft.entities if e.pool == "restricted" and e.home_tier == t),
                        key=lambda e: e.entity_id):
            s = planners[t].fill_mentions(rng_for(cs, "fillA", e.entity_id), e.entity_id,
                                          e.restricted_mentions, home_facts, False,
                                          can_add_factory(t))
            if s:
                shortfalls.append({"entity": e.entity_id, "tier": t, "phase": "A", "short": s})

    # ---- phase B: bridge mentions of cross-tier entities (cf-dependent)
    for e in sorted((e for e in ft.entities if e.cls == "cross_tier"), key=lambda e: e.entity_id):
        t = e.low_tier
        lower = [f for f in ft.facts if f.role == "lower" and f.subject == e.entity_id]
        r = rng_for(cs, "planB", e.entity_id)
        planners[t].seed(r, lower, bridge=True)
        target = e.lower_mentions
        if not e.hub:  # keep non-hubs strictly in the concatenated regime
            target = min(target, max(1, cap_nonhub - total_docs(e.entity_id) + planners[t].mentions(e.entity_id)))
        if target > len(docs_by_tier[t]):
            shortfalls.append({"entity": e.entity_id, "tier": t, "phase": "B",
                               "short": target - len(docs_by_tier[t]), "reason": "tier has too few docs"})
            target = len(docs_by_tier[t])
        s = planners[t].fill_mentions(r, e.entity_id, target, lower, True,
                                      lambda m: True)  # lower facts only name open public-only ents
        if s:
            shortfalls.append({"entity": e.entity_id, "tier": t, "phase": "B", "short": s})

    # ---- word targets
    for t, docs in docs_by_tier.items():
        for d in docs:
            r = rng_for(cs, "words", d.doc_id)
            base = 260 + 70 * len(d.fact_ids) + r.randint(-60, 120)
            d.target_words = max(cfg.docs.words_min, min(cfg.docs.words_max, base))

    # ---- summaries
    entity_docs: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for t, p in planners.items():
        for d in p.docs:
            for eid in d.entities:
                entity_docs[eid].setdefault(t, []).append(d.doc_id)
    planned = {eid: sum(len(v) for v in tiers_.values()) for eid, tiers_ in entity_docs.items()}
    regime = {eid: ("summarized" if n >= thr else "concatenated") for eid, n in planned.items()}
    canary_doc = {cid: d.doc_id for docs in docs_by_tier.values() for d in docs for cid in d.canary_ids}

    all_docs = [d for t in cfg.tier_names for d in docs_by_tier[t]]
    plan = Plan(all_docs, dict(entity_docs), planned, regime, canary_doc, shortfalls, {})
    plan.stats = _stats(cfg, ft, plan)
    check_plan(cfg, ft, plan)
    return plan


def _stats(cfg: Config, ft: FactTable, plan: Plan) -> dict[str, Any]:
    cross = [e for e in ft.entities if e.cls == "cross_tier"]
    by_t = defaultdict(list)
    for d in plan.docs:
        by_t[d.tier].append(d)
    return {
        "docs_per_tier": {t: len(v) for t, v in by_t.items()},
        "facts_per_doc": {t: round(sum(len(d.fact_ids) for d in v) / max(1, len(v)), 2) for t, v in by_t.items()},
        "entities_per_doc": {t: round(sum(len(d.entities) for d in v) / max(1, len(v)), 2) for t, v in by_t.items()},
        "words_total": sum(d.target_words for d in plan.docs),
        "cross_regime": {
            "hub_summarized": sum(e.hub and plan.predicted_regime.get(e.entity_id) == "summarized" for e in cross),
            "hub_not_summarized": sum(e.hub and plan.predicted_regime.get(e.entity_id) != "summarized" for e in cross),
            "nonhub_concatenated": sum((not e.hub) and plan.predicted_regime.get(e.entity_id) == "concatenated" for e in cross),
            "nonhub_summarized": sum((not e.hub) and plan.predicted_regime.get(e.entity_id) == "summarized" for e in cross),
        },
        "n_shortfalls": len(plan.shortfalls),
    }


def check_plan(cfg: Config, ft: FactTable, plan: Plan) -> None:
    errs: list[str] = []
    fm = {f.fact_id: f for f in ft.facts}
    em = {e.entity_id: e for e in ft.entities}
    placed = defaultdict(int)
    for d in plan.docs:
        for fid in d.fact_ids:
            f = fm[fid]
            placed[fid] += 1
            if f.tier != d.tier:
                errs.append(f"{fid} (tier {f.tier}) placed in {d.doc_id}")
            for m in f.mentions:  # entity visibility rules carried through to docs
                if em[m].cls == "restricted_only" and d.tier != em[m].home_tier:
                    errs.append(f"{d.doc_id}: names restricted-only {m} outside home tier")
                if em[m].cls == "public_only" and d.tier != cfg.public_tier:
                    errs.append(f"{d.doc_id}: names public-only {m} in non-public doc")
        if not d.fact_ids:
            errs.append(f"{d.doc_id}: empty doc")
    for f in ft.facts:
        if placed[f.fact_id] == 0:
            errs.append(f"{f.fact_id} never placed")
        if f.is_canary and placed[f.fact_id] != 1:
            errs.append(f"canary fact {f.fact_id} placed {placed[f.fact_id]}x")
    for c in ft.canaries:
        if c.canary_id not in plan.canary_doc:
            errs.append(f"{c.canary_id} has no origin doc")
    for e in ft.entities:
        if e.cls == "cross_tier" and len(plan.entity_docs.get(e.entity_id, {})) < 2:
            errs.append(f"{e.entity_id} cross-tier but appears in < 2 tiers")
    if errs:
        raise AssertionError("plan invalid:\n  " + "\n  ".join(errs[:30]))


def attach_canary_origins(ft: FactTable, plan: Plan) -> None:
    for c in ft.canaries:
        c.origin_doc_id = plan.canary_doc[c.canary_id]


def write_plan(plan: Plan, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan.to_json(), indent=2))


# --------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Build/inspect the document plan (no LLM calls).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", type=float, nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    cfs = a.sweep or [None]
    results = []
    for cf in cfs:
        cfg = load_config(a.config, {"entities.cross_tier_fraction": cf} if cf is not None else None)
        ft = build_fact_table(cfg)
        plan = build_plan(cfg, ft)
        results.append((cfg, ft, plan))
        print(f"cf={cfg.entities.cross_tier_fraction}: {json.dumps(plan.stats)}")
        for s in plan.shortfalls[:5]:
            print("  shortfall:", s)
        if a.write:
            attach_canary_origins(ft, plan)
            write_plan(plan, cfg.out_dir)
            print(f"  wrote {cfg.out_dir}/plan.json")

    if len(results) > 1:
        pub = results[0][0].public_tier
        for (c1, _, p1), (c2, _, p2) in zip(results, results[1:]):
            assert p1.canary_doc == p2.canary_doc, "canary -> doc mapping changed across sweep"
            for scope, keep in (("restricted", lambda d: d.tier != pub), ("public", lambda d: d.tier == pub)):
                a1 = {d.doc_id: _phase_a(d) for d in p1.docs if keep(d)}
                a2 = {d.doc_id: _phase_a(d) for d in p2.docs if keep(d)}
                assert a1 == a2, f"phase-A placement of {scope} docs changed across sweep"
            full1 = {d.doc_id: tuple(d.fact_ids) for d in p1.docs if d.tier != pub}
            full2 = {d.doc_id: tuple(d.fact_ids) for d in p2.docs if d.tier != pub}
            changed = sum(full1[k] != full2[k] for k in full1)
            print(f"cf {c1.entities.cross_tier_fraction} -> {c2.entities.cross_tier_fraction}: "
                  f"canary->doc identical; phase-A placement identical; "
                  f"restricted docs gaining bridge facts: {changed}/{len(full1)}")


def _phase_a(d: DocPlan) -> tuple[str, ...]:
    b = set(d.bridge_fact_ids)
    return tuple(f for f in d.fact_ids if f not in b)


if __name__ == "__main__":
    main()
