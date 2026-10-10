"""Ground-truth fact table: principals, entities, facts, canaries.

Everything here is generated in code, before any LLM call. Documents are later
verbalizations of these facts.

Sweep invariance (cross_tier_fraction = cf):
  * The entity pool, the restricted/public split, home tiers, lower tiers, hub
    flags, every fact, and every canary code depend only on `company_seed`.
  * Restricted-tier entities (R) are ranked once by `sweep_seed`; the first
    round(cf*|R|) are cross-tier. Larger cf -> superset of cross-tier entities.
  * Facts for every R entity are generated as if it were cross-tier; for a
    restricted-only entity its lower-tier facts are dropped. Hence restricted
    facts/canaries are byte-identical across the sweep; only the public side moves.

Leak-hygiene rules (prevent *incidental* cross-tier entities):
  * A non-public fact never names a public-only entity (its node would otherwise
    receive restricted description fragments and silently become cross-tier).
  * Restricted relations connect two R entities with the same home tier.
  * Sibling-tier (non-public lower tier) facts are attributes only.
  * QC entities (type-c targets) relate only to other QC entities.
The company name is the one unavoidable incidental cross-tier entity; it is
excluded from all fraction accounting.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, load_config, rng_for
from .names import (CANARY_KINDS, CODE_REGEX, ENTITY_TYPES, CanaryCodeFactory,
                    code_match_regex, code_variants, make_names)

CLASSES = ("public_only", "cross_tier", "restricted_only")


# ------------------------------------------------------------------ records
@dataclass
class User:
    user_id: str
    tier: str
    groups: list[str]
    access_set: str


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    type: str
    title: str | None
    pool: str                    # "restricted" (R) | "open" | "qc"
    cls: str                     # public_only | cross_tier | restricted_only
    home_tier: str               # tier of restricted facts (public for public-only)
    low_tier: str | None         # lower-tier side for R entities (if/when cross-tier)
    hub: bool                    # cross-tier hub -> summarized merge regime
    sweep_rank: int | None       # rank in the sweep permutation (R only)
    restricted_mentions: int     # planned #docs at home tier (R only)
    lower_mentions: int          # planned #docs at low tier (public-only: public docs)
    qc_clean: bool


@dataclass
class Fact:
    fact_id: str
    subject: str
    predicate: str
    object: dict[str, str]       # {"entity": "E012"} | {"literal": "Q3 FY27"}
    sentence: str
    tier: str
    role: str                    # public | lower | restricted | canary
    mentions: list[str]          # entity ids named in the sentence
    is_canary: bool = False
    canary_id: str | None = None


@dataclass
class Canary:
    canary_id: str
    kind: str                    # contamination | structural (depends on cf)
    fact_id: str
    entity_id: str
    label: str                   # e.g. "escrow reference"
    exact: str
    context_sentence: str
    variants: list[str]
    match_regex: str
    tier: str
    acl_groups: list[str]
    acl_users: list[str]
    # filled by plan.py / ingest-time mapping
    origin_doc_id: str | None = None
    origin_char_span: list[int] | None = None
    origin_chunks: list[str] | None = None


@dataclass
class FactTable:
    company: str
    config_hash: str
    cross_tier_fraction: float
    groups: list[str]
    users: list[User]
    access_sets: dict[str, list[str]]
    entities: list[Entity]
    facts: list[Fact]
    canaries: list[Canary]
    stats: dict[str, Any] = field(default_factory=dict)

    # convenience
    def entity(self, eid: str) -> Entity:
        return self._emap()[eid]

    def _emap(self) -> dict[str, Entity]:
        if not hasattr(self, "_em"):
            self._em = {e.entity_id: e for e in self.entities}  # type: ignore[attr-defined]
        return self._em  # type: ignore[attr-defined]

    def to_json(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "config_hash": self.config_hash,
            "cross_tier_fraction": self.cross_tier_fraction,
            "principals": {
                "groups": self.groups,
                "users": [asdict(u) for u in self.users],
                "access_sets": self.access_sets,
            },
            "entities": [asdict(e) for e in self.entities],
            "facts": [asdict(f) for f in self.facts],
            "stats": self.stats,
        }


# ------------------------------------------------------------ fact catalog
_STATUS = ["on track", "at risk", "in pilot", "paused pending review", "in early design"]
_PRICING = ["enterprise tier", "standard tier", "internal-only tier", "partner tier"]
_CHARTER = ["data retention tooling", "payments reconciliation", "fleet telemetry",
            "identity and access tooling", "demand forecasting models",
            "customer onboarding flows", "warehouse automation", "incident response tooling"]
_REGION = ["the northern region", "the coastal region", "the central region",
           "the eastern region", "the river valley region"]


def _quarter(r): return f"Q{r.randint(1, 4)} FY{r.randint(26, 28)}"
def _money(r): return f"${r.randint(3, 95) / 10:.1f}M"
def _people(r): return f"{r.randint(4, 40)} people"


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def ref(e: Entity, start: bool = False) -> str:
    """Surface reference; canonical name always appears verbatim."""
    if e.type in ("team", "product", "site"):
        return ("The " if start else "the ") + e.canonical_name
    return e.canonical_name


# attribute predicates: type -> [(predicate, value_fn(rng, entity), template)]
# template fields: {S}=subject at sentence start, {v}=value
ATTRS: dict[str, list[tuple[str, Any, str]]] = {
    "person": [
        ("title", lambda r, e: e.title, "{S} works as {art} {v}."),
        ("joined_year", lambda r, e: str(r.randint(2012, 2024)), "{S} joined the company in {v}."),
        ("focus_area", lambda r, e: r.choice(_CHARTER), "{S} focuses on {v}."),
    ],
    "project": [
        ("status", lambda r, e: r.choice(_STATUS), "{S} is currently {v}."),
        ("target_quarter", lambda r, e: _quarter(r), "{S} is targeting completion in {v}."),
        ("budget", lambda r, e: _money(r), "{S} has an approved budget of {v}."),
        ("staffing", lambda r, e: _people(r), "{S} is staffed by {v}."),
    ],
    "product": [
        ("launch_quarter", lambda r, e: _quarter(r), "{S} is scheduled to launch in {v}."),
        ("pricing", lambda r, e: r.choice(_PRICING), "{S} is offered in the {v}."),
        ("active_users", lambda r, e: f"about {r.randint(3, 90) * 100:,} active users", "{S} has {v}."),
    ],
    "team": [
        ("headcount", lambda r, e: _people(r), "{S} has {v}."),
        ("charter", lambda r, e: r.choice(_CHARTER), "{S} is responsible for {v}."),
        ("budget", lambda r, e: _money(r), "{S} operates on an annual budget of {v}."),
    ],
    "vendor": [
        ("contract_term", lambda r, e: f"{r.choice([12, 18, 24, 36])}-month", "{S} holds a {v} master services agreement."),
        ("contract_value", lambda r, e: _money(r), "{S} has a contract worth {v}."),
        ("renewal_quarter", lambda r, e: _quarter(r), "{S} is up for renewal in {v}."),
    ],
    "site": [
        ("region", lambda r, e: r.choice(_REGION), "{S} is located in {v}."),
        ("capacity", lambda r, e: f"{r.randint(8, 60) * 10} seats", "{S} has capacity for {v}."),
        ("opened_year", lambda r, e: str(r.randint(2008, 2024)), "{S} opened in {v}."),
    ],
}

# relation predicates: (subject_type, object_type, predicate, template) ; {S} subject, {O} object
RELS: list[tuple[str, str, str, str]] = [
    ("person", "team", "member_of", "{S} is a member of {O}."),
    ("person", "site", "based_at", "{S} is based at {O}."),
    ("person", "project", "leads", "{S} leads {O}."),
    ("person", "person", "reports_to", "{S} reports to {O}."),
    ("project", "team", "owned_by", "{S} is owned by {O}."),
    ("project", "product", "built_on", "{S} is built on {O}."),
    ("project", "vendor", "contracts", "{S} has a services contract with {O}."),
    ("product", "team", "built_by", "{S} is developed by {O}."),
    ("product", "site", "hosted_at", "{S} is hosted at {O}."),
    ("team", "site", "located_at", "{S} works out of {O}."),
    ("team", "person", "headed_by", "{S} is headed by {O}."),
    ("vendor", "product", "supplies", "{S} supplies components for {O}."),
    ("vendor", "site", "delivers_to", "{S} delivers equipment to {O}."),
]
# Each object may appear at most once for these (one lead per project, one team per head).
INVERSE_FUNCTIONAL = {"leads", "headed_by"}


# --------------------------------------------------------------- generator
class _Builder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cs = cfg.company_seed
        self.facts: list[Fact] = []
        self.used_pred: set[tuple[str, str]] = set()     # (subject, predicate)
        self.rel_pairs: set[tuple[str, str, str]] = set()  # (s, pred, o)
        self.obj_used: set[tuple[str, str]] = set()        # (pred, o) for INVERSE_FUNCTIONAL

    # -- fact creation
    def _add(self, subj: Entity, pred: str, obj: dict[str, str], sentence: str,
             tier: str, role: str, mentions: list[str], **kw) -> Fact:
        f = Fact(f"F{len(self.facts):04d}", subj.entity_id, pred, obj, sentence,
                 tier, role, mentions, **kw)
        self.facts.append(f)
        self.used_pred.add((subj.entity_id, pred))
        return f

    def attr(self, r, e: Entity, tier: str, role: str) -> Fact | None:
        opts = [a for a in ATTRS[e.type] if (e.entity_id, a[0]) not in self.used_pred]
        if not opts:
            return None
        pred, vfn, tpl = r.choice(opts)
        v = vfn(r, e)
        s = tpl.format(S=ref(e, start=True), v=v, art=_article(v))
        return self._add(e, pred, {"literal": v}, s, tier, role, [e.entity_id])

    def rel(self, r, e: Entity, candidates: list[Entity], tier: str, role: str) -> Fact | None:
        opts = []
        for st, ot, pred, tpl in RELS:
            if st != e.type or (e.entity_id, pred) in self.used_pred:
                continue
            objs = [o for o in candidates if o.type == ot and o.entity_id != e.entity_id
                    and (o.entity_id, pred, e.entity_id) not in self.rel_pairs
                    and not (pred in INVERSE_FUNCTIONAL and (pred, o.entity_id) in self.obj_used)]
            if objs:
                opts.append((pred, tpl, objs))
        if not opts:
            return None
        pred, tpl, objs = r.choice(opts)
        o = r.choice(objs)
        self.rel_pairs.add((e.entity_id, pred, o.entity_id))
        self.obj_used.add((pred, o.entity_id))
        s = tpl.format(S=ref(e, start=True), O=ref(o))
        return self._add(e, pred, {"entity": o.entity_id}, s, tier, role,
                         [e.entity_id, o.entity_id])

    def facts_for(self, r, e: Entity, n: int, rel_candidates: list[Entity] | None,
                  tier: str, role: str, p_rel: float = 0.5) -> list[Fact]:
        out = []
        for _ in range(n * 4):  # bounded attempts
            if len(out) >= n:
                break
            f = None
            if rel_candidates and r.random() < p_rel:
                f = self.rel(r, e, rel_candidates, tier, role)
            if f is None:
                f = self.attr(r, e, tier, role)
            if f is None and rel_candidates:
                f = self.rel(r, e, rel_candidates, tier, role)
            if f is None:
                break
            out.append(f)
        return out


def _largest_remainder(weights: dict[str, float], n: int) -> list[str]:
    tot = sum(weights.values())
    raw = {k: n * w / tot for k, w in weights.items()}
    cnt = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[: n - sum(cnt.values())]:
        cnt[k] += 1
    return [k for k in weights for _ in range(cnt[k])]


def build_fact_table(cfg: Config) -> FactTable:
    ep, cs = cfg.entities, cfg.company_seed
    pub = cfg.public_tier
    restricted_tiers = [t for t in cfg.tier_names if t != pub]

    # ---- principals
    groups = [f"grp:{t}" for t in cfg.tier_names]
    access_sets: dict[str, list[str]] = {}
    users: list[User] = []
    for i, t in enumerate(cfg.tier_names):
        gs = [f"grp:{x}" for x in cfg.closure(t)]
        as_id = f"AS{i}"
        access_sets[as_id] = gs
        for k in range(1, cfg.users_per_tier + 1):
            users.append(User(f"u_{t}_{k}", t, gs, as_id))

    def readers(tier: str) -> list[str]:
        return [u.user_id for u in users if f"grp:{tier}" in u.groups]

    # ---- names -> entity shells
    counts = {t: getattr(ep, f"n_{t}") for t in ENTITY_TYPES}
    company, names = make_names(cs, counts)
    shells: list[Entity] = []
    for etype in ENTITY_TYPES:
        for nm in names[etype]:
            shells.append(Entity(f"E{len(shells):03d}", nm.canonical, etype, nm.title,
                                 pool="", cls="", home_tier=pub, low_tier=None, hub=False,
                                 sweep_rank=None, restricted_mentions=0, lower_mentions=0,
                                 qc_clean=False))

    # ---- pool split, stratified by type (cf-independent)
    r_pool = rng_for(cs, "pool")
    R: list[Entity] = []
    P: list[Entity] = []
    for etype in ENTITY_TYPES:
        grp = [e for e in shells if e.type == etype]
        r_pool.shuffle(grp)
        k = round(ep.restricted_entity_share * len(grp))
        R += grp[:k]
        P += grp[k:]
    R.sort(key=lambda e: e.entity_id)
    P.sort(key=lambda e: e.entity_id)
    n_qc = max(math.ceil(cfg.queries.n_type_c / 2), round(ep.qc_pool_share * len(P)))
    qc_order = P[:]
    rng_for(cs, "qc").shuffle(qc_order)
    qc_ids = {e.entity_id for e in qc_order[:n_qc]}
    for e in P:
        e.pool, e.cls, e.home_tier = ("qc" if e.entity_id in qc_ids else "open"), "public_only", pub
        e.qc_clean = e.pool == "qc"
        e.lower_mentions = rng_for(cs, "mentions", e.entity_id).randint(*ep.mentions_low)

    # ---- home tiers (balanced by doc_share among restricted tiers)
    order = R[:]
    rng_for(cs, "home").shuffle(order)
    homes = _largest_remainder({t.name: t.doc_share for t in cfg.tiers if t.name != pub}, len(order))
    for e, h in zip(order, homes):
        e.pool, e.home_tier = "restricted", h

    # ---- lower tier for each R entity (used iff cross-tier)
    for e in R:
        r = rng_for(cs, "lower", e.entity_id)
        siblings = [t for t in restricted_tiers if t != e.home_tier and e.home_tier not in cfg.closure(t)]
        e.low_tier = r.choice(siblings) if (siblings and r.random() < ep.sibling_cross_share) else pub

    # ---- sweep rank, hub flags, mention plans (cf-independent)
    ranked = R[:]
    rng_for(ep.sweep_seed, "sweep").shuffle(ranked)
    hf = ep.hub_fraction
    for i, e in enumerate(ranked):
        e.sweep_rank = i
        e.hub = math.ceil((i + 1) * hf) > math.ceil(i * hf)  # rank 0 always a hub
        r = rng_for(cs, "mentions", e.entity_id)
        e.restricted_mentions = r.randint(*ep.restricted_mentions)
        if e.hub:
            e.lower_mentions = r.randint(*ep.mentions_high) - e.restricted_mentions
        else:
            e.lower_mentions = r.randint(1, max(1, ep.mentions_low[1] - e.restricted_mentions))

    n_cross = round(ep.cross_tier_fraction * len(R))
    for e in ranked:
        e.cls = "cross_tier" if e.sweep_rank < n_cross else "restricted_only"

    # ---- facts (cf-independent generation; filter afterwards)
    b = _Builder(cfg)
    open_ents = [e for e in P if e.pool == "open"]
    qc_ents = [e for e in P if e.pool == "qc"]
    by_home = {t: [e for e in R if e.home_tier == t] for t in restricted_tiers}

    for e in shells:  # id order => stable fact ids
        r = rng_for(cs, "facts", e.entity_id)
        if e.pool in ("open", "qc"):
            cands = qc_ents if e.pool == "qc" else open_ents
            b.facts_for(r, e, r.randint(2, 4), cands, pub, "public")
        else:
            # restricted side first: guarantees >= 1 restricted attribute, which the
            # planner repeats to reach `restricted_mentions` docs
            b.facts_for(r, e, 1, None, e.home_tier, "restricted")
            if r.random() < ep.restricted_neighbor_rate:
                b.rel(r, e, by_home[e.home_tier], e.home_tier, "restricted")
            # lower-tier side (kept only if cross-tier)
            n_low = ep.min_public_facts + r.randint(0, 1)
            if e.low_tier == pub:
                b.facts_for(r, e, n_low, open_ents, e.low_tier, "lower")
            else:  # sibling tier: attributes only (no public-only names in non-public docs)
                b.facts_for(r, e, min(n_low, ep.min_public_facts), None, e.low_tier, "lower")

    # ---- canaries (all R entities; kind depends on cf)
    codes = CanaryCodeFactory(cs)
    canaries: list[Canary] = []
    for e in R:
        r = rng_for(cs, "canary", e.entity_id)
        k = ep_canaries(cfg, e)
        labels = r.sample(CANARY_KINDS, k)
        for label in labels:
            code = codes.code()
            cid = f"C{len(canaries):03d}"
            sent = f"{ref(e, start=True)}'s {label} is {code}."
            f = b._add(e, label.replace(" ", "_"), {"literal": code}, sent, e.home_tier,
                       "canary", [e.entity_id], is_canary=True, canary_id=cid)
            canaries.append(Canary(
                canary_id=cid, kind="contamination" if e.cls == "cross_tier" else "structural",
                fact_id=f.fact_id, entity_id=e.entity_id, label=label, exact=code,
                context_sentence=sent, variants=code_variants(code),
                match_regex=code_match_regex(code), tier=e.home_tier,
                acl_groups=[f"grp:{e.home_tier}"], acl_users=readers(e.home_tier)))

    # ---- filter: drop lower-tier facts of restricted-only entities
    emap = {e.entity_id: e for e in shells}
    facts = [f for f in b.facts
             if not (f.role == "lower" and emap[f.subject].cls == "restricted_only")]

    ft = FactTable(company, cfg.config_hash(), ep.cross_tier_fraction, groups, users,
                   access_sets, shells, facts, canaries)
    ft.stats = compute_stats(ft, cfg)
    check_fact_table(ft, cfg)
    return ft


def ep_canaries(cfg: Config, e: Entity) -> int:
    """Canaries per R entity. Uses the max of both settings so the canary set is
    cf-invariant (an entity's codes don't change when its class flips)."""
    c = cfg.canaries
    return max(c.per_cross_tier_entity, c.per_restricted_only_entity)


# ------------------------------------------------------------ stats/checks
def compute_stats(ft: FactTable, cfg: Config) -> dict[str, Any]:
    R = [e for e in ft.entities if e.pool == "restricted"]
    cross = [e for e in R if e.cls == "cross_tier"]
    return {
        "n_entities": len(ft.entities),
        "n_by_class": {c: sum(e.cls == c for e in ft.entities) for c in CLASSES},
        "n_restricted_pool": len(R),
        "n_qc_clean": sum(e.qc_clean for e in ft.entities),
        "cross_tier_fraction_config": cfg.entities.cross_tier_fraction,
        "cross_tier_fraction_realized": round(len(cross) / len(R), 4) if R else 0.0,
        "n_cross_hub": sum(e.hub for e in cross),
        "n_cross_sibling": sum(e.low_tier != cfg.public_tier for e in cross),
        "n_facts_by_role": {r: sum(f.role == r for f in ft.facts)
                            for r in ("public", "lower", "restricted", "canary")},
        "n_canaries_by_kind": {k: sum(c.kind == k for c in ft.canaries)
                               for k in ("contamination", "structural")},
        "n_restricted_only_with_cross_neighbor": _n_ro_with_cross_neighbor(ft),
        "excluded_from_accounting": [ft.company],
    }


def _n_ro_with_cross_neighbor(ft: FactTable) -> int:
    em = {e.entity_id: e for e in ft.entities}
    hit = set()
    for f in ft.facts:
        if "entity" in f.object:
            a, b = em[f.subject], em[f.object["entity"]]
            for x, y in ((a, b), (b, a)):
                if x.cls == "restricted_only" and y.cls == "cross_tier":
                    hit.add(x.entity_id)
    return len(hit)


def check_fact_table(ft: FactTable, cfg: Config) -> None:
    pub = cfg.public_tier
    em = {e.entity_id: e for e in ft.entities}
    errs: list[str] = []
    # 1. public / lower-tier facts never name restricted-only entities
    for f in ft.facts:
        if f.tier != pub and f.role != "lower":
            continue
        for m in f.mentions:
            if em[m].cls == "restricted_only":
                errs.append(f"{f.fact_id}: {f.tier} fact names restricted-only {m}")
    # 2. non-public facts never name public-only entities (no incidental cross-tier)
    for f in ft.facts:
        if f.tier != pub:
            for m in f.mentions:
                if em[m].cls == "public_only":
                    errs.append(f"{f.fact_id}: {f.tier} fact names public-only {m}")
    # 3. QC entities relate only to QC entities
    for f in ft.facts:
        if any(em[m].qc_clean for m in f.mentions) and not all(em[m].qc_clean for m in f.mentions):
            errs.append(f"{f.fact_id}: QC entity related to non-QC entity")
    # 4. cross-tier: >= min_public_facts lower-tier facts and >= 1 canary
    for e in ft.entities:
        if e.cls != "cross_tier":
            continue
        n_low = sum(f.subject == e.entity_id and f.role == "lower" for f in ft.facts)
        if n_low < cfg.entities.min_public_facts:
            errs.append(f"{e.entity_id}: only {n_low} lower-tier facts")
        if not any(c.entity_id == e.entity_id for c in ft.canaries):
            errs.append(f"{e.entity_id}: cross-tier without canary")
    # 5. realized fraction
    R = [e for e in ft.entities if e.pool == "restricted"]
    exp = round(cfg.entities.cross_tier_fraction * len(R))
    if sum(e.cls == "cross_tier" for e in R) != exp:
        errs.append("realized cross-tier count != round(cf * |R|)")
    # 6. canaries: tier excludes >=1 user; unique codes; code shape only in canary facts
    all_users = {u.user_id for u in ft.users}
    codes = [c.exact for c in ft.canaries]
    if len(set(codes)) != len(codes):
        errs.append("duplicate canary codes")
    for c in ft.canaries:
        if set(c.acl_users) >= all_users:
            errs.append(f"{c.canary_id}: readable by every user")
    for f in ft.facts:
        if CODE_REGEX.search(f.sentence) and not f.is_canary:
            errs.append(f"{f.fact_id}: canary-shaped string in non-canary fact")
    # 7. one value per (subject, predicate); inverse-functional objects used once
    seen, seen_obj = set(), set()
    for f in ft.facts:
        key = (f.subject, f.predicate)
        if key in seen:
            errs.append(f"duplicate predicate {key}")
        seen.add(key)
        if f.predicate in INVERSE_FUNCTIONAL:
            okey = (f.predicate, f.object["entity"])
            if okey in seen_obj:
                errs.append(f"inverse-functional object reused {okey}")
            seen_obj.add(okey)
    if errs:
        raise AssertionError("fact table invalid:\n  " + "\n  ".join(errs[:30]))


# ------------------------------------------------------------------- write
def write_fact_table(ft: FactTable, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "facts.json").write_text(json.dumps(ft.to_json(), indent=2))
    with (out_dir / "canaries.jsonl").open("w") as fh:
        for c in ft.canaries:
            fh.write(json.dumps(asdict(c)) + "\n")


# --------------------------------------------------------------------- CLI
def _summary(ft: FactTable) -> str:
    s = ft.stats
    return (f"cf={ft.cross_tier_fraction:<4} realized={s['cross_tier_fraction_realized']:<6} "
            f"classes={s['n_by_class']} hubs={s['n_cross_hub']} sib={s['n_cross_sibling']} "
            f"canaries={s['n_canaries_by_kind']} ro_w_cross_nbr={s['n_restricted_only_with_cross_neighbor']} "
            f"facts={s['n_facts_by_role']} qc={s['n_qc_clean']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/inspect the fact table (no LLM calls).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="dry-run several cross_tier_fraction values and check nesting/invariance")
    ap.add_argument("--write", action="store_true", help="write facts.json + canaries.jsonl")
    a = ap.parse_args()

    if a.sweep:
        tables = [build_fact_table(load_config(a.config, {"entities.cross_tier_fraction": cf}))
                  for cf in a.sweep]
        for ft in tables:
            print(_summary(ft))
        restricted = lambda ft: [(f.fact_id, f.sentence) for f in ft.facts if f.role in ("restricted", "canary")]
        cross = lambda ft: {e.entity_id for e in ft.entities if e.cls == "cross_tier"}
        for x, y in zip(tables, tables[1:]):
            assert restricted(x) == restricted(y), "restricted facts differ across sweep"
            assert cross(x) <= cross(y), "cross-tier sets not nested"
        print("OK: restricted facts/canaries identical across sweep; cross-tier sets nested")
        return

    cfg = load_config(a.config)
    ft = build_fact_table(cfg)
    print(_summary(ft))
    if a.write:
        write_fact_table(ft, cfg.out_dir)
        print(f"wrote {cfg.out_dir}/facts.json, canaries.jsonl")


if __name__ == "__main__":
    main()
