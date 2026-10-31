"""Config schema, loader, and deterministic RNG helpers for corpus generation.

Pinned against lightrag-hku==1.5.7 (see `LightRAGParams`).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ----------------------------------------------------------------- sections
@dataclass
class TierSpec:
    """One access tier == one group `grp:<name>`.

    `includes` lists tiers whose docs this tier can also read. A user of tier T
    holds `grp:X` for every X in closure(T); a doc written at tier L has
    acl_groups = [grp:L]. Access: user.groups ∩ doc.acl_groups != ∅.
    """
    name: str
    includes: list[str] = field(default_factory=list)
    doc_share: float = 0.25  # fraction of docs authored at this tier


@dataclass
class EntityParams:
    # counts per entity type (scaled from these, not inferred from n_docs)
    n_person: int = 24
    n_project: int = 10
    n_product: int = 6
    n_team: int = 6
    n_vendor: int = 5
    n_site: int = 4
    # cross_tier_fraction = |cross_tier| / |entities appearing in any restricted doc|
    cross_tier_fraction: float = 0.3
    # Fraction of the entity pool that lives in restricted docs at all.
    restricted_entity_share: float = 0.5
    # Nested sweep: cross-tier entities are a prefix of a fixed permutation of
    # restricted-tier entities (seeded by `sweep_seed`), so the sets at 0.1 ⊂ 0.3 ⊂ 0.5.
    sweep_seed: int = 1234
    # Fraction of restricted-tier entities given a same-home-tier relation to another
    # restricted-tier entity. When the partner is cross-tier, 1-hop expansion from a
    # visible node reaches the restricted-only one -> nonzero structural leakage.
    restricted_neighbor_rate: float = 0.6
    # Share of cross-tier entities whose lower-tier side is a sibling restricted tier
    # (e.g. home team_a, also mentioned in team_b docs) instead of public.
    sibling_cross_share: float = 0.25
    # Share of public-only entities reserved as type-(c) QC targets: no relation, at
    # any tier, to a restricted-tier entity (so 1-hop expansion stays clean).
    qc_pool_share: float = 0.3
    # Mention regimes (number of distinct docs mentioning a cross-tier entity).
    # LightRAG 1.5.7 LLM-summarizes merged descriptions only when
    # fragments >= force_llm_summary_on_merge (8); below that it joins with <SEP>.
    # Hubs land in the summarized regime, the rest in the concatenated regime.
    hub_fraction: float = 0.3
    # Docs per restricted-pool entity at its home tier (cf-invariant).
    restricted_mentions: tuple[int, int] = (2, 3)
    mentions_low: tuple[int, int] = (2, 5)
    mentions_high: tuple[int, int] = (9, 12)
    # Public facts per type-(a)/(c) target entity (non-trivial honest answers).
    min_public_facts: int = 2


@dataclass
class CanaryParams:
    per_cross_tier_entity: int = 1        # contamination canaries
    per_restricted_only_entity: int = 1   # structural-leakage control canaries
    # Single regex family for every canary code; nothing else emits this shape.
    code_regex: str = r"\b[A-Z]{2}-\d{4}-[A-Z]\b"


@dataclass
class DocParams:
    n_docs: int = 50
    words_min: int = 300
    words_max: int = 1200
    doc_types: tuple[str, ...] = ("memo", "meeting_notes", "wiki_page", "status_report", "email_thread")


@dataclass
class QueryParams:
    n_type_a: int = 12   # cross-tier contamination probes
    n_type_b: int = 8    # restricted-only structural probes
    n_type_c: int = 8    # public-only QC
    # Run every query for every user (exec = positive controls).
    all_users: bool = True


@dataclass
class LLMParams:
    provider: str = "anthropic"
    model: str = "claude-sonnet-5-5"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 4
    concurrency: int = 4
    cache_dir: str = ".llm_cache"
    # provider: ollama (ignored by other providers; not part of the cache key)
    host: str = "http://localhost:11434"
    num_ctx: int = 16384      # prompt + doc; Ollama's default context silently truncates
    think: bool = False       # Qwen3 reasoning would land in the document text
    timeout: int = 600        # seconds per request
    seed: int | None = None   # sampling seed; batched decoding is still not bit-exact


@dataclass
class LightRAGParams:
    """Mirror of lightrag-hku==1.5.7 defaults, used for planning/validation only."""
    version: str = "1.5.7"
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"
    force_llm_summary_on_merge: int = 8
    summary_max_tokens: int = 1200


@dataclass
class Config:
    run_name: str
    seed: int = 42
    company_seed: int | None = None  # defaults to seed; fixes the fictional company/names
    out_root: str = "corpus"
    tiers: list[TierSpec] = field(default_factory=list)
    users_per_tier: int = 2
    entities: EntityParams = field(default_factory=EntityParams)
    canaries: CanaryParams = field(default_factory=CanaryParams)
    docs: DocParams = field(default_factory=DocParams)
    queries: QueryParams = field(default_factory=QueryParams)
    llm: LLMParams = field(default_factory=LLMParams)
    lightrag: LightRAGParams = field(default_factory=LightRAGParams)

    # ------------------------------------------------------------- derived
    @property
    def out_dir(self) -> Path:
        return Path(self.out_root) / self.run_name

    @property
    def tier_names(self) -> list[str]:
        return [t.name for t in self.tiers]

    @property
    def public_tier(self) -> str:
        roots = [t.name for t in self.tiers if not t.includes]
        if len(roots) != 1:
            raise ValueError(f"exactly one tier must have no `includes` (public); got {roots}")
        return roots[0]

    def closure(self, tier: str) -> list[str]:
        """Tiers readable by a member of `tier` (including itself), in config order."""
        by_name = {t.name: t for t in self.tiers}
        seen: set[str] = set()
        stack = [tier]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(by_name[cur].includes)
        return [n for n in self.tier_names if n in seen]

    def readers_of(self, tier: str) -> list[str]:
        """Tiers whose members can read docs authored at `tier`."""
        return [t for t in self.tier_names if tier in self.closure(t)]

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def validate(self) -> None:
        names = self.tier_names
        if len(set(names)) != len(names):
            raise ValueError("duplicate tier names")
        for t in self.tiers:
            for inc in t.includes:
                if inc not in names:
                    raise ValueError(f"tier {t.name} includes unknown tier {inc}")
        _ = self.public_tier
        for t in self.tiers:  # cycle check
            if any(t.name in self.closure(i) for i in t.includes):
                raise ValueError(f"cycle in tier includes at {t.name}")
        share = sum(t.doc_share for t in self.tiers)
        if abs(share - 1.0) > 1e-6:
            raise ValueError(f"tier doc_share must sum to 1, got {share}")
        e = self.entities
        if not 0.0 <= e.cross_tier_fraction <= 1.0:
            raise ValueError("cross_tier_fraction must be in [0, 1]")
        if e.cross_tier_fraction == 1.0:
            raise ValueError("cross_tier_fraction=1.0 leaves no restricted-only entities (no structural control)")
        if e.mentions_high[0] < self.lightrag.force_llm_summary_on_merge:
            raise ValueError("mentions_high[0] must be >= force_llm_summary_on_merge to reach the summarized regime")
        if e.mentions_low[1] >= self.lightrag.force_llm_summary_on_merge:
            raise ValueError("mentions_low[1] must be < force_llm_summary_on_merge to stay in the concatenated regime")
        if e.restricted_mentions[1] >= e.mentions_low[1]:
            raise ValueError("restricted_mentions max must be < mentions_low max (non-hubs need >= 1 bridge doc)")
        if not self.docs.words_min < self.docs.words_max:
            raise ValueError("words_min must be < words_max")
        if self.users_per_tier < 1:
            raise ValueError("users_per_tier must be >= 1")


# ------------------------------------------------------------------ loading
def _build(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    for k, v in list(data.items()):
        if isinstance(v, list) and k in {"mentions_low", "mentions_high", "restricted_mentions", "doc_types"}:
            data[k] = tuple(v)
    return cls(**data)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML; `overrides` uses dotted keys, e.g. {"entities.cross_tier_fraction": 0.5}."""
    raw = yaml.safe_load(Path(path).read_text())
    for dotted, val in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = val
    cfg = Config(
        run_name=raw["run_name"],
        seed=raw.get("seed", 42),
        company_seed=raw.get("company_seed"),
        out_root=raw.get("out_root", "corpus"),
        tiers=[TierSpec(**t) for t in raw["tiers"]],
        users_per_tier=raw.get("users_per_tier", 2),
        entities=_build(EntityParams, raw.get("entities")),
        canaries=_build(CanaryParams, raw.get("canaries")),
        docs=_build(DocParams, raw.get("docs")),
        queries=_build(QueryParams, raw.get("queries")),
        llm=_build(LLMParams, raw.get("llm")),
        lightrag=_build(LightRAGParams, raw.get("lightrag")),
    )
    if cfg.company_seed is None:
        cfg.company_seed = cfg.seed
    cfg.run_name = cfg.run_name.format(
        cf=cfg.entities.cross_tier_fraction, seed=cfg.seed, n=cfg.docs.n_docs
    )
    cfg.validate()
    return cfg


def dump_config(cfg: Config, path: str | Path) -> None:
    d = asdict(cfg)
    d["config_hash"] = cfg.config_hash()
    Path(path).write_text(yaml.safe_dump(json.loads(json.dumps(d, default=list)), sort_keys=False))


# ---------------------------------------------------------------------- RNG
def rng_for(seed: int, *labels: str) -> random.Random:
    """Independent, reproducible RNG stream per (seed, label path).

    Adding a new stage never shifts the draws of existing stages.
    """
    h = hashlib.sha256(("|".join([str(seed), *labels])).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))
