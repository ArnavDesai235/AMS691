# Authorization-Aware LightRAG — Corpus, Canaries & Gold Data (Design Notes)

Owner: Arnav · Status: full 300-doc corpus built (qwen3), ingested, and evaluated on C1–C4 at cf = 0.3 (original + tuned retrieval); cf = 0.1 / 0.5 sweep not run · Pinned: `lightrag-hku==1.5.7`

## 1. Purpose

Synthetic multi-tenant corpus for measuring **structural leakage** (wrong node retrieved) vs **contamination** (authorized node whose merged description carries restricted text) across C1–C4. All ground truth (entities, facts, ACLs, canaries) is generated in code first; an LLM only verbalizes known facts into prose.

## 2. Access model

| Tier | Group | Can read docs of |
|---|---|---|
| public | `grp:public` | public |
| team_a | `grp:team_a` | public, team_a |
| team_b | `grp:team_b` | public, team_b |
| exec | `grp:exec` | all tiers |

- 2 users per tier (`u_public_1`, `u_team_a_1`, …). Users of a tier share one access set (AS0–AS3).
- **One ACL per document.** Every chunk inherits its doc's ACL, so a canary's origin chunk always has an unambiguous ACL.
- `manifest.jsonl` stores both `acl_groups` and the expanded `acl_users`, so enforcement code can test `u ∈ ACL(c)` directly.

## 3. Entity classes and canaries

| Class | Appears in | Purpose |
|---|---|---|
| public_only | public docs only | baseline; a reserved "QC" subset (no relations to restricted entities) are type-(c) targets |
| cross_tier | lower tier (usually public, sometimes sibling team) + home restricted tier | **contamination probe**: LightRAG merges both into one node |
| restricted_only | home restricted tier only | **structural leakage control** |

- **Canary** = unique fictional code (e.g. `TM-1841-G`) planted exactly once, in one restricted doc, in the same sentence as its entity's full name. Contamination canaries sit on cross-tier entities; structural canaries on restricted-only entities.
- Each canary record carries exact string, spelling variants, a tolerant regex (`match_regex`), origin doc + char span, ACL, entity.
- `cross_tier_fraction` (cf) = |cross_tier| / |entities in any restricted doc|. Sweep values 0.1 / 0.3 / 0.5.
- **Sweep is nested**: cross-tier sets satisfy 0.1 ⊂ 0.3 ⊂ 0.5; restricted facts, canary codes and canary→doc mapping are identical across cf. Only public-side "bridge" mentions change.

## 4. LightRAG 1.5.7 behaviour the corpus is designed around

| Finding | Source | Consequence |
|---|---|---|
| Entities merge by **exact, case-sensitive name** (no case folding) | `normalize_extracted_info` (utils.py) | Docs must always use full canonical names; generator enforces this |
| Merged description is **LLM-summarized only at ≥ 8 fragments** (or ≥ 1200 tokens); otherwise fragments are joined with `<SEP>` | `_handle_entity_relation_summary`, `FORCE_LLM_SUMMARY_ON_MERGE=8` | Two regimes: **hubs** (9+ mentioning docs → summarized blend) vs **non-hubs** (≤ 5 → concatenated). Report contamination per regime |
| Fixed-token chunking, 1200 tokens / 100 overlap, tiktoken `gpt-4o-mini` | `chunker/token_size.py` | Docs > ~900 words span 2 chunks; a canary in the overlap can sit in both |
| `ainsert(ids=...)` accepts our doc IDs; chunk store records parent doc | `lightrag.py` | ACL tagging at chunking = lookup of `full_doc_id` in our manifest |
| Default entity types have no Project/Product type | `prompt.py` | Irrelevant to merging (name-only) |

**Notes for C4 (Sai):** below 8 fragments the stored description is already `<SEP>`-separated, but fragment order is by timestamp/length, not aligned with `source_id` order, so per-fragment source attribution is not recoverable from the stored node — it must be captured at merge time.

**Note for analysis:** the planned hub flag is a prediction; group entities by the **realized `source_id` count after ingest** (≥ 8 vs < 8).

## 5. Leak-hygiene rules (so metrics are not trivially 0 or polluted)

- Non-public facts never name public-only entities (prevents accidental cross-tier entities). The company name is the one unavoidable exception; it is excluded from all accounting.
- Restricted relations link restricted-pool entities with the same home tier; some restricted-only entities neighbour cross-tier ones so **1-hop expansion can reach them** (nonzero structural leakage in C1/C2).
- Type-(c) QC entities relate only to other QC entities (1-hop stays clean).
- Canary codes use one regex family nothing else emits; codes are alphanumeric so they survive summarization better than prose facts.
- Exec users can read every canary → **positive controls** (recall), excluded from leakage/contamination denominators. Report per tier.

## 6. Pipeline (`corpus_gen/`)

| Step | File | Output |
|---|---|---|
| Config + seeded RNG | `config.py`, `configs/dev.yaml` (50 docs), `configs/full.yaml` (300 docs) | — |
| Fictional names, canary codes | `names.py` (rejects dictionary words and any token found in real web text) | — |
| Ground-truth fact table | `facts.py` | `facts.json`, `canaries.jsonl` |
| Fact → document plan | `plan.py` | `plan.json` |
| LLM backends + cache | `llm.py` (`claude_code` / `anthropic` / `ollama` / `mock`) | `.llm_cache/` |
| Deterministic draft repair | `repair.py` (expand shortened names, dedupe/anchor canary codes, insert missing fact sentences) | — |
| Verbalize + hard checks | `verbalize.py` | `docs/*.txt`, `manifest.jsonl`, `gen_report.json` |

Each draft must pass: all planned names verbatim, no shortened names, no foreign/forbidden entity names, each canary exactly once in-sentence with its entity, all fact values present, **no unplanned restatement of another tier's literal value about a planned entity** (added in PR #21), no invented proper nouns/real brands, length in range. Failures are fed back for rewrite (max 6 attempts); a doc that never passes fails the run.

## 7. Output schemas (per run: `corpus/<run_name>/`)

- `manifest.jsonl`: `doc_id, title, doc_type, acl_label, acl_groups, acl_users, entities, entity_names, entity_mentions (char spans), fact_ids, bridge_fact_ids, canary_ids, n_words, sha256`
- `facts.json`: `principals {groups, users, access_sets}`, `entities [entity_id, canonical_name, type, cls, home_tier, low_tier, hub, ...]`, `facts [fact_id, subject, predicate, object, sentence, tier, role, mentions, is_canary, canary_id]`, `stats`
- `canaries.jsonl`: `canary_id, kind, fact_id, entity_id, label, exact, variants, match_regex, tier, acl_groups, acl_users, origin_doc_id, origin_char_span, origin_chunks (filled at ingest)`
- `queries.jsonl` (one row per query × user): `qid, query_id, type (a/b/c), text, target_entity, target_name, user_id, user_tier, access_set, authorized_fact_ids, gold_answer, gold_kind (facts|not_available), forbidden_canary_ids (ALL canaries the user cannot read = metric set), target_canary_ids, neighbor_canary_ids (attribution), authorized_canary_ids (canary recall), eval_role (probe|positive_control|qc)`
- `oracle_splits.json`: per access set `{groups, users, doc_ids}`, `user_to_access_set`, and `all` (shared index)
- `validation_report.json`: independent checks re-derived from the files (canary placement, stray codes, cross-tier fraction, entity placement from text, gold safety, oracle splits) + manual-review list

## 8. Current status

**Dev corpus `dev_cf0.3_s42`** — 50 docs (20 public, 10 per restricted tier), ~35k words, 46/50 accepted first try, 4 after one retry; 27 canaries (8 contamination, 19 structural). Independent check: every canary only in its origin doc at the recorded span; no forbidden names anywhere. Generation cost $1.27 via Claude Code.

**Full corpus `full_cf0.3_s42`** — generated on the cluster with `qwen3:32b-q8_0` via Ollama (no API cost): 300 docs (120 public, 60 per restricted tier), 300 entities (150 public_only, 45 cross_tier, 105 restricted_only), 300 canaries (90 contamination, 210 structural), 1,120 query rows. 228 docs passed on the first attempt, 63 on the second, 8 on the third and 1 on the fifth; 233 docs needed deterministic repairs. Ingested into LightRAG (`rag_storage/full_cf0.3_s42`; one public doc, `d-public-099`, still marked truncated). See `REPORT_qwen_corpus_and_ingest.md`.

**Full-run results (cf = 0.3, 840 rows with forbidden canaries; original → tuned retrieval)** — details in `results/full_cf0.3_s42{,_tuned}/` and `reports/retrieval_sweep.md`:

| | structural canary | contamination canary (in context) | answer canary | recall@10 |
|---|---|---|---|---|
| C1 | 1.00 → 0.97 | 0.83 → 0.47 | 0.48 → 0.52 | 0.60 → 0.75 |
| C2 | 0 → 0 | 0.88 → 0.47 | 0.11 → 0.10 | 0.66 → 0.81 |
| C3 | 0 → 0 | 0.75 → 0.49 | 0.10 → 0.10 | 0.66 → 0.78 |
| C4 | 0 → 0 | 0 → 0 | 0 → 0 | 0.67 → 0.77 |

Tuned retrieval (`configs/rag_full_tuned.yaml`) raises recall@10 by 10–15 points and cuts context contamination in C1–C3 by retrieving less; it does not reduce answer leakage in C1–C3. C4 has zero detected leakage on every measure. By realized `source_id` regime (≥ 8 = hub), canaries reach the retrieved context at similar rates (C2/C3 tuned: 0.47–0.59), but reach the *answer* about 4× more often for hubs (C2/C3 tuned: 0.27–0.29 vs 0.06–0.07; hub n = 126 rows, uncontrolled for entity/query type).

**Paraphrase check (in-session judge, 30 string-negative rows per config, tuned run):** judge found a paraphrased restricted fact in 8/30 C2 rows, 4/30 C3 and 0/30 C4 (`results/full_cf0.3_s42_tuned/judge_insession.json`). The pipeline judge (`acl_rag/judge.py`, sbatch step `judge`, job 99240) is queued and has not run.

## 9. Known limitations

- ~15% of restricted docs gain bridge facts at higher cf and are regenerated (their canaries stay in the same doc).
- Invented names built from real words (e.g. "Marrow Creek") can evade the invented-noun check; validator will list capitalized phrases for manual review.
- `origin_chunks` filled from LightRAG's chunk store at ingest (Week 3).
- cf = 0.1 has few contamination canaries (dev: 3; full: 30); interpret with care.
- Low-entropy fact values (`status` has 5 values; pricing 4; region 5) are easy filler for the writer. In `full_cf0.3_s42`, one restricted fact (F0675, team_b status "on track" for Project Barkil) is restated in team_a doc `d-team_a-032`; a scan of all 262 restricted non-canary facts found no other case (exact substring, so paraphrases are not caught). This caused C4's only two tuned unauthorized-fact flags; the corpus was not regenerated.
- The raw `answer_unauthorized_fact_rate` substring-matches restricted values anywhere in an answer, so it can over-count: it can also match text about another entity (original-C4 rows Qb011, "partner tier"). `answer_unauthorized_fact_rate_adj` (PR #21) ignores values the user's own readable docs state about the target; report both.
- Only the cf = 0.3 corpus exists. The nested cf sweep is untested end to end.

## 10. Next

- cf = 0.1 / 0.5 sweep: roughly 1–2 GPU-days per cf value uncached (full ingest took 13.4 h on 2× RTX 8000; dev C1–C4 took 7.4 h for 224 rows), so the 8 h default in `scripts/run_ollama_job.sbatch` must be raised. Seeding the new index with the cf = 0.3 LLM cache should avoid most re-extraction (untested). Cheapest informative option: cf = 0.5 only, C3 and C4 only.
- The new restricted-value check only applies to docs that are regenerated (~15% of restricted docs at other cf values); the F0675 doc is redone only if it gains bridge facts, so fix it explicitly or keep documenting the flag.
- Week 3: ACL tagging at chunking, union propagation, fill `origin_chunks` from LightRAG's chunk store.

## 11. Commands

```bash
pip install -r requirements.txt
python -m corpus_gen.facts   --config configs/full.yaml --sweep 0.1 0.3 0.5   # dry-run, no LLM
python -m corpus_gen.plan    --config configs/full.yaml --sweep 0.1 0.3 0.5   # dry-run, no LLM
python -m corpus_gen.run --config configs/dev.yaml                              # everything: docs, queries, gold, oracle, validation
python -m corpus_gen.run --config configs/full.yaml --set entities.cross_tier_fraction=0.5
python -m corpus_gen.validate corpus/dev_cf0.3_s42                             # re-validate a corpus
```
