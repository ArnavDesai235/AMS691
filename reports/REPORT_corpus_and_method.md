# Corpus and Method (report sections: Arnav)

Authorization-Aware LightRAG, AMS 691.01 Group 4. These are the "corpus and method" sections of the report. Numbers are from `results/full_cf0.3_s42{,_tuned}/`; figures are in `reports/figures/` (regenerate with `python scripts/make_plots.py`). Enforcement (C2/C3) and C4 are described in `DESIGN_enforcement.md` and `DESIGN_c4.md`.

## 1. Problem and measurement design

LightRAG builds one shared knowledge graph. Entities with the same name are merged across documents, and each merged node keeps a `source_id` listing the chunks it came from. Retrieval has no notion of who is asking. We separate two ways this can leak a document the user may not read:

- **Structural leakage:** a retrieved item (entity, relation or chunk) is one the user is not allowed to read.
- **Contamination:** the retrieved item is one the user *is* allowed to read, but its merged description contains text from a source the user cannot read. This arises only from merging, so it is invisible to item-level access checks.

Both are measured on planted **canaries** (fictional codes such as `TM-1841-G`), and again on the generated answer.

## 2. Access model and ACL propagation

Four tiers: `public`, `team_a`, `team_b`, `exec`. A user in `team_a` reads public and `team_a` documents, likewise for `team_b`; `exec` reads everything. There are two users per tier (8 users, 4 access sets). Every document has exactly one ACL, and every chunk inherits it from its document (a lookup of the chunk's `full_doc_id` in `manifest.jsonl`), so the origin chunk of any canary has an unambiguous ACL.

For a merged entity or relation `v` with source chunks `S(v)`:

> ACL(v) = union over c in S(v) of ACL(c)

and `v` is eligible for user `u` iff `u` is in ACL(v), i.e. `u` can read at least one source of `v`. We chose union over intersection because intersection would hide every cross-tier entity from most users. The cost of union is exactly contamination: an eligible node's description may still contain text from sources `u` cannot read. The union is computed at query time from `source_id` and the manifest (`acl_rag/acl.py`, `union_acl`); it is not stored on the nodes.

## 3. Synthetic corpus

All ground truth (entities, facts, ACLs, canaries) is generated in code first; an LLM only verbalizes known facts into prose. Entities fall in three classes:

| Class | Appears in | Purpose |
|---|---|---|
| public_only | public documents | baseline; a reserved subset with no links to restricted entities forms the clean QC queries |
| cross_tier | a lower tier and its home restricted tier | contamination probe: LightRAG merges both into one node |
| restricted_only | home restricted tier only | structural-leakage control |

`cross_tier_fraction` (cf) is the share of entities in restricted documents that are cross-tier. The corpus evaluated here uses cf = 0.3. The design supports a nested sweep (0.1, 0.3, 0.5 with identical restricted facts and canary codes); **only cf = 0.3 has been built and evaluated**.

**Size (cf = 0.3, seed 42).** 300 documents (120 public, 60 per restricted tier), 300 entities (150 public-only, 45 cross-tier, 105 restricted-only), 300 canaries (90 contamination canaries on cross-tier entities, 210 structural canaries on restricted-only entities), 1,120 query rows.

**Canaries.** Each canary is planted exactly once, in one restricted document, in the same sentence as its entity's full name. Each record stores the exact string, spelling variants, a tolerant regex, the origin document and character span, and the ACL. Codes use one pattern that nothing else in the corpus emits.

**Generation.** Names and codes come from a seeded generator that rejects dictionary words. Documents were written by `qwen3:32b-q8_0` (Ollama) from a fact plan, with hard checks on every draft: all planned names verbatim, no shortened names, no forbidden entity names, each canary exactly once in its sentence, all planned fact values present, no unplanned restatement of another tier's value about a planned entity, no invented proper nouns, length in range. Failed drafts are rewritten (up to 6 attempts), after a deterministic repair pass for shortened names and missing fact sentences. Of 300 documents, 228 passed on the first attempt, 63 on the second, 8 on the third and 1 on the fifth; 233 needed a deterministic repair. An independent validator re-derives canary placement, stray codes, the realized cross-tier fraction and entity placement from the files and reports no errors.

**Queries and gold answers.** One row per (query, user): type a (cross-tier entity; contamination probe for users below its home tier), type b (restricted-only entity; gold answer "not available" for users without access), type c (clean public-only entity; QC). Queries use fixed templates with the exact canonical name, so retrieval reliably reaches the target and a leak is attributable to merged-node content rather than retrieval drift. Gold answers are written from only the facts the user may read and hard-checked against forbidden canaries and unauthorized values. The 8 users give 480 rows for type a and 320 each for types b and c.

## 4. Metrics

For each (query, user) row let F be the canaries the user cannot read. Rates are over the 840 rows of non-`exec` users (`exec` can read every canary and acts as a positive control for recall). Each rate has a 95% bootstrap interval over rows.

- **structural canary:** a canary in F appears in the text of a retrieved item the user cannot read.
- **contamination canary:** a canary in F appears in the name, description or keywords of a retrieved item the user *can* read.
- **answer canary:** a canary in F appears in the generated answer (tolerant regex). Split by whether it was in the context.
- **unauthorized-fact rate:** the answer repeats a restricted fact value about the target. The raw version substring-matches anywhere in the answer; the *adjusted* version ignores values that the user's own readable documents already state about the target (a corpus labelling artefact, not a permission failure).
- Also reported: recall@k, answer accuracy, and latency.

Rows are also sliced by the entity's realized merge regime. LightRAG summarizes a merged description with the LLM only at 8 or more fragments; below that the fragments are concatenated. We call an entity a **hub** if its realized `source_id` count after ingest is at least 8 (the planned hub flag is only a prediction).

## 5. Results at cf = 0.3 (tuned retrieval)

| | structural | contamination (context) | answer canary | recall@10 |
|---|---|---|---|---|
| C1 no ACL | 0.97 | 0.47 | 0.52 | 0.75 |
| C2 post-retrieval dropout | 0 | 0.47 | 0.10 | 0.81 |
| C3 ACL-union pre-filter | 0 | 0.49 | 0.10 | 0.78 |
| C4 fragment-level filtering | 0 | 0 | 0 | 0.77 |

(Figures 1 and 2. With the original retrieval settings contamination in context is higher, 0.75-0.88 for C1-C3, because more is retrieved; tuning mainly cuts context contamination by retrieving less and raises recall@10 by 10-15 points. It does not reduce answer leakage in C1-C3.)

Findings:

1. **Structural leakage is eliminated by any ACL check on retrieved items** (C2, C3, C4: 0), and is near-total without one (C1: 0.97).
2. **Contamination survives item-level enforcement.** C2 and C3 still put a restricted canary into the *context of an authorized item* in about 47-49% of rows, and into the *answer* in about 10%. This is the headline result: filtering which items are retrieved does not stop restricted text arriving inside an item the user is allowed to see.
3. **Fragment-level filtering (C4) removed all detected leakage:** 0 on every measure, including the adjusted unauthorized-fact rate (Figure 4; raw 0.002 vs adjusted 0).
4. **Hubs matter for what reaches the answer, not the context** (Figure 3). Canary reach into the context is similar for hubs and non-hubs (C2/C3: 0.47-0.59), but into the answer about 4 times more often for hubs (0.27-0.29 vs 0.06-0.07). The hub group is small (126 rows) and not controlled for entity or query type, so treat this as a pattern worth testing, not an established effect.

5. **The string metrics miss paraphrased leakage in C2 and C3, but not in C4.** We judged 60 rows per config in which the string metrics found no leak and the asking user had restricted facts about the entity (`results/full_cf0.3_s42_tuned/judge_insession.json`): 30 random rows, then 30 further rows with one row per distinct entity. The judge saw only the restricted facts and the answer. It found a paraphrased restricted fact in 13 of 60 C2 rows (22%, Wilson 95% interval 13-34%), 8 of 60 C3 rows (13%, 7-24%) and 0 of 60 C4 rows (0-6%). Examples: "has about 8,900 active users" restated as a count, "focuses on data retention tooling" restated as work on "internal data retention systems". For all 21 leaks, none of the asking user's readable documents states the same value about the entity (crude keyword check). The leaks come from 8 distinct entities in C2 and 7 in C3, so the effective sample is smaller than the row count suggests. The string metrics therefore understate leakage for C2/C3, and the C4 result holds under this check at this sample size.

## 6. Limitations

- **Single corpus and seed.** One corpus at cf = 0.3 with seed 42, so there is no variance estimate across corpora, and the cf sweep is untested end to end. Intervals above are over rows, not over corpora or seeds.
- **Detection is string-based.** The canary regexes and the unauthorized-fact check catch verbatim or near-verbatim leakage only. The LLM judge for paraphrased leakage is implemented (`acl_rag/judge.py`) but was not run with the pipeline model (the cluster job was cancelled). The paraphrase estimates in finding 5 come from a 180-row sample judged by Claude instead, on the first 650-1000 characters of each answer, with no repeat judging and no second rater, and with a few borderline calls (listed in the results file), so they are indicative only. Reported string-metric rates are a lower bound on leakage.
- **Corpus artefact.** One restricted fact (F0675, team_b status "on track" for one project) is restated in a team_a document, which caused C4's only two unauthorized-fact flags under the raw metric. The corpus was not regenerated; the adjusted metric accounts for it, and a scan of all 262 restricted non-canary facts (exact substring) found no other case.
- **Low-entropy fact values** (status, pricing, region) are easy filler for the writer; invented names built from real words can evade the invented-noun check.
- **cf = 0.1 would have few contamination canaries** (30 of 300), so interpret it with care if built.
- **One public document** (`d-public-099`) is marked truncated at ingest.
- Retrieval settings were tuned on half the queries and confirmed on the other half; the original-vs-tuned comparison is on the same queries.
