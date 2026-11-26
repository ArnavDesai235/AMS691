# Retrieval sweep (full_cf0.3_s42)

Why `configs/rag_full_tuned.yaml` changes the query block. Tool: `acl_rag/sweep_retrieval.py`
(retrieval only, `rag.aquery_data`, same ACL hooks as `acl_rag.run`, keywords from the full run).
Raw summaries: `reports/results/sweep/`.

## Diagnosis (full C1-C4 run, job 89670)

In the C4 run, 964 of 968 missed relevant docs were linked to the target entity in the graph and
were cut during chunk selection; 4 were extraction misses. Mix mode retrieves `top_k`=40 entities
and relations, interleaves their chunks with vector chunks, keeps `chunk_top_k`=20 of the merged
list, then truncates to the 30k-token budget, of which entity/relation descriptions take 14k. The
target's 4-6 docs compete with 39 other entities' chunks. High-level keywords are mostly generic
("What do we know" in 464 of 1120 queries).

## Method

- Grid: `top_k` {10, 20, 40} x entity/relation tokens {6000/8000, 3000/3000} x `chunk_top_k`
  {20, 40} x keywords {orig, ll} (ll: low-level keywords also used as high-level).
- Full corpus split in half by `query_id` hash (all users of a query together): grid on the
  tune half (640 rows), best setting confirmed on the test half (480 rows).
- The current setting is always in the grid and must reproduce the full run's ranked docs
  (`baseline_match`).

## Results, recall@10

| | tune: current | tune: chosen | test: current | test: chosen | test gain |
|---|---|---|---|---|---|
| C1 | 0.580 | 0.741 | 0.634 | 0.773 | +13.9 |
| C2 | 0.643 | 0.788 | 0.679 | 0.830 | +15.1 |
| C3 | 0.650 | 0.768 | 0.678 | 0.788 | +11.0 |
| C4 | 0.661 | 0.765 | 0.689 | 0.788 | +9.9 |

Chosen: `top_k` 10, `max_entity_tokens` 3000, `max_relation_tokens` 3000, `chunk_top_k` 20,
`hl_from_ll`. Test half, recall@all: 0.75-0.79 -> 0.89-0.92; context goes from ~45 entities and
~17 chunks to ~17 entities and ~20 chunks.

- `top_k` 40 -> 10 is most of the gain; `hl_from_ll` adds ~4-5 points; the smaller budgets 0-1.
- `chunk_top_k` 40 leaves recall@10 unchanged and raises recall@all by ~3 points; not adopted, to
  keep the context size.
- `baseline_match` = 1.0 for every config on the test half. On the tune half C4 was 0.79: that run
  used the pre-run LLM cache, so C4 re-summarized hub items; with the full run's cache it is 1.0.
- Dev (50 docs) starts at 0.86-0.91 and gains 1.5-2.7 points; the problem is one of scale.

## Full rerun with the tuned config (job 92389)

`results/full_cf0.3_s42_tuned` vs `results/full_cf0.3_s42`, all 1120 rows per config. Every row
retrieved the docs the sweep predicted. Authorized = positive_control + qc (650 rows).

| | recall@10 | fact recall | accuracy | authorized acc. | answer canary | contam. context | p95 latency |
|---|---|---|---|---|---|---|---|
| C1 | 0.602 -> 0.754 | 0.805 -> 0.815 | 0.265 -> 0.254 | 0.457 -> 0.437 | 0.479 -> 0.519 | 0.83 -> 0.47 | 113 -> 74 s |
| C2 | 0.658 -> 0.806 | 0.829 -> 0.840 | 0.326 -> 0.332 | 0.469 -> 0.489 | 0.107 -> 0.102 | 0.88 -> 0.47 | 120 -> 95 s |
| C3 | 0.662 -> 0.776 | 0.821 -> 0.843 | 0.331 -> 0.368 | 0.469 -> 0.471 | 0.098 -> 0.095 | 0.75 -> 0.49 | 129 -> 99 s |
| C4 | 0.672 -> 0.774 | 0.825 -> 0.844 | 0.567 -> 0.612 | 0.466 -> 0.506 | 0 -> 0 | 0 -> 0 | 178 -> 101 s |

- Recall +10 to +15 points translates into only +1 to +2 points of fact recall: the answer model,
  not retrieval, now limits accuracy. Authorized accuracy moves -2 (C1) to +4 (C4).
- Latency: p95 down 20-43% in every config; total job time 1d15h vs 2d9h. C4's tail cost over C3
  is almost gone (101 s vs 99 s).
- Contamination: fewer entities halve the rows with a contaminated item in context, but C2/C3
  answer leakage stays ~10%: the canary is in the target entity's own merged description, which is
  still retrieved. Retrieval tuning does not fix contamination; C4 stays at 0 leakage.
- C1 (no ACL) leaks slightly more (0.479 -> 0.519): better recall also reaches restricted target docs.
- Same caveats as the original run: one seed, string-match scoring, same-tier users share answers.
