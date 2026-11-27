# AMS 691 Group 4: Authorization-Aware LightRAG. Progress report, PR #1 to #28

As of 2026-11-27. Repo: ArnavDesai235/AMS691 (28 PRs: 27 merged, 1 closed unmerged (#11, obsolete), none open). Single corpus (`full_cf0.3_s42`), single seed (42), cf = 0.3.

## 1. Where the project stands

| Proposal item | Status |
|---|---|
| Synthetic multi-tenant corpus, ACLs, cross-tier entities, canaries, queries, gold | Done (300 docs, 300 canaries, 1,120 query rows) |
| ACL tagging at chunking, union propagation | Done. ACL(c) = doc ACL; ACL(v) = union over `source_id`; stored form derived offline and verified against query-time (PR #27) |
| C1 / C2 / C3 / C4 configs | Done (Bala: C2/C3, Sai: C4) |
| Canary detector, leakage and contamination metrics | Done (structural, contamination, answer, unauthorized-fact raw + adjusted) |
| recall@k, answer quality, latency | Done |
| LLM judge for paraphrase leakage | Code done and tested; NOT run with the pipeline model (no GPU). 180-row in-session sample instead |
| Multiple seeds | Not done (needs GPU: 1.5-3 GPU-days per seed) |
| cf sweep 0.1 / 0.3 / 0.5 | Not done (only cf = 0.3 built and evaluated) |
| Plots | Four figures done (PR #24) |
| Report: corpus and method (Arnav) | Draft merged (PR #26) |
| Report: enforcement (Bala), results (Sai); Zoom recording (Sai) | Not verified by me |

## 2. Headline results (cf = 0.3, tuned retrieval, 840 rows from non-exec users)

| | structural | contamination (context) | answer canary | recall@10 |
|---|---|---|---|---|
| C1 no ACL | 0.97 | 0.47 | 0.52 | 0.75 |
| C2 post-retrieval dropout | 0 | 0.47 | 0.10 | 0.81 |
| C3 ACL-union pre-filter | 0 | 0.49 | 0.10 | 0.78 |
| C4 fragment-level filtering | 0 | 0 | 0 | 0.77 |

- Item-level enforcement (C2, C3) removes structural leakage but not contamination: ~47-49% of rows still carry a restricted canary inside an authorized item's description, ~10% reach the answer.
- C4 has zero detected leakage on every measure, including the adjusted unauthorized-fact rate.
- Original (untuned) retrieval had higher context contamination (0.75-0.88 for C1-C3) because more was retrieved. Tuning raised recall@10 by 10-15 points and did not reduce answer leakage in C1-C3.
- Hubs (>= 8 source chunks) reach the answer ~4x more often than non-hubs (C2/C3: 0.27-0.29 vs 0.06-0.07). Hub n = 126 rows, uncontrolled: a pattern, not an established effect.
- Paraphrase check (in-session judge, 60 string-negative rows per config): paraphrased restricted facts in 13/60 C2 (22%, 13-34%), 8/60 C3 (13%, 7-24%), 0/60 C4 (0-6%). All 21 leaks state values absent from the user's readable docs. Strict/lenient range: C2 12-15, C3 7-10, C4 0. One judge, truncated answers, no second rater: indicative only.

## 3. What was done, PR by PR

Authors: Arnav (#1, 3-10, 14, 16-19, 21-28), Bala (#2, 11, 12, 13), Sai (#15, 20).

**Foundation (Oct 10-21)**
- **#1 Corpus generator + dev corpus (Arnav).** Facts, plan and verbalize pipeline; access model (public, team_a, team_b, exec; one ACL per document); cross-tier, restricted-only and public-only entities; canaries planted once in restricted docs with tolerant regexes; query/gold generation; independent validator. Dev corpus: 50 docs, 27 canaries.
- **#2 C1/C2/C3 enforcement (Bala).** ACL enforcement inside LightRAG's retrieval (C2 post-retrieval dropout, C3 ACL-union pre-filtering with oversample-then-filter), access-set-scoped cache keys, runner, metrics.
- **#3 Ollama backend (Arnav).** `qwen3:32b-q8_0` + `bge-m3` on the cluster H100; SLURM job script; no API cost.
- **#4 C4 integration (Arnav, for Sai).** Authorized-fragment descriptions (fragment store + per-user description assembly) wired into the shared runner.
- **#5, #6, #7, #9 Ingest hardening (Arnav).** Output-token cap and aligned timeouts; retry failed docs without duplicate rows; `check_index` verifies the index against the corpus plan; detect and retry runaway truncated extractions.
- **#8 Leakage metrics (Arnav).** Structural and contamination detectors at item and answer level, per tier, per type, per eval role and per merge regime, with bootstrap 95% intervals.

**Full corpus and first results (Oct 31 - Nov 12)**
- **#10 Full corpus and ingest (Arnav).** 300 docs generated with qwen3 (228 passed first attempt, 63 second, 8 third, 1 fifth; 233 needed deterministic repair); ingested into LightRAG (13 h 25 min on 2x RTX 8000). One public doc, `d-public-099`, still marked truncated.
- **#12 Packed results (Bala).** Per-row results gzipped with a MANIFEST of hashes; scorers read either form.
- **#13 Answer cap (Bala).** Answer length capped separately from extraction (`query.answer_num_predict`).
- **#14 wordfreq required (Arnav).** The real-name check can no longer silently disable itself.
- **#15, #16 C4 audit (Sai, Arnav).** Offline C4 index audit; guards for zero cases and a stale fragment store.
- **#17 Full C1-C4 results (Arnav).** Job 89670 (2 d 9 h): metrics, leakage, packed rows. Contamination: C1-C3 0.75-0.88, C4 0.

**Tuning and re-run (Nov 15-23)**
- **#18 Retrieval tuning (Arnav).** Sweep tool (tuned on half the queries, confirmed on the other half) plus `rag_full_tuned.yaml`: top_k 10, entity/relation tokens 3000, chunk_top_k 20, `hl_from_ll` on. +10 to +15 recall@10.
- **#19 Tuned results (Arnav).** Job 92389 (1 d 15 h): metrics, leakage, original-vs-tuned comparison.
- **#20 C4 validation (Sai).** Documents the full-run C4 validation and investigates scoring flags. Open items from it: the C3 unauthorized-fact rate and `join` vs `auto` description strategy are not covered in the doc.

**Scoring fixes, report, and Week-3 items (Nov 26-27)**
- **#21 Restricted-value guard + adjusted metric (Arnav).** Root cause of C4's only scoring flags: one restricted fact (F0675, team_b status) restated in a team_a doc. Added a generation-time check against restating another tier's value, and `answer_unauthorized_fact_rate_adj` (ignores values the user's own readable docs state). Raw rate kept.
- **#22 Rescore (Arnav).** Full and tuned runs rescored; only the new `_adj` fields changed. Raw -> adjusted for C4 tuned: 0.002 -> 0.
- **#23 DESIGN_corpus update (Arnav).** Status, results table, hub finding, limitations.
- **#24 Figures (Arnav).** `scripts/make_plots.py`: leakage by config, original vs tuned, hub vs non-hub, raw vs adjusted. Palette validated and renders checked.
- **#25 LLM judge (Arnav).** `acl_rag/judge.py` + sbatch `judge` step + 5 tests. Cluster job 99240 was queued for a week, then cancelled by choice. Not run with qwen3.
- **#26 Corpus-and-method report sections (Arnav).** Access model, ACL union, corpus, metrics, results, limitations.
- **#27 Stored ACLs + `origin_chunks` (Arnav).** `acl_rag/stored_acl.py`: per-chunk, per-node and per-edge ACLs derived offline from the index. 0 disagreements with the query-time `ACLIndex` over 3,719 nodes and 6,053 edges; all 300 canaries in exactly one chunk of their origin doc and in none elsewhere. 3 tests.
- **#28 Paraphrase judge, 60 rows per config (Arnav).** In-session judge results, strict/lenient sensitivity, docs updated to say the pipeline judge was not run.
- **#11 (Bala), closed unmerged.** `.gitignore` tweak, obsolete: main no longer ignores `results/`.

## 4. Known limitations (state these in the final report)

1. One corpus, one seed, one cf (0.3). No variance across corpora or seeds; intervals are over rows only.
2. Detection is string-based. The pipeline LLM judge was not run; paraphrase leakage is estimated from a 180-row sample judged by Claude (one judge, truncated answers, no second rater).
3. F0675 restated in a team_a doc (corpus artefact); corpus not regenerated. The adjusted metric accounts for it. The new guard only applies to regenerated docs.
4. Low-entropy fact values (status, pricing, region) are easy filler; invented names built from real words can evade the check.
5. LightRAG does not store the ACL union at merge time. Our stored form is an offline equivalent, verified against the query-time path.
6. Hub finding rests on 126 rows, uncontrolled for entity or query type.
7. Retrieval tuned on half the queries, confirmed on the other half; original-vs-tuned compares the same queries.

## 5. What teammates should check

### Bala (C2 / C3, enforcement)
1. **Your results rows.** Confirm the C2 and C3 rows in the results table and in `reports/REPORT_corpus_and_method.md` (section 5) match what you expect from your runs: C2 0 / 0.47 / 0.10 / 0.81, C3 0 / 0.49 / 0.10 / 0.78.
2. **The union ACL.** PR #27 reimplements ACL(v) = union over `source_id` chunks offline and checks it against your `ACLIndex`. Read `acl_rag/stored_acl.py` and confirm it agrees with your intended semantics, in particular fail-closed behavior (unknown doc gives an empty ACL) and the edge-key normalization.
3. **The paraphrase finding.** C2 22% and C3 13% of string-negative rows leak a paraphrased restricted fact. Check that the description in the report of how C2/C3 behave is consistent with it, and spot-check a few rows in `results/full_cf0.3_s42_tuned/judge_insession.json`.
4. **Your report section (enforcement).** Make sure it is consistent with the corpus-and-method section: same definitions of structural vs contamination, same denominators (840 non-exec rows).

### Sai (C4, results, recording)
1. **The C4 claim.** C4 shows zero detected leakage on every measure and 0/60 paraphrase leaks (0-6% interval). Confirm this matches your own validation, and note the interval is an upper bound, not proof of zero.
2. **PR #20 gaps.** Your doc does not yet cover the C3 unauthorized-fact rate or `join` vs `auto` description strategy. Add them or say they are out of scope.
3. **Latency.** C4 had the highest p95 before tuning (178 s vs 113-129 s) and 101 s tuned. Check the latency figure and the explanation you want in the results section.
4. **Plots.** The four figures are in `reports/figures/` (`python scripts/make_plots.py`). Check they are what you want for the results section, and say if you need a different layout or a latency-vs-leakage plot.
5. **Multiple seeds.** The proposal assigns the multi-seed experiment runner to you. It has not been run: only seed 42 exists, so the final report must say single seed unless a GPU becomes available.
6. **Assembly.** Final report assembly and the Zoom recording are yours per the proposal; I have not seen either. Check the recording covers the contamination finding as the headline.

### Both
- **Read PR #26's report text** (`reports/REPORT_corpus_and_method.md`) and flag any statement about C2/C3/C4 that is wrong.
- **Optional second rater.** Rate about 30 rows of the paraphrase sample independently (`results/full_cf0.3_s42_tuned/judge_insession.json` lists the rows; the judge prompt is in `acl_rag/judge.py`) so the paraphrase numbers have a second opinion.
- **Agree the limitations wording** (single seed, single cf, judge not run on the pipeline model) before submission.
