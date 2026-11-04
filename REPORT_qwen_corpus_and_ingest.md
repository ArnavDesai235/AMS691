# Full corpus with qwen3 and full-corpus ingestion — report

Arnav Desai · Nov 1–4, 2026 · branch `qwen-repair`

## Summary

- **Full corpus generated with `qwen3:32b-q8_0` on the cluster** (`corpus/full_cf0.3_s42`):
  300 docs, 300 canaries (90 contamination, 210 structural), 1,120 query rows, 0 validation errors.
  Getting there needed deterministic repairs of qwen3 drafts: first-try pass rate went from
  14/50 to 34/50 on the dev plan, and every remaining doc now passes the unchanged hard checks.
- **Full corpus ingested into LightRAG** (`rag_storage/full_cf0.3_s42`, not in git, 122 MB):
  300/300 docs processed, 300/300 planned entities found by exact name, realized cross-tier
  fraction 0.293, all 26 planned hubs realized. One public doc (`d-public-099`) is still
  marked truncated (see open issues).
- **Ingestion pipeline reviewed and debugged.** Four fixes: `check_index` hub planning,
  leakage scoring of canary-named nodes, sampled retries for docs that fail deterministically,
  and a full-corpus Ollama timeout suited to the RTX 8000s. 62 tests pass.
- **New finding for the analysis:** on the full index, 29 *structural* canaries are visible
  outside their ACL through merged generic nodes ("Escrow Reference", "Security Team",
  "Vault Key", ...). On the dev index this was 0. C3 will therefore show structural-canary
  contamination, not only cross-tier contamination.

## 1. Corpus generation with qwen3

The dev corpus (`dev_cf0.3_s42`) was written by Claude Sonnet. The full corpus is generated
on the cluster with the same model used for retrieval (`configs/full.yaml`, provider `ollama`).

### What failed and how it was fixed

The hard checks in `corpus_gen/verbalize.py` were left unchanged. Instead, drafts are repaired
deterministically before checking (`corpus_gen/repair.py`), only ever moving text toward the
planned ground truth:

| qwen3 habit | Repair |
|---|---|
| Shortened names ("Vadesis" for "Vadesis Gummenbel") | expand to the canonical name (names are unique coined tokens) |
| Canary code repeated in summaries / action items | keep the one occurrence anchored next to its entity, scrub the rest |
| Canary code not in a sentence with its entity | scrub it and insert the planned canary sentence after the entity's first paragraph |
| Fact value paraphrased ("a 480-seat capacity") | insert the planned fact sentence verbatim |
| Same habits in gold answers | `repair_gold`: expand names, append missing authorized fact sentences |
| Docs a few words short | exact word counts in the fix prompt; fresh salted samples after attempt 4; whole-word bounds |
| Invented people ("Sarah M. Delarue") | new check: common given name + unplanned rare surname is rejected |

Also: `nvidia-smi | head -1` under `pipefail` killed 2-GPU jobs with SIGPIPE (fixed in the
sbatch script).

### Run history (reports in `reports/corpus_gen/`)

| Run | Docs ok | First try | Notes |
|---|---|---|---|
| dev smoke 1 (87295) | 35/50 | 14 | no repair; 12/27 canaries lost |
| dev smoke 2 (87611) | 46/50 | 24 | names + canary dedupe |
| dev smoke 3 (88065) | 50/50 | 34 | + fact insertion; gold answers then failed |
| dev smoke 4 (88073) | 50/50 | 34 | + gold repair; end to end, 224 rows, 0 errors |
| full 1 (88076) | 282/300 | 180 | 15 unanchored canaries, 3 short |
| full 2 (88116) | 298/300 | 227 | + re-anchoring; 2 short email threads |
| full 3 (88123) | 300/300 | 230 | passed validation; 3 invented people found afterwards |
| full final (88148) | 300/300 | 228 | invented-name check; regenerated 2 docs |

The 50-doc qwen3 dev corpus is committed as `corpus/devqwen_cf0.3_s42` for comparison with the
Sonnet dev corpus.

### Final full corpus (`corpus/full_cf0.3_s42`)

| | |
|---|---|
| Docs | 300 (120 public, 60 team_a, 60 team_b, 60 exec); 170,179 words |
| Attempts | 1: 228, 2: 63, 3: 8, 5: 1; 233 docs repaired |
| Canaries | 300/300 placed (90 contamination, 210 structural) |
| Cross-tier fraction | 0.30 (45 entities) |
| Queries | 140 × 8 users = 1,120 rows (a 480, b 320, c 320) |
| Gold answers | 224 without an LLM call, 896 on the first attempt; 0 contain a forbidden canary |
| Validation | 0 errors; one harmless warning ("Resubmitting") |

Prose is plainer than Sonnet's and repaired docs repeat full names often; the ground truth is as
strict as for the dev corpus.

## 2. Ingestion pipeline review

Checked `acl_rag/ingest.py`, `rag_factory.py`, `fragments.py` (C4 capture), `check_index.py`,
`leakage.py`, the sbatch script, the dev index from Oct 31 and the logs of all earlier ingest jobs.

**Behaviour verified as correct**
- Resume: LightRAG 1.5.7 auto-resumes PENDING/PROCESSING docs after an interrupted job;
  `prepare_resume` clears FAILED docs and `dup-*` rows before re-inserting.
- Dev index: 50/50 docs, 57 chunks, 0 chunks with a wrong parent, 55/55 entities, 2,188 C4
  fragments. Earlier failed jobs were caused by truncation (86046) and Ollama read timeouts
  (86328), both handled by `--rounds`.

**Bugs fixed**
1. `check_index` took planned hubs from `facts.json`'s `hub` flag, which is only enforced for
   cross-tier entities. It reported 6 "planned hubs not realized" on the dev index where the
   ingest report (using `plan.json`'s `predicted_regime`) had 1. Both now use `plan.json`.
2. `leakage.py` searched only entity descriptions and relation description/keywords. Extraction
   turns most canary codes into nodes (dev 25/27, full 282/300), so a node named after a canary
   was not counted. Names and relation endpoints are now scored. Re-scoring the dev results gave
   identical rates (C1 was already at 100% structural leakage; C2–C4 never serve such nodes),
   but the metric is now correct for the full corpus.
3. Deterministic retry failures: extraction runs at temperature 0 / seed 42, so a chunk that sends
   qwen3 into a repetition loop does so on every retry. `ingest --retry-temperature T` samples
   every round that only retries docs at T with a new seed per round (recorded in the report).
   Applied by a wrapper around the LLM function, because LightRAG deep-copies `llm_model_kwargs`
   in some paths.
4. On 2× RTX 8000 a 6,144-token runaway outlasts a 600 s timeout and fails as a ReadTimeout
   instead of a retryable truncation. `configs/rag_full.yaml` uses `ollama_timeout: 1200`.
   `num_predict` stays 6,144: 15 successful extractions in the full run were 4.1k–6.0k tokens.

Also added (untested on a real job): `OLLAMA_NUM_PARALLEL` / `OLLAMA_KV_CACHE_TYPE` overrides in
the sbatch script so a small rerun can use one 48 GB GPU.

## 3. Full-corpus ingest

The H100 was fully booked (other users' multi-day jobs), so ingestion ran on quadro2/quadro1.

| Job | Node | Time | Result |
|---|---|---|---|
| 88153 | quadro2 (2× RTX 8000) | 13 h 25 m | round 1: 300 inserted; round 2: 47 retried; round 3: 17 retried → 288 ok, 11 ReadTimeout, 1 truncated |
| 88423 | quadro1 (2× RTX 8000) | 52 m | `--retry-temperature 0.3`: 12 → 2 → 1 → `d-public-099` still truncated |

Throughput ~32 docs/hour. In the first pass 47 of 300 docs (16%) hit a runaway generation
(truncated at the output cap or timed out); sampled retries resolved all but one.

### Index check (`reports/index/full_cf0.3_s42/index_check.json`)

| | Full | Dev |
|---|---|---|
| Docs processed | 300/300 (1 truncated) | 50/50 |
| Graph | 3,719 nodes, 6,053 edges, 302 chunks | 769 nodes, 932 edges, 57 chunks |
| Planned entities found | 300/300 | 55/55 |
| Cross-tier realized | 0.293 (1 planned not mixed: Drugoan Platform Group) | 0.296 |
| Hubs (≥ 8 chunks) | 99 realized, 26 planned, 0 planned not realized | 10 realized, 5 planned, 0 not realized |
| Structural canaries in index / exposed via merge | 204/210 / **29** | 19/19 / 0 |
| Contamination canaries in index / exposed via merge | 88/90 / 34 | 8/8 / 6 |
| Canary codes as node names | 282/300 | 25/27 |
| C4 fragments | 13,935 | 2,188 |

## 4. Open issues and observations

1. **`d-public-099` remains truncated** after three sampled retries. It is public, has no
   canaries, and all six of its planned entities are in the graph; some of its relations may be
   missing. Proposed: accept the index and note it.
2. **Structural canaries leak through generic nodes.** The 29 exposed structural canaries are
   carried by nodes such as "Escrow Reference", "Security Team", "Internal Audit", "Budget",
   "Audit Team", "Vault Key". These role/concept nodes appear in docs of every tier, so their
   merged descriptions mix restricted and public sources. This is extractor behaviour at scale,
   not a pipeline bug, and it should appear in the write-up.
3. **Many unplanned hubs** ("Zastama Industries" 200 chunks, "Finance Lead" 85, ...). Planned
   hubs all formed, so the per-regime analysis still holds; the slice by realized regime will be
   dominated by generic nodes unless restricted to planned entities.
4. **Runaway answers at query time:** in the dev C1–C4 run (86916), 30 answers hit the 6,144-token
   cap (~25k characters each). They slow the run and inflate `answer_canary` text. A lower
   answer-generation cap than the extraction cap is worth considering before the full run.
5. **C1–C4 on the full corpus is large:** dev took 7.4 h on the H100 for 224 rows; the full set has
   1,120 rows (~37 h at the same rate, more on RTX 8000s). The sbatch default time limit (8 h) must
   be raised.

## 5. Dev C1–C4 results (Sonnet corpus, qwen3 index; `reports/results/dev_cf0.3_s42/`)

Leakage rates over the 168 rows with forbidden canaries (exec rows excluded):

| | structural item | struct canary | contam canary | answer canary | accuracy (all 224) |
|---|---|---|---|---|---|
| C1 | 1.000 | 1.000 | 0.857 | 0.441 | 0.375 |
| C2 | 0 | 0 | 0.893 | 0.131 | 0.460 |
| C3 | 0 | 0 | 0.881 | 0.131 | 0.429 |
| C4 | 0 | 0 | 0 | 0 | 0.759 |

## 6. Files in this branch

- `corpus_gen/repair.py`, changes in `corpus_gen/verbalize.py`, `corpus_gen/queries.py`
- `acl_rag/check_index.py`, `acl_rag/leakage.py`, `acl_rag/ingest.py`, `acl_rag/rag_factory.py`
- `configs/dev_ollama.yaml`, `configs/rag_full.yaml` (timeout), `scripts/run_ollama_job.sbatch`
- `tests/test_repair.py` and new tests in `test_check_index.py`, `test_leakage.py`, `test_ingest_resume.py`
- `corpus/full_cf0.3_s42/` (final corpus), `corpus/devqwen_cf0.3_s42/` (qwen3 dev smoke corpus)
- `reports/corpus_gen/` (every generation run), `reports/index/` (ingest and index-check reports, dev
  and full), `reports/results/dev_cf0.3_s42/` (dev C1–C4 metrics and leakage summaries),
  `reports/slurm_jobs.psv` (all jobs referenced here)

Not in git: the LightRAG index directories (`rag_storage/`), per-row C1–C4 result files
(`results/*/C?.jsonl`, 133 MB) and job logs; all remain on the cluster under `~/project/AMS691`.

## 7. Reproduce

```bash
# corpus (quadro2, 2 GPUs)
STEPS=corpus CORPUS_CONFIG=configs/full.yaml MIN_GPU_MEM_MIB=45000 \
  sbatch --nodelist=quadro2 --gres=gpu:2 --time=3-00:00:00 scripts/run_ollama_job.sbatch
# ingest + index check
STEPS="ingest check" RAG_CONFIG=configs/rag_full.yaml MIN_GPU_MEM_MIB=45000 \
  INGEST_ARGS="--rounds 3 --retry-temperature 0.3" \
  sbatch --nodelist=quadro2 --gres=gpu:2 --time=3-00:00:00 scripts/run_ollama_job.sbatch
```

## 8. Next steps

1. Accept the index with `d-public-099` noted (or one more retry at a higher temperature).
2. Decide on an answer-generation output cap, then run C1–C4 + metrics + leakage on the full
   index with a multi-day time limit.
3. Add the generic-node leakage path (open issue 2) to `DESIGN_corpus.md` / the write-up, and
   update `DESIGN_corpus.md` for the Ollama generator.
