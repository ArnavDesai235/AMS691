# ACL Enforcement C1–C3, Cache Keying, Runner & Metrics (Design Notes)

Owner: Bala · Status: C1/C2/C3 + cache keying implemented and tested against LightRAG 1.5.7's real `kg_query`; not yet run on a real index (needs an LLM + embedding key) · Code: `acl_rag/`

## 1. One switch, one index

All configurations query the **same** LightRAG index. `acting_as(acl, user, "C2")` sets the user and configuration for one query (a context variable); patches installed once by `install(rag, acl)` read it. Outside `acting_as` nothing changes, so ingest is stock LightRAG.

| Config | What happens | Where it hooks (1.5.7) |
|---|---|---|
| C1 | stock retrieval | — |
| C2 | retrieve as usual (vector search + one-hop expansion), then drop every entity, relation and chunk with u ∉ ACL(v) before the context is built | `operate._perform_kg_search` (entities, relations, vector chunks), `operate._merge_all_chunks` (entity/relation-derived chunks) |
| C3 | same check *inside* retrieval, so unauthorized items are never candidates | entity / relation / chunk vector stores (oversample → filter → top_k), graph reads used by one-hop expansion (`get_nodes_batch`, `get_nodes_edges_batch`, `get_edges_batch`), chunk reads (`get_by_ids`, `get_vectors_by_ids`); each node/edge `source_id` is cut to readable chunks |
| C4 | C3 + a description filter registered with `set_c4_description_filter(fn)` (C4 owner); refuses to run without one | called on every node/edge C3 lets through |

C3 does not touch a node's merged `description`; contamination is expected to remain (C4's job).

## 2. ACL(v)

`ACL(chunk)` = `acl_users` of the chunk's `full_doc_id` in `manifest.jsonl`. `ACL(node or edge)` = union over the chunk ids in its `source_id` (split on `<SEP>`). Computed at query time from LightRAG's stores and memoised (`acl_rag/acl.py`). Fails closed: a chunk whose doc is missing from the manifest has an empty ACL. Week 3's stored union must give the same sets.

## 3. Query cache

LightRAG 1.5.7's answer-cache key has no user or configuration in it, so a stock cache would serve an exec's answer to a public user, and a C1 answer under C2. The `query` cache key is salted with `C1` (C1 context is the same for everyone) or `<config>|<access set>` (`user_to_access_set` from `oracle_splits.json`). The keyword cache depends only on the query text and stays shared, so all users of a query get the same keywords.

## 4. Commands

```bash
export OPENAI_API_KEY=...        # backend in configs/rag.yaml (openai | ollama)
python -m acl_rag.ingest  --rag-config configs/rag.yaml            # index + chunk_map.json + ingest_report.json
python -m acl_rag.run     --rag-config configs/rag.yaml --config C1 # results/<corpus>/C1.jsonl
python -m acl_rag.run     --rag-config configs/rag.yaml --config C2
python -m acl_rag.run     --rag-config configs/rag.yaml --config C3
python -m acl_rag.metrics --rag-config configs/rag.yaml            # recall@k, answer quality, latency
python tests/test_acl_enforcement.py                               # or pytest
```

`ingest_report.json` answers "do merges happen on our names": per entity, whether a node exists under the exact canonical name, its source-chunk count and realized regime (≥ 8 vs < 8), plus hub-prediction mismatches.

Each result row stores the retrieved entities and relations (served description and source_id), the chunks in rank order with doc ids, the answer and latency, which is what leakage/contamination scoring needs. Use `--concurrency 1 --no-llm-cache` for latency numbers.

## 5. Metrics (`acl_rag/metrics.py`)

- **recall@k** (doc level): relevant docs = docs the user may read that state one of the row's `authorized_fact_ids`; retrieved = doc ids of context chunks in rank order. Rows with gold "not available" are excluded.
- **Answer quality**: fact recall (authorized fact values present in the answer), forbidden canary in the answer, unauthorized fact value in the answer, abstention; `correct` = all authorized facts and no leak (facts rows), or abstained and no leak ("not available" rows).
- **Latency**: mean / p50 / p95 seconds per query.

## 6. Tests (`tests/test_acl_enforcement.py`)

Runs all 224 dev rows through LightRAG 1.5.7's real `kg_query`, with in-memory stores built from the corpus (one chunk per doc, nodes merged by canonical name with `<SEP>` descriptions, edges from entity-valued facts). ACLs are re-derived in the test independently of `acl_rag`. Checks: C1 leaks (the check has teeth); C2 and C3 return no unauthorized entity, relation or chunk; no structural canary on type-b probes; C3 still contaminated on type-a probes; positive controls keep their target canary; same-access-set users get identical retrieval; C3 fills top_k where C2 loses slots; no scope = stock; the answer cache never crosses access sets; C4 refuses without a filter; unknown docs fail closed.

## 7. Notes for the team

- **Type-b neighbours:** `neighbor_canary_ids` on type-b rows can include *contamination* canaries (a cross-tier neighbour such as Satisen Labs is authorized, but its merged description carries the restricted fact). So under C2/C3, type-b `target_canary_ids` and structural neighbour canaries go to 0, but contamination neighbour canaries can remain. Report the two kinds separately.
- C3 oversampling is capped by `acl.max_candidates` (4096); with LightRAG's cosine threshold a store can still return fewer than top_k authorized items.
- The C3 node degree used for ranking still counts unauthorized edges (ranking only; no content is served).
