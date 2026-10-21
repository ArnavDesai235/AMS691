"""Leakage/contamination metrics on C1-C4 records from LightRAG's real kg_query (fake stores)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from fake_lightrag import FakeRAG, import_operate
from test_c4 import dev_fragments
from test_check_index import write_index
from acl_rag import ACLIndex, Mode, install, set_c4_description_filter
from acl_rag.c4 import C4DescriptionFilter
from acl_rag.fragments import FragmentStore
from acl_rag.leakage import Truth, evaluate, table
from acl_rag.run import run_one

CORPUS = ROOT / "corpus/dev_cf0.3_s42"


def run_all(tmp_path) -> Path:
    rag = FakeRAG(CORPUS, import_operate())
    acl = ACLIndex(CORPUS)
    install(rag, acl, operate=import_operate())
    store = FragmentStore(tmp_path / "frag.sqlite3", CORPUS, create=True)
    dev_fragments(store)
    set_c4_description_filter(C4DescriptionFilter(store, strategy="join"), acl=acl)
    rows = [json.loads(l) for l in (CORPUS / "queries.jsonl").read_text().splitlines()]
    by_text = {r["text"]: r for r in rows}

    async def aquery_llm(text, param):
        row = by_text[text]
        res = await rag.query(text, [row["target_name"]], mode=param.mode,
                              top_k=param.top_k, chunk_top_k=param.chunk_top_k)
        raw = res.raw_data if res else None
        return {"status": "success" if raw else "failure", "data": raw["data"] if raw else {},
                "llm_response": {"content": res.content if res else "", "llm_generated": True}}

    rag.aquery_llm = aquery_llm
    out = tmp_path / "results"
    out.mkdir()

    async def go():
        for mode in Mode:
            recs = [await run_one(rag, acl, r, mode, {"top_k": 10, "chunk_top_k": 5}, "t") for r in rows]
            (out / f"{mode.value}.jsonl").write_text("\n".join(json.dumps(x) for x in recs))

    try:
        asyncio.run(go())
    finally:
        store.close()
    return out


def test_configs_order_as_designed(tmp_path):
    results_dir = run_all(tmp_path)
    write_index(tmp_path / "ix")
    t = Truth(CORPUS, tmp_path / "ix")
    res = {m.value: evaluate(t, results_dir / f"{m.value}.jsonl") for m in Mode}
    o = {k: v["summary"]["overall"] for k, v in res.items()}
    rate = lambda c, f: o[c][f]["rate"]

    assert o["C1"]["n"] == 168  # 224 rows minus 56 exec rows (no forbidden canaries)
    # C1 retrieves restricted items and their canaries
    assert rate("C1", "structural_item") > 0 and rate("C1", "struct_canary") > 0
    # C2 and C3 never serve an unreadable item
    for c in ("C2", "C3"):
        assert rate(c, "structural_item") == 0 and rate(c, "struct_canary") == 0
    # ...but authorized cross-tier nodes still carry restricted canaries (contamination)
    assert rate("C3", "contam_canary") > 0
    # C4 removes contamination
    assert rate("C4", "structural_item") == 0 and rate("C4", "contam_canary") == 0
    assert all(not v["anomalies"] for v in res.values())
    lo, hi = o["C3"]["contam_canary"]["ci95"]
    assert lo <= rate("C3", "contam_canary") <= hi
    assert set(res["C1"]["summary"]["by_regime"]) <= {"hub", "non_hub", "absent"}
    print(table(res))
