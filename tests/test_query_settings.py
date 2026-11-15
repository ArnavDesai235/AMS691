"""query block: token budgets, hl_from_ll, and the tuned full config."""
import asyncio
from pathlib import Path

import pytest

from acl_rag import run
from acl_rag.rag_factory import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_token_budgets_passed_only_when_set():
    p = run._query_param({"top_k": 10, "max_entity_tokens": 3000, "max_relation_tokens": 3000})
    assert (p.top_k, p.max_entity_tokens, p.max_relation_tokens) == (10, 3000, 3000)
    default = run._query_param({})
    assert (default.top_k, default.max_entity_tokens, default.max_relation_tokens) == (40, 6000, 8000)


class FakeRag:
    llm_response_cache = object()

    def _build_global_config(self):
        return {}


@pytest.mark.parametrize("extracted, expected", [
    ((["What do we know"], ["Tanla Terlusvi"]), (["Tanla Terlusvi"], ["Tanla Terlusvi"])),
    ((["overview"], []), (["overview"], [])),     # no low-level keywords: keep what LightRAG gave
])
def test_hl_from_ll(monkeypatch, extracted, expected):
    from lightrag import operate
    seen = {}

    async def fake(text, param, global_config, hashing_kv):
        seen["args"] = (text, hashing_kv)
        return extracted

    monkeypatch.setattr(operate, "get_keywords_from_query", fake)
    rag = FakeRag()
    param = run._query_param({})
    asyncio.run(run._hl_from_ll(rag, "What do we know about Tanla Terlusvi?", param))
    assert (param.hl_keywords, param.ll_keywords) == expected
    assert seen["args"] == ("What do we know about Tanla Terlusvi?", rag.llm_response_cache)


def test_tuned_config_differs_only_in_query_and_results():
    base = load_config(ROOT / "configs" / "rag_full.yaml")
    tuned = load_config(ROOT / "configs" / "rag_full_tuned.yaml")
    assert tuned["results_dir"] != base["results_dir"]
    assert tuned["working_dir"] == base["working_dir"] and tuned["corpus"] == base["corpus"]
    for key in set(base) | set(tuned):
        if key not in ("query", "results_dir"):
            assert tuned.get(key) == base.get(key), key
    q = tuned["query"]
    assert (q["top_k"], q["chunk_top_k"], q["max_entity_tokens"], q["max_relation_tokens"], q["hl_from_ll"]) \
        == (10, 20, 3000, 3000, True)
    changed = {"top_k", "max_entity_tokens", "max_relation_tokens", "hl_from_ll"}
    assert {k: v for k, v in q.items() if k not in changed} == \
        {k: v for k, v in base["query"].items() if k not in changed}
