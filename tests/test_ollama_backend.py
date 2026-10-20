"""Ollama backend wiring: options reach LightRAG's Ollama call intact."""
from pathlib import Path

import pytest
import yaml

from acl_rag.rag_factory import _llm_and_embedding, ollama_llm_kwargs

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "rag.yaml").read_text())


def test_config_options_forwarded():
    kw = ollama_llm_kwargs(CFG["backend"])
    assert kw["options"] == {"num_ctx": 32768, "think": False, "temperature": 0.0, "seed": 42}
    assert kw["host"] == CFG["backend"]["ollama_host"]


def test_small_context_rejected():
    with pytest.raises(ValueError):
        ollama_llm_kwargs({**CFG["backend"], "ollama_num_ctx": 2048})


def test_embedding_options():
    _, emb = _llm_and_embedding(CFG["backend"])
    assert emb.embedding_dim == CFG["backend"]["embedding_dim"]
    assert emb.func.keywords["options"] == {"num_ctx": CFG["backend"]["ollama_embed_num_ctx"]}


def test_lightrag_passes_options_and_think_to_chat(monkeypatch):
    """Run LightRAG 1.5.7's own Ollama adapter against a fake client."""
    import asyncio

    import ollama
    from lightrag.llm import ollama as lr_ollama

    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen["client"] = kw

        async def chat(self, **kw):
            seen["chat"] = kw
            return {"message": {"content": "ok"}}

        async def _client_aclose(self):
            pass

    monkeypatch.setattr(ollama, "AsyncClient", FakeClient)
    monkeypatch.setattr(lr_ollama.ollama, "AsyncClient", FakeClient)
    out = asyncio.run(lr_ollama._ollama_model_if_cache(
        "qwen3:32b-q8_0", "hi", max_tokens=64, **ollama_llm_kwargs(CFG["backend"])))
    assert out == "ok"
    assert seen["chat"]["think"] is False
    assert seen["chat"]["options"] == {"num_ctx": 32768, "temperature": 0.0, "seed": 42, "num_predict": 64}
