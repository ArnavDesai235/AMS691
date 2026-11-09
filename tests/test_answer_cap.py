"""query.answer_num_predict: an output cap for answer generation only.

    python -m pytest tests/test_answer_cap.py
    python tests/test_answer_cap.py                     # no pytest needed
    LIGHTRAG_SRC=/path/to/LightRAG python tests/...     # from a source checkout

Checks that (1) the cap reaches only LightRAG 1.5.7's "query" role, while the
"keyword" and "extract" roles (keywords, entity/relation and C4 summaries) keep
backend.ollama_num_predict; (2) the role kwargs keep every other backend option;
(3) the answer-cache key changes with the cap and is unchanged without one; and
(4) both run configs set the cap without touching the extraction cap.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import yaml  # noqa: E402

from acl_rag.enforce import Mode, Scope, _salt  # noqa: E402
from acl_rag.rag_factory import answer_num_predict, ollama_llm_kwargs, query_role_kwargs  # noqa: E402

BACKEND = {"llm": "ollama", "llm_model": "m", "ollama_host": "http://localhost:11434",
           "ollama_num_ctx": 32768, "ollama_think": False, "ollama_temperature": 0,
           "ollama_seed": 42, "ollama_timeout": 1200, "ollama_num_predict": 6144}


def _stub(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__getattr__ = lambda attr: type(attr, (), {"__init__": lambda self, *a, **k: None})
    sys.modules[name] = m


def _import_lightrag_class():
    """LightRAG from the installed package, else from LIGHTRAG_SRC with stubs for
    modules that building the class (without initialising storages) does not use."""
    cls = None
    try:
        cls = importlib.import_module("lightrag.lightrag").LightRAG
    except ModuleNotFoundError:
        src = os.environ.get("LIGHTRAG_SRC")
        if not src:
            return None
        sys.path.insert(0, src)
        for _ in range(40):
            try:
                cls = importlib.import_module("lightrag.lightrag").LightRAG
                break
            except ModuleNotFoundError as e:
                _stub(e.name)
                for k in [k for k in sys.modules if k.startswith("lightrag")]:
                    del sys.modules[k]
    if cls is None:
        return None
    # The default vector store is only constructed here, never used; a source
    # checkout without nano_vectordb installed needs a stand-in to build the class.
    try:
        importlib.import_module("nano_vectordb")
    except ModuleNotFoundError:
        _stub("nano_vectordb")
    return cls


# ---------------------------------------------------------------- config helpers
def test_unset_means_no_cap():
    assert answer_num_predict({"query": {"mode": "mix"}}) is None
    assert answer_num_predict({}) is None


def test_too_small_cap_is_refused():
    try:
        answer_num_predict({"query": {"answer_num_predict": 100}})
    except ValueError:
        return
    raise AssertionError("a cap below 256 tokens must be refused")


def test_query_role_kwargs_change_only_the_cap():
    base = ollama_llm_kwargs(BACKEND)
    q = query_role_kwargs(BACKEND, 3072)
    assert q["options"]["num_predict"] == 3072
    assert base["options"]["num_predict"] == 6144, "base kwargs must not be mutated"
    assert {k: v for k, v in q["options"].items() if k != "num_predict"} == \
           {k: v for k, v in base["options"].items() if k != "num_predict"}
    assert q["host"] == base["host"] and q["timeout"] == base["timeout"]


def test_both_run_configs_cap_answers_but_not_extraction():
    for name in ("rag.yaml", "rag_full.yaml"):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
        assert answer_num_predict(cfg) == 3072, name
        assert cfg["backend"]["ollama_num_predict"] == 6144, name


# ---------------------------------------------------------------- answer cache key
def test_cache_key_unchanged_without_cap_and_split_by_cap():
    s = Scope(Mode.C3, "u_public_1", "AS0")
    old = _salt("abc", s)                                  # key before this change
    assert _salt("abc", s, "") == old
    assert _salt("abc", s, "|answer_num_predict=3072") != old
    assert _salt("abc", s, "|answer_num_predict=3072") != _salt("abc", s, "|answer_num_predict=2048")


# ---------------------------------------------------------------- real LightRAG 1.5.7 roles
def test_cap_reaches_only_the_query_role():
    LightRAG = _import_lightrag_class()
    if LightRAG is None:
        print("SKIP test_cap_reaches_only_the_query_role: lightrag not importable (set LIGHTRAG_SRC)")
        return
    import numpy as np
    from lightrag.utils import EmbeddingFunc, Tokenizer

    class WordTokenizer:
        def encode(self, s):
            return list(range(len(s.split())))

        def decode(self, toks):
            return " ".join("w" for _ in toks)

    seen: list[dict] = []

    async def llm(prompt, system_prompt=None, history_messages=None, **kw):
        seen.append(kw)
        return "ok"

    async def emb(texts):
        return np.zeros((len(texts), 8))

    rag = LightRAG(
        working_dir=tempfile.mkdtemp(), llm_model_func=llm, llm_model_name="m",
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=100, func=emb),
        tokenizer=Tokenizer("words", WordTokenizer()),
        llm_model_kwargs=ollama_llm_kwargs(BACKEND),
        role_llm_configs={"query": {"kwargs": query_role_kwargs(BACKEND, 3072)}},
    )
    funcs = rag._build_global_config()["role_llm_funcs"]

    async def call(role):
        seen.clear()
        await funcs[role]("hello")
        return seen[0]["options"]["num_predict"]

    got = {role: asyncio.run(call(role)) for role in ("query", "keyword", "extract")}
    assert got == {"query": 3072, "keyword": 6144, "extract": 6144}, got


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
