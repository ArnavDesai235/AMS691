"""Config loading and LightRAG construction shared by ingest / run / metrics."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    corpus = Path(cfg["corpus"])
    name = corpus.name
    cfg["corpus"] = corpus
    cfg["working_dir"] = Path(cfg["working_dir"].format(corpus_name=name))
    cfg["results_dir"] = Path(cfg["results_dir"].format(corpus_name=name))
    return cfg


def _llm_and_embedding(b: dict[str, Any]):
    from lightrag.utils import EmbeddingFunc  # noqa: PLC0415

    if b["llm"] == "openai":
        from lightrag.llm.openai import openai_complete_if_cache  # noqa: PLC0415

        async def llm(prompt, system_prompt=None, history_messages=None, **kw):
            return await openai_complete_if_cache(
                b["llm_model"], prompt, system_prompt=system_prompt,
                history_messages=history_messages or [], **kw)
    elif b["llm"] == "ollama":
        from lightrag.llm.ollama import ollama_model_complete  # noqa: PLC0415
        llm = ollama_model_complete
    else:
        raise ValueError(f"backend.llm {b['llm']!r} not supported (openai | ollama)")

    if b["embedding"] == "openai":
        from lightrag.llm.openai import openai_embed  # noqa: PLC0415
        emb = EmbeddingFunc(embedding_dim=b["embedding_dim"], max_token_size=8192,
                            func=partial(openai_embed.func, model=b["embedding_model"]))
    elif b["embedding"] == "ollama":
        from lightrag.llm.ollama import ollama_embed  # noqa: PLC0415
        emb = EmbeddingFunc(embedding_dim=b["embedding_dim"], max_token_size=8192,
                            func=partial(ollama_embed.func, embed_model=b["embedding_model"],
                                         host=b["ollama_host"]))
    else:
        raise ValueError(f"backend.embedding {b['embedding']!r} not supported (openai | ollama)")
    return llm, emb


async def build_rag(cfg: dict[str, Any]):
    """Create and initialise the LightRAG instance for this corpus (stock, no ACL yet)."""
    from lightrag import LightRAG  # noqa: PLC0415

    b = cfg["backend"]
    llm, emb = _llm_and_embedding(b)
    lr = cfg.get("lightrag", {})
    cfg["working_dir"].mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = dict(
        working_dir=str(cfg["working_dir"]),
        llm_model_func=llm,
        llm_model_name=b["llm_model"],
        llm_model_max_async=b.get("llm_max_async", 4),
        embedding_func=emb,
    )
    for k in ("chunk_token_size", "chunk_overlap_token_size", "tiktoken_model_name",
              "force_llm_summary_on_merge"):
        if k in lr:
            kwargs[k] = lr[k]
    if b["llm"] == "ollama":
        kwargs["llm_model_kwargs"] = {"host": b["ollama_host"]}
    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    return rag


async def build_acl_rag(cfg: dict[str, Any]):
    """LightRAG + ACL index with C2/C3 enforcement installed."""
    from .acl import ACLIndex  # noqa: PLC0415
    from .enforce import install  # noqa: PLC0415

    rag = await build_rag(cfg)
    acl = ACLIndex(cfg["corpus"])
    install(rag, acl, max_candidates=cfg.get("acl", {}).get("max_candidates", 4096))
    return rag, acl
