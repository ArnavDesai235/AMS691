"""corpus_gen's Ollama provider: options, think stripping, truncation never cached."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus_gen import llm as llm_mod
from corpus_gen.config import LLMParams, load_config


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, **kw):
        self.calls.append(kw)
        return self.replies.pop(0)


def reply(text, done="stop"):
    return {"message": {"content": text}, "done_reason": done, "prompt_eval_count": 10, "eval_count": 5}


def make(tmp_path, monkeypatch, replies):
    import ollama
    fake = FakeClient(replies)
    monkeypatch.setattr(ollama, "Client", lambda **kw: fake)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    p = LLMParams(provider="ollama", model="qwen3:32b-q8_0", max_retries=3, seed=42, cache_dir="c")
    return llm_mod.LLM(p, root=tmp_path), fake


def test_options_and_think_stripping(tmp_path, monkeypatch):
    llm, fake = make(tmp_path, monkeypatch, [reply("<think>plan</think>\nTITLE: X\n\nBody.")])
    r = llm.complete("sys", "prompt")
    assert r.text == "TITLE: X\n\nBody." and (r.input_tokens, r.output_tokens) == (10, 5)
    kw = fake.calls[0]
    assert kw["think"] is False and kw["model"] == "qwen3:32b-q8_0"
    assert kw["options"] == {"temperature": 0.7, "num_predict": 4096, "num_ctx": 16384, "seed": 42}
    assert [m["role"] for m in kw["messages"]] == ["system", "user"]
    assert llm.complete("sys", "prompt").cached  # second call served from cache


def test_truncated_reply_is_retried_not_cached(tmp_path, monkeypatch):
    llm, fake = make(tmp_path, monkeypatch, [reply("TITLE: X\n\nloop loop", "length"), reply("TITLE: X\n\nok")])
    assert llm.complete("sys", "p").text == "TITLE: X\n\nok"
    assert len(fake.calls) == 2
    cached = list((tmp_path / "c").glob("*.json"))
    assert len(cached) == 1 and "loop" not in cached[0].read_text()


def test_all_attempts_truncated_raises(tmp_path, monkeypatch):
    llm, _ = make(tmp_path, monkeypatch, [reply("x", "length")] * 3)
    with pytest.raises(RuntimeError, match="max_tokens"):
        llm.complete("sys", "p")
    assert not list((tmp_path / "c").glob("*.json"))


def test_full_config_uses_ollama_and_host_override():
    from corpus_gen.verbalize import yaml_value
    cfg = load_config(ROOT / "configs/full.yaml", {"llm.host": yaml_value("http://127.0.0.1:41234")})
    assert cfg.llm.provider == "ollama" and cfg.llm.model == "qwen3:32b-q8_0"
    assert cfg.llm.host == "http://127.0.0.1:41234" and cfg.llm.think is False
    assert cfg.run_name == "full_cf0.3_s42"


def test_concurrent_identical_prompts_never_read_a_partial_cache_file(tmp_path, monkeypatch):
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    real_write = Path.write_text

    def slow_write(self, data, *a, **kw):  # widen the window between create and finish
        with open(self, "w") as fh:
            fh.write(data[: len(data) // 2])
            fh.flush()
            _time.sleep(0.2)
            fh.write(data[len(data) // 2:])
        return len(data)

    monkeypatch.setattr(Path, "write_text", slow_write)
    llm = llm_mod.LLM(LLMParams(provider="mock", cache_dir="c"), root=tmp_path)
    first = threading.Event()
    real_call = llm._call_with_retry

    def call(system, prompt):
        first.set()
        return real_call(system, prompt)

    llm._call_with_retry = call
    prompt = "FACTS\n- Atlas ships in April.\nabout 20 words"

    def late_reader():
        first.wait()
        _time.sleep(0.1)  # arrives while the writer is mid-file
        return llm.complete("s", prompt).text

    with ThreadPoolExecutor(2) as ex:
        a = ex.submit(llm.complete, "s", prompt)
        b = ex.submit(late_reader)
        assert a.result().text == b.result()
    assert [f.name for f in (tmp_path / "c").iterdir() if f.suffix == ".tmp"] == []
