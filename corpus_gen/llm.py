"""LLM backends with an on-disk response cache.

Providers (cfg.llm.provider):
  claude_code  headless Claude Code (`claude -p`), uses the CLI's own login; no API key.
               Temperature is not settable via the CLI; reproducibility comes from the cache.
  anthropic    Anthropic Python SDK; needs ANTHROPIC_API_KEY.
  ollama       local Ollama server (llm.host), e.g. qwen3:32b-q8_0 on the cluster H100.
               A response that hits max_tokens (done_reason=length) is retried, never cached.
  mock         deterministic offline stub for pipeline tests (no network).

Cache key = sha256(provider, model, system, prompt, temperature, max_tokens, salt).
Re-running with the same plan reproduces every document byte-for-byte; delete the
cache dir (or change `salt`) to resample.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import LLMParams


@dataclass
class LLMResult:
    text: str
    cached: bool
    cost_usd: float
    input_tokens: int
    output_tokens: int


class LLM:
    def __init__(self, p: LLMParams, root: Path | None = None):
        self.p = p
        self.cache = (root or Path.cwd()) / p.cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        self.calls = 0
        if p.provider == "anthropic":
            import anthropic  # noqa: PLC0415
            self._client = anthropic.Anthropic()
        elif p.provider == "ollama":
            import ollama  # noqa: PLC0415
            self._client = ollama.Client(host=p.host, timeout=p.timeout)

    # -------------------------------------------------------------- public
    def complete(self, system: str, prompt: str, salt: str = "") -> LLMResult:
        key = hashlib.sha256(json.dumps(
            [self.p.provider, self.p.model, system, prompt, self.p.temperature,
             self.p.max_tokens, salt]).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            d = json.loads(path.read_text())
            return LLMResult(d["text"], True, 0.0, d.get("input_tokens", 0), d.get("output_tokens", 0))
        res = self._call_with_retry(system, prompt)
        # Atomic: identical prompts run concurrently (e.g. gold answers for access sets
        # with the same authorized facts), and a reader must never see a partial file.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"text": res.text, "provider": self.p.provider,
                                   "model": self.p.model, "input_tokens": res.input_tokens,
                                   "output_tokens": res.output_tokens, "cost_usd": res.cost_usd}))
        os.replace(tmp, path)
        with self._lock:
            self.spent_usd += res.cost_usd
            self.calls += 1
        return res

    # ------------------------------------------------------------- backends
    def _call_with_retry(self, system: str, prompt: str) -> LLMResult:
        last: Exception | None = None
        for attempt in range(self.p.max_retries):
            try:
                return getattr(self, f"_call_{self.p.provider}")(system, prompt)
            except Exception as e:  # noqa: BLE001  transient CLI/API failures
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.p.max_retries} attempts: {last}")

    def _call_claude_code(self, system: str, prompt: str) -> LLMResult:
        cmd = ["claude", "-p", "--output-format", "json", "--model", self.p.model,
               "--system-prompt", system, "--tools", "", "--strict-mcp-config",
               "--setting-sources", "", "--no-session-persistence"]
        # prompt via stdin: avoids argv length limits and shell quoting issues
        out = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                             timeout=600, cwd=str(self.cache))
        if out.returncode != 0:
            raise RuntimeError(f"claude -p exit {out.returncode}: {out.stderr[-500:]}")
        d = json.loads(out.stdout)
        if d.get("is_error") or d.get("subtype") not in (None, "success"):
            raise RuntimeError(f"claude -p error: {str(d)[:500]}")
        u = d.get("usage", {})
        return LLMResult(d["result"], False, float(d.get("total_cost_usd") or 0.0),
                         int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
                         + int(u.get("cache_read_input_tokens", 0)),
                         int(u.get("output_tokens", 0)))

    def _call_anthropic(self, system: str, prompt: str) -> LLMResult:
        msg = self._client.messages.create(
            model=self.p.model, max_tokens=self.p.max_tokens, temperature=self.p.temperature,
            system=system, messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return LLMResult(text, False, 0.0, msg.usage.input_tokens, msg.usage.output_tokens)

    def _call_ollama(self, system: str, prompt: str) -> LLMResult:
        options = {"temperature": self.p.temperature, "num_predict": self.p.max_tokens,
                   "num_ctx": self.p.num_ctx}
        if self.p.seed is not None:
            options["seed"] = self.p.seed
        r = self._client.chat(model=self.p.model, think=self.p.think, options=options,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}])
        if r.get("done_reason") == "length":
            raise RuntimeError(f"ollama output hit max_tokens={self.p.max_tokens} (runaway or cap too low)")
        text = re.sub(r"<think>.*?</think>\s*", "", r["message"]["content"] or "", flags=re.S).strip()
        if not text:
            raise RuntimeError("ollama returned an empty response")
        return LLMResult(text, False, 0.0, int(r.get("prompt_eval_count") or 0), int(r.get("eval_count") or 0))

    def _call_mock(self, system: str, prompt: str) -> LLMResult:
        """Echo the FACTS block as prose; good enough to exercise the checks."""
        facts = re.findall(r"^- (.+)$", prompt.split("FACTS", 1)[-1], flags=re.M)
        title = "Mock document"
        filler = ("The group reviewed progress and agreed to revisit open items at the next "
                  "checkpoint. Owners will circulate updates before the review. ")
        m = re.search(r"about (\d+) words", prompt)
        target = int(m.group(1)) if m else 400
        body = " ".join(facts)
        while len(body.split()) < target:
            body += " " + filler
        return LLMResult(f"TITLE: {title}\n\n{body.strip()}", False, 0.0, 0, len(body.split()))


def check_provider(p: LLMParams) -> None:
    """Fail fast with an actionable message before a long run."""
    if p.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("llm.provider=anthropic needs ANTHROPIC_API_KEY (or use provider: claude_code)")
    if p.provider == "ollama":
        import ollama  # noqa: PLC0415
        try:
            models = {m.model for m in ollama.Client(host=p.host, timeout=30).list().models}
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"llm.provider=ollama: no Ollama server at {p.host}: {e}")
        if p.model not in models:
            raise SystemExit(f"llm.provider=ollama: model {p.model!r} not pulled on {p.host} (have {sorted(models)})")
    if p.provider == "claude_code":
        try:
            subprocess.run(["claude", "--version"], capture_output=True, check=True, timeout=30)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"llm.provider=claude_code needs the `claude` CLI on PATH: {e}")
