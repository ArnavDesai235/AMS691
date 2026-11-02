"""Deterministic repairs applied to a draft before the hard checks (verbalize.check_doc).

Repairs only move text toward the form the checks demand, so they cannot weaken the
corpus: a repaired draft must still pass every check, and ground truth (names, codes,
values) is never invented here.

  names     a bare first name, surname or distinctive token outside a full-name mention
            is expanded to the full canonical name. Names are unique coined tokens, so
            the mapping is unambiguous. Adjacent words that already belong to the name
            ("Talhuor Integration team", "project Talhuor") are absorbed, not duplicated.
  canaries  a planned code that appears more than once keeps the one occurrence that sits
            in a sentence with its entity's full name; the others are removed. Nothing is
            done if no such anchored occurrence exists (the LLM must fix that).
"""
from __future__ import annotations

import re

_SEP = re.compile(r"(?<=[.!?])\s+|\n+")
_ID_NOUN = r"(?:code|ID|id|number|key|reference|identifier|tag)"
_MAX_EDITS = 500


def _spans(text: str, needle: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in _SEP.finditer(text):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(text)))
    return [(a, b) for a, b in out if text[a:b].strip()]


# ------------------------------------------------------------------- names
def expand_names(text: str, names: dict[str, list[str]]) -> tuple[str, list[str]]:
    """names: canonical name -> short tokens to expand (e.g. {"Vadesis Gummenbel": ["Vadesis", "Gummenbel"]})."""
    log: list[str] = []
    for canon, toks in names.items():
        parts = canon.split()
        lower = [p.lower() for p in parts]
        for tok in toks:
            if tok.lower() not in lower:
                continue
            idx = lower.index(tok.lower())
            pat = re.compile(rf"\b{re.escape(tok)}\b", flags=re.I)
            for _ in range(_MAX_EDITS):
                covered = _spans(text, canon)
                m = next((m for m in pat.finditer(text)
                          if not any(a <= m.start() < b for a, b in covered)), None)
                if m is None:
                    break
                start, end, i, j = m.start(), m.end(), idx, idx
                while i > 0:  # absorb preceding name words: "project Talhuor" -> one name
                    p = re.search(r"(\w+)[ \t]+$", text[:start])
                    if not p or p.group(1).lower() != lower[i - 1]:
                        break
                    start, i = p.start(1), i - 1
                while j < len(parts) - 1:  # absorb following name words
                    n = re.match(r"[ \t]+(\w+)\b", text[end:])
                    if not n or n.group(1).lower() != lower[j + 1]:
                        break
                    end, j = end + n.end(1), j + 1
                if j < len(parts) - 1:  # "Talhuor team" -> generic tail word belongs to the name
                    n = re.match(r"[ \t]+(\w+)\b", text[end:])
                    if n and n.group(1).lower() == lower[-1]:
                        end += n.end(1)
                log.append(f'name "{text[start:end]}" -> "{canon}"')
                text = text[:start] + canon + text[end:]
    return text, log


# ---------------------------------------------------------------- canaries
def _scrub(text: str, code: str) -> str:
    c = re.escape(code)
    text = re.sub(rf"[ \t]*\((?:[^()\n]{{0,30}}?:\s*)?{c}\s*\)", "", text)        # "(key ID: X)" / "(X)"
    text = re.sub(rf"\b({_ID_NOUN})\s*:?\s*{c}", r"\1 noted earlier", text)       # "budget code X"
    return text.replace(code, "the same reference")                               # anything else


def dedupe_codes(title: str, body: str, codes: dict[str, str]) -> tuple[str, str, list[str]]:
    """codes: canary code -> canonical name of its entity."""
    log: list[str] = []
    for code, canon in codes.items():
        n = title.count(code) + body.count(code)
        if n <= 1 and code not in title:
            continue
        keep = None
        for a, b in _sentence_spans(body):
            s = body[a:b]
            if code in s and canon in s:
                keep = a + s.index(code)
                break
        if keep is None:
            continue
        mark = "\x00KEEP\x00"
        body = body[:keep] + mark + body[keep + len(code):]
        body = _scrub(body, code).replace(mark, code)
        title = _scrub(title, code) if code in title else title
        log.append(f"code {code}: kept 1 of {n} occurrences")
    return title, body, log


# ------------------------------------------------------------------- values
def insert_facts(body: str, missing: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """missing: (planned fact sentence, canonical name to anchor on).

    Each sentence is inserted verbatim as its own paragraph right after the first
    paragraph that names the anchor entity (at the end if none does). The sentence
    comes from the fact table, so nothing is invented.
    """
    log: list[str] = []
    for sentence, anchor in missing:
        i = body.find(anchor)
        if i < 0:
            body = body.rstrip() + "\n\n" + sentence
        else:
            end = body.find("\n\n", i)
            end = len(body) if end < 0 else end
            body = body[:end].rstrip() + "\n\n" + sentence + body[end:]
        log.append(f"inserted fact: {sentence}")
    return body, log
