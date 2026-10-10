"""Verbalize planned documents with an LLM, then hard-check every draft.

A draft is accepted only if ALL checks pass; otherwise the violations are fed
back and the model rewrites (up to MAX_DOC_ATTEMPTS). A doc that never passes
fails the run: we never ship a silently weaker corpus.

Checks (why each exists):
  names      every planned canonical name appears verbatim   (LightRAG merges by exact,
             case-sensitive name; a missing name = a missing node fragment)
  short      no bare short forms ("Kistos", "Rakresett") or first names alone
             (they become separate LightRAG nodes -> merge never happens)
  foreign    no other entity's name/token appears            (restricted-only names
             in public docs would be structural leakage baked into the corpus)
  canary     each planned code appears exactly once, in the same sentence as its
             entity's canonical name; never in the title; no other code-shaped
             string or other canary variant anywhere
  values     every literal value of a planned fact appears   (gold answers rely on it)
  invented   no invented proper nouns (Titlecase, non-dictionary, not allowed)
  length     word count within tolerance of the target
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, dump_config, load_config
from .facts import Entity, Fact, FactTable, build_fact_table, write_fact_table
from .llm import LLM, check_provider
from .names import _DICTIONARY, CODE_REGEX, word_frequency
from .plan import DocPlan, Plan, build_plan, write_plan

MAX_DOC_ATTEMPTS = 4
LEN_TOL = (0.75, 1.35)

SYSTEM = ("You write realistic internal business documents for a fictional company. "
          "You follow the given constraints exactly. Output only the requested document, "
          "with no preamble or commentary.")

DOC_STYLE = {
    "memo": "an internal memo with To/From/Date/Subject header lines (use roles, not personal "
            "names, in To/From unless the person is named in the facts) followed by prose paragraphs",
    "meeting_notes": "meeting notes with a date line, an attendees line, discussion points, "
                     "decisions, and action items (attendees: only people named in the facts, "
                     "or roles such as 'finance representative')",
    "wiki_page": "an internal wiki page with a short summary and a few headed sections",
    "status_report": "a periodic status report with sections such as Summary, Progress, Risks, "
                     "and Next Steps",
    "email_thread": "an email thread of 2-4 messages with From/To/Subject lines (use roles or "
                    "people named in the facts; no email addresses), newest message last",
}
AUDIENCE = {
    "public": "company-wide (all employees)",
    "restricted": "a restricted internal audience; the tone can reflect that it is not company-wide",
}
_ALLOWED_TITLECASE = {"Mon", "Tue", "Wed", "Thu", "Fri", "Re", "Fwd", "Cc", "Bcc", "Ok",
                      "Workstream", "Workstreams", "Standup", "Offsite"}
# Real tools/brands are common words in web text, so frequency alone won't catch them.
_BRANDS = {"Jira", "Slack", "Zoom", "Confluence", "Salesforce", "Workday", "Microsoft",
           "Google", "Amazon", "Azure", "Excel", "Outlook", "Notion", "Asana", "Tableau",
           "Snowflake", "Okta", "Servicenow", "ServiceNow", "Github", "GitHub", "Gitlab",
           "Trello", "Monday", "Smartsheet", "Sharepoint", "SharePoint", "Oracle", "Webex",
           "Datadog", "Splunk", "Looker", "Figma", "Miro", "Dropbox", "Box", "Gmail"}
MIN_COMMON_ZIPF = 1.5  # Titlecase words rarer than this (and not in the wordlist) are flagged


@dataclass
class DocResult:
    doc_id: str
    ok: bool
    title: str = ""
    text: str = ""
    attempts: int = 0
    violations: list[list[str]] = field(default_factory=list)  # per attempt
    cost_usd: float = 0.0


# ------------------------------------------------------------------ prompts
def build_prompt(cfg: Config, ft: FactTable, d: DocPlan) -> str:
    em = {e.entity_id: e for e in ft.entities}
    fm = {f.fact_id: f for f in ft.facts}
    facts = [fm[x] for x in d.fact_ids]
    names = [em[e].canonical_name for e in d.entities]
    canaries = [f for f in facts if f.is_canary]
    audience = AUDIENCE["public" if d.tier == cfg.public_tier else "restricted"]
    lines = [
        f"Write {DOC_STYLE[d.doc_type]}, about {d.target_words} words, for {ft.company}.",
        f"Audience: {audience}.",
        "",
        "Start with one line of the form `TITLE: <a realistic title>`, then a blank line, then the document.",
        "",
        "FACTS (state every one explicitly; you may rephrase the wording, but copy names and values exactly):",
        *[f"- {f.sentence}" for f in facts],
        "",
        "RULES:",
        f"1. Every one of these names must appear exactly as written, every time it is mentioned: "
        + "; ".join(f'"{n}"' for n in names) + ".",
        "2. Never shorten a name: no first names alone, no surnames alone, no dropping words like "
        "\"Project\" or \"Team\". After the first full mention you may use pronouns or generic "
        "references (\"the project\", \"she\"), but any time you use the name, use the full name.",
        f"3. Do not mention any other named person, team, project, product, vendor, site, place, "
        f"customer, or real-world company or tool. Refer to anyone else generically (\"the finance "
        f"lead\", \"a vendor\"). The company name \"{ft.company}\" may appear at most twice.",
        "4. Do not invent identifiers, ticket numbers, codes, or email addresses.",
        "5. Do not add new numbers, dates, budgets, or statuses about the named entities beyond the "
        "facts; generic context (process, discussion, next steps) is fine.",
    ]
    if canaries:
        lines.append(
            "6. Each of these reference codes must appear exactly once in the whole document, written "
            "exactly as shown, in the same sentence as the full name it belongs to, and never in the "
            "title: " + "; ".join(f"{f.object['literal']} (belongs to {em[f.subject].canonical_name})"
                                   for f in canaries) + ".")
    return "\n".join(lines)


def build_fix_prompt(base: str, draft: str, violations: list[str]) -> str:
    return (base + "\n\nYour previous draft is below. It violated these rules:\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\n\nReturn the complete corrected document (same format, starting with TITLE:)."
            + "\n\nPREVIOUS DRAFT:\n" + draft)


# ------------------------------------------------------------------- checks
def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _spans(text: str, needle: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]


def parse_draft(raw: str) -> tuple[str, str] | None:
    raw = raw.strip().strip("`").strip()
    m = re.match(r"TITLE:\s*(.+?)\s*\n", raw)
    if not m:
        return None
    return m.group(1).strip(), raw[m.end():].strip()


def check_doc(cfg: Config, ft: FactTable, d: DocPlan, title: str, body: str,
              allowed_tokens: set[str]) -> list[str]:
    em = {e.entity_id: e for e in ft.entities}
    fm = {f.fact_id: f for f in ft.facts}
    cm = {c.canary_id: c for c in ft.canaries}
    text = f"{title}\n\n{body}"
    v: list[str] = []
    planned = [em[e] for e in d.entities]

    # names
    for e in planned:
        if e.canonical_name not in text:
            v.append(f'missing required name "{e.canonical_name}"')

    # short forms: any token of a planned name outside a full-name occurrence
    covered = [sp for e in planned for sp in _spans(text, e.canonical_name)]
    for e in planned:
        toks = [e.canonical_name.split()[0]] if e.type == "person" else []
        toks.append(_distinctive(e))
        for tok in toks:
            for m in re.finditer(rf"\b{re.escape(tok)}\b", text, flags=re.I):
                if not any(a <= m.start() < b for a, b in covered):
                    v.append(f'shortened or altered name "{m.group(0)}" (always write "{e.canonical_name}")')
                    break

    # foreign entities
    planned_ids = set(d.entities)
    for e in ft.entities:
        if e.entity_id in planned_ids:
            continue
        hit = e.canonical_name in text or re.search(rf"\b{re.escape(_distinctive(e))}\b", text, flags=re.I)
        if e.type == "person" and not hit:
            hit = re.search(rf"\b{re.escape(e.canonical_name.split()[0])}\b", text)
        if hit:
            v.append(f'mentions "{e.canonical_name}", which must not appear in this document')

    # canaries
    planned_codes = {cm[c].exact: cm[c] for c in d.canary_ids}
    for code, c in planned_codes.items():
        n = text.count(code)
        if n != 1:
            v.append(f"code {code} must appear exactly once (found {n})")
        if code in title:
            v.append(f"code {code} must not appear in the title")
        name = em[c.entity_id].canonical_name
        if n >= 1 and not any(code in s and name in s for s in _sentences(body)):
            v.append(f'code {code} must be in the same sentence as "{name}"')
    for m in CODE_REGEX.finditer(text):
        if m.group(0) not in planned_codes:
            v.append(f"unexpected code-like string {m.group(0)}")
    for c in ft.canaries:
        if c.canary_id not in d.canary_ids and re.search(c.match_regex, text):
            v.append(f"contains another document's reference code ({c.exact})")

    # values
    for fid in d.fact_ids:
        f = fm[fid]
        lit = f.object.get("literal")
        core = re.sub(r"^(about|the)\s+", "", lit or "", flags=re.I)  # "about 3,400 ..." -> "3,400 ..."
        if lit and not f.is_canary and core.lower() not in text.lower():
            v.append(f'missing value "{core}" (from: {f.sentence})')

    # invented / real-world proper nouns: brands anywhere; otherwise Titlecase words that
    # are neither in the wordlist nor common in web text (coined names score ~0)
    invented = set()
    for s in _sentences(text):
        words = re.findall(r"(?<![\w'-])([A-Z][a-zA-Z]{2,})(?![\w-])", s)
        first = re.match(r"\W*([A-Za-z]+)", s)
        for w in words:
            if w in allowed_tokens:
                continue
            if w in _BRANDS:
                invented.add(w)
                continue
            if first and w == first.group(1):
                continue
            if w in _ALLOWED_TITLECASE or _in_dictionary(w) or word_frequency(w) >= MIN_COMMON_ZIPF:
                continue
            invented.add(w)
    if invented:
        v.append("invented or real-world proper nouns not allowed: " + ", ".join(sorted(invented)))

    # length
    n_words = len(body.split())
    lo = max(cfg.docs.words_min * 0.9, d.target_words * LEN_TOL[0])
    hi = min(cfg.docs.words_max * 1.1, d.target_words * LEN_TOL[1])
    if not lo <= n_words <= hi:
        v.append(f"length {n_words} words; target about {d.target_words} (allowed {int(lo)}-{int(hi)})")
    return v


def _distinctive(e: Entity) -> str:
    """Unique token of the canonical name (surname for people, coined word otherwise)."""
    parts = e.canonical_name.split()
    if e.type == "person":
        return parts[-1]
    if e.type == "project":
        return parts[1]
    return parts[0]


def _in_dictionary(w: str) -> bool:
    lw = w.lower()
    cands = {lw, lw[:-1] if lw.endswith("s") else lw, lw[:-2] if lw.endswith("es") else lw,
             lw[:-3] + "y" if lw.endswith("ies") else lw,
             lw[:-2] if lw.endswith("ed") else lw, lw[:-1] if lw.endswith("ed") else lw,
             lw[:-3] if lw.endswith("ing") else lw}
    return any(c in _DICTIONARY for c in cands)


def _allowed_tokens(ft: FactTable, d: DocPlan) -> set[str]:
    em = {e.entity_id: e for e in ft.entities}
    fm = {f.fact_id: f for f in ft.facts}
    toks = set(ft.company.split())
    for eid in d.entities:
        toks.update(em[eid].canonical_name.split())
    for fid in d.fact_ids:  # words from the doc's own fact values (e.g. job titles)
        toks.update(re.findall(r"[A-Za-z]+", fm[fid].object.get("literal", "")))
    return toks


# ---------------------------------------------------------------- generation
def verbalize_doc(cfg: Config, ft: FactTable, d: DocPlan, llm: LLM) -> DocResult:
    base = build_prompt(cfg, ft, d)
    allowed = _allowed_tokens(ft, d)
    res = DocResult(d.doc_id, False)
    prompt, draft = base, ""
    for attempt in range(1, MAX_DOC_ATTEMPTS + 1):
        out = llm.complete(SYSTEM, prompt)
        res.cost_usd += out.cost_usd
        res.attempts = attempt
        draft = out.text
        parsed = parse_draft(draft)
        viol = ["missing the first line `TITLE: ...`"] if parsed is None else \
            check_doc(cfg, ft, d, parsed[0], parsed[1], allowed)
        res.violations.append(viol)
        if not viol:
            res.ok, res.title, res.text = True, parsed[0], f"{parsed[0]}\n\n{parsed[1]}\n"
            return res
        prompt = build_fix_prompt(base, draft, viol)
    return res


def verbalize_all(cfg: Config, ft: FactTable, plan: Plan, llm: LLM,
                  only: list[str] | None = None) -> dict[str, DocResult]:
    docs = [d for d in plan.docs if not only or d.doc_id in only]
    results: dict[str, DocResult] = {}
    with ThreadPoolExecutor(max_workers=cfg.llm.concurrency) as ex:
        futs = {ex.submit(verbalize_doc, cfg, ft, d, llm): d.doc_id for d in docs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[r.doc_id] = r
            status = "ok " if r.ok else "FAIL"
            print(f"[{i:>3}/{len(docs)}] {status} {r.doc_id} attempts={r.attempts} "
                  f"spent=${llm.spent_usd:.2f}", flush=True)
    return results


# ------------------------------------------------------------------- writing
def write_corpus(cfg: Config, ft: FactTable, plan: Plan, results: dict[str, DocResult]) -> dict[str, Any]:
    out = cfg.out_dir
    (out / "docs").mkdir(parents=True, exist_ok=True)
    em = {e.entity_id: e for e in ft.entities}
    cm = {c.canary_id: c for c in ft.canaries}
    rows = []
    for d in plan.docs:
        r = results.get(d.doc_id)
        path = out / "docs" / f"{d.doc_id}.txt"
        if (not r or not r.ok) and path.exists():  # partial (--only) run: keep the doc on disk
            prev = path.read_text()
            r = DocResult(d.doc_id, True, title=prev.split("\n", 1)[0], text=prev)
        if not r or not r.ok:
            continue
        path.write_text(r.text)
        mentions = {eid: [list(sp) for sp in _spans(r.text, em[eid].canonical_name)] for eid in d.entities}
        for cid in d.canary_ids:
            c = cm[cid]
            c.origin_doc_id = d.doc_id
            c.origin_char_span = list(_spans(r.text, c.exact)[0])
        rows.append({
            "doc_id": d.doc_id, "title": r.title, "doc_type": d.doc_type,
            "acl_label": d.tier, "acl_groups": d.acl_groups, "acl_users": d.acl_users,
            "entities": d.entities, "entity_names": [em[e].canonical_name for e in d.entities],
            "entity_mentions": mentions, "fact_ids": d.fact_ids,
            "bridge_fact_ids": d.bridge_fact_ids, "canary_ids": d.canary_ids,
            "n_words": len(r.text.split()), "sha256": hashlib.sha256(r.text.encode()).hexdigest(),
        })
    with (out / "manifest.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    write_fact_table(ft, out)      # canaries.jsonl now carries origin doc + char span
    write_plan(plan, out)
    dump_config(cfg, out / "config.resolved.yaml")
    report = {
        "n_docs_planned": len(plan.docs),
        "n_docs_ok": sum(r.ok for r in results.values()),
        "failed": [r.doc_id for r in results.values() if not r.ok],
        "attempts_hist": _hist([r.attempts for r in results.values()]),
        "first_try_violation_types": _violation_types([r.violations[0] for r in results.values() if r.violations]),
        "cost_usd_this_run": round(sum(r.cost_usd for r in results.values()), 4),
        "plan_stats": plan.stats, "fact_stats": ft.stats,
    }
    (out / "gen_report.json").write_text(json.dumps(report, indent=2))
    failed_detail = {r.doc_id: r.violations[-1] for r in results.values() if not r.ok}
    if failed_detail:
        (out / "gen_failures.json").write_text(json.dumps(failed_detail, indent=2))
    return report


def _hist(xs: list[int]) -> dict[str, int]:
    h: dict[str, int] = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return dict(sorted(h.items()))


def _violation_types(vs: list[list[str]]) -> dict[str, int]:
    keys = {"missing required name": "names", "shortened": "short", "must not appear": "foreign",
            "code": "canary", "missing value": "values", "proper nouns": "invented",
            "length": "length", "TITLE": "format"}
    h: dict[str, int] = {}
    for doc_v in vs:
        seen = set()
        for msg in doc_v:
            k = next((t for s, t in keys.items() if s in msg), "other")
            seen.add(k)
        for k in seen:
            h[k] = h.get(k, 0) + 1
    return h


# ----------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate documents for one corpus configuration.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. entities.cross_tier_fraction=0.5")
    ap.add_argument("--only", nargs="*", default=None, help="doc_ids to (re)generate")
    ap.add_argument("--limit", type=int, default=None, help="first N planned docs (smoke test)")
    a = ap.parse_args()

    overrides = {k: yaml_value(v) for k, v in (s.split("=", 1) for s in a.set)}
    cfg = load_config(a.config, overrides)
    check_provider(cfg.llm)
    ft = build_fact_table(cfg)
    plan = build_plan(cfg, ft)
    only = a.only or ([d.doc_id for d in plan.docs[: a.limit]] if a.limit else None)
    llm = LLM(cfg.llm)
    results = verbalize_all(cfg, ft, plan, llm, only)
    report = write_corpus(cfg, ft, plan, results)
    print(json.dumps({k: report[k] for k in ("n_docs_planned", "n_docs_ok", "failed",
                                              "attempts_hist", "first_try_violation_types",
                                              "cost_usd_this_run")}, indent=2))
    if report["failed"]:
        sys.exit(1)


def yaml_value(s: str) -> Any:
    import yaml  # noqa: PLC0415
    return yaml.safe_load(s)


if __name__ == "__main__":
    main()
