"""C1/C2/C3 enforcement + cache keying, run through LightRAG 1.5.7's real kg_query.

    python -m pytest tests/test_acl_enforcement.py        # with lightrag-hku installed
    python tests/test_acl_enforcement.py                  # same, no pytest needed
    LIGHTRAG_SRC=/path/to/LightRAG python tests/...       # from a source checkout

Every (query, user) row of the dev corpus is run with the target's name as the
retrieval keyword. ACLs are re-derived here from the graph's *full* source_id and
the manifest, independently of acl_rag, so the code is not grading itself.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from fake_lightrag import FakeRAG, import_operate  # noqa: E402

operate = import_operate()

from acl_rag import ACLIndex, Mode, acting_as, install, set_c4_description_filter  # noqa: E402

CORPUS = ROOT / "corpus" / "dev_cf0.3_s42"
SEP = "<SEP>"


def _rows():
    with open(CORPUS / "queries.jsonl") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _manifest():
    with open(CORPUS / "manifest.jsonl") as fh:
        return {d["doc_id"]: set(d["acl_users"]) for d in map(json.loads, fh)}


def _canary_res():
    with open(CORPUS / "canaries.jsonl") as fh:
        return {c["canary_id"]: re.compile(c["match_regex"]) for c in map(json.loads, fh)}


ROWS, DOC_USERS, CANARY = _rows(), _manifest(), _canary_res()
with open(CORPUS / "canaries.jsonl") as _fh:
    KIND = {c["canary_id"]: c["kind"] for c in map(json.loads, _fh)}


def fresh():
    rag = FakeRAG(CORPUS, operate)
    acl = ACLIndex(CORPUS)
    install(rag, acl, operate=operate)
    return rag, acl


# ---------------------------------------------------------------- independent oracle
def chunk_users(rag, cid):
    row = rag.text_chunks.rows.get(cid) or {}
    return DOC_USERS.get(row.get("full_doc_id"), set())


def union_users(rag, source_id):
    out = set()
    for c in (source_id or "").split(SEP):
        if c.strip():
            out |= chunk_users(rag, c.strip())
    return out


def violations(rag, user, raw):
    """Retrieved items the user may not see (structural leakage), by item kind."""
    g = rag.chunk_entity_relation_graph
    d = raw["data"]
    bad = []
    for e in d["entities"]:
        full = g.nodes[e["entity_name"]]["source_id"]
        if user not in union_users(rag, full):
            bad.append(("entity", e["entity_name"]))
    for r in d["relationships"]:
        full = (g._edge(r["src_id"], r["tgt_id"]) or {}).get("source_id")
        if user not in union_users(rag, full):
            bad.append(("relation", (r["src_id"], r["tgt_id"])))
    for c in d["chunks"]:
        if user not in chunk_users(rag, c["chunk_id"]):
            bad.append(("chunk", c["chunk_id"]))
    return bad


def served_text(raw):
    d = raw["data"]
    return "\n".join([e["description"] for e in d["entities"]] +
                     [r["description"] for r in d["relationships"]] +
                     [c["content"] for c in d["chunks"]])


async def run_all(mode: str):
    rag, acl = fresh()
    out = {}
    for row in ROWS:
        with acting_as(acl, row["user_id"], mode):
            res = await rag.query(row["text"], [row["target_name"]])
        out[row["qid"]] = (row, res.raw_data if res is not None else None)
    return rag, out


_CACHE: dict[str, tuple] = {}


def results(mode: str):
    if mode not in _CACHE:
        _CACHE[mode] = asyncio.run(run_all(mode))
    return _CACHE[mode]


# ---------------------------------------------------------------- tests
def test_c1_leaks_so_the_check_has_teeth():
    rag, res = results("C1")
    n = sum(bool(violations(rag, row["user_id"], raw)) for row, raw in res.values() if raw)
    assert n > 0, "C1 should retrieve unauthorized items for some rows"


def test_c2_and_c3_never_return_unauthorized_items():
    for mode in ("C2", "C3"):
        rag, res = results(mode)
        bad = {qid: violations(rag, row["user_id"], raw) for qid, (row, raw) in res.items() if raw}
        bad = {k: v for k, v in bad.items() if v}
        assert not bad, f"{mode} structural leakage: {list(bad.items())[:3]}"


def test_c2_and_c3_no_structural_canary_on_type_b():
    """Structural canaries (on restricted-only entities) must vanish everywhere.
    Contamination canaries may still ride in an authorized neighbour's merged
    description (that is contamination, not structural leakage), but never in
    a served chunk."""
    for mode in ("C2", "C3"):
        _, res = results(mode)
        for row, raw in res.values():
            if raw is None or row["type"] != "b" or row["eval_role"] != "probe":
                continue
            chunks = "\n".join(c["content"] for c in raw["data"]["chunks"])
            everything = served_text(raw)
            for cid in row["target_canary_ids"] + row["neighbor_canary_ids"]:
                where = everything if KIND[cid] == "structural" else chunks
                assert not CANARY[cid].search(where), f"{mode} {row['qid']} leaked {cid}"


def test_c3_still_has_contamination_on_type_a():
    """C3 must not touch merged descriptions; contamination is C4's job."""
    _, res = results("C3")
    hits = 0
    for row, raw in res.values():
        if raw is None or row["type"] != "a" or row["eval_role"] != "probe":
            continue
        desc = "\n".join(e["description"] for e in raw["data"]["entities"])
        hits += any(CANARY[c].search(desc) for c in row["forbidden_canary_ids"])
    assert hits > 0, "expected contaminated descriptions under C3 on type-a probes"


def test_positive_controls_keep_their_target_canary():
    for mode in ("C1", "C2", "C3"):
        _, res = results(mode)
        missing = []
        for row, raw in res.values():
            if row["eval_role"] != "positive_control" or not row["authorized_canary_ids"]:
                continue
            text = served_text(raw) if raw else ""
            tgt = [c for c in row["authorized_canary_ids"]
                   if c not in row["neighbor_canary_ids"]]
            if tgt and not any(CANARY[c].search(text) for c in tgt):
                missing.append(row["qid"])
        assert not missing, f"{mode} dropped authorized canaries: {missing[:5]}"


def test_users_in_same_access_set_get_identical_retrieval():
    for mode in ("C2", "C3"):
        _, res = results(mode)
        seen = {}
        for row, raw in res.values():
            key = (row["query_id"], row["access_set"])
            sig = None if raw is None else json.dumps(
                [[e["entity_name"] for e in raw["data"]["entities"]],
                 [c["chunk_id"] for c in raw["data"]["chunks"]]])
            assert seen.setdefault(key, sig) == sig, f"{mode} {key} differs between users"


def test_c3_fills_top_k_with_authorized_chunks():
    """Pre-filtering oversamples, so C3 keeps at least as many chunks as C2 drops to."""
    _, c2 = results("C2")
    _, c3 = results("C3")
    fewer = more = 0
    for qid, (row, raw2) in c2.items():
        raw3 = c3[qid][1]
        n2 = len(raw2["data"]["chunks"]) if raw2 else 0
        n3 = len(raw3["data"]["chunks"]) if raw3 else 0
        fewer += n3 < n2
        more += n3 > n2
    assert more > 0 and fewer == 0, (more, fewer)


def test_no_scope_is_stock_lightrag():
    async def go():
        rag, acl = fresh()
        row = ROWS[0]
        plain = await rag.query(row["text"], [row["target_name"]])
        with acting_as(acl, row["user_id"], "C1"):
            c1 = await rag.query(row["text"], [row["target_name"]])
        return plain.raw_data["data"], c1.raw_data["data"]
    a, b = asyncio.run(go())
    assert a == b


def test_answer_cache_never_crosses_access_sets():
    async def go():
        rag, acl = fresh()
        q = next(r for r in ROWS if r["type"] == "a" and r["user_tier"] == "exec")
        answers = {}
        for user in ("u_exec_1", "u_public_1", "u_exec_2", "u_public_2"):
            with acting_as(acl, user, "C3"):
                res = await rag.query(q["text"], [q["target_name"]])
            answers[user] = res.content if res else None
        c3_calls = len(rag.llm_calls)
        for user in ("u_exec_1", "u_public_1"):
            with acting_as(acl, user, "C1"):
                await rag.query(q["text"], [q["target_name"]])
        return answers, c3_calls, len(rag.llm_calls)
    answers, c3_calls, total = asyncio.run(go())
    assert c3_calls == 2, "exec and public must each generate once; same-set users reuse"
    assert answers["u_exec_1"] == answers["u_exec_2"]
    assert answers["u_public_1"] == answers["u_public_2"]
    assert answers["u_exec_1"] != answers["u_public_1"]
    assert total == c3_calls + 1, "C1 answers are shared across users, but not with C3"


def test_c4_requires_a_filter():
    _, acl = fresh()
    try:
        with acting_as(acl, "u_public_1", Mode.C4):
            pass
    except NotImplementedError:
        pass
    else:
        raise AssertionError("C4 without a description filter must refuse to run")

    async def strip(kind, key, data, scope, acl_):
        return {**data, "description": "[filtered]"}

    set_c4_description_filter(strip)
    try:
        async def go():
            rag, acl2 = fresh()
            row = next(r for r in ROWS if r["type"] == "a" and r["user_tier"] == "team_a")
            with acting_as(acl2, row["user_id"], "C4"):
                res = await rag.query(row["text"], [row["target_name"]])
            return res.raw_data["data"]["entities"]
        ents = asyncio.run(go())
        assert ents and all(e["description"] == "[filtered]" for e in ents)
    finally:
        set_c4_description_filter(None)


def test_unknown_doc_fails_closed():
    async def go():
        rag, acl = fresh()
        rag.text_chunks.rows["chunk-ghost"] = {"content": "x", "full_doc_id": "d-not-in-manifest"}
        return await acl.chunk_acls(["chunk-ghost"])
    assert asyncio.run(go()) == {"chunk-ghost": frozenset()}


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
