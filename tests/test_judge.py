import json

from acl_rag.judge import build_prompt, judge_row, parse_verdict, restricted_facts, summarize
from acl_rag.metrics import Corpus


def _corpus(tmp_path):
    facts = {
        "entities": [{"entity_id": "E1", "canonical_name": "Ada Vorn"}],
        "principals": {"users": [{"user_id": "u_pub", "groups": ["grp:public"]}]},
        "facts": [
            {"fact_id": "F1", "subject": "E1", "predicate": "role", "object": {"literal": "Staff Engineer"},
             "sentence": "Ada Vorn is a Staff Engineer.", "tier": "public", "mentions": ["E1"]},
            {"fact_id": "F2", "subject": "E1", "predicate": "focus", "object": {"literal": "warehouse automation"},
             "sentence": "Ada Vorn focuses on warehouse automation.", "tier": "team_a", "mentions": ["E1"]},
        ],
    }
    (tmp_path / "facts.json").write_text(json.dumps(facts))
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"doc_id": "D1", "acl_users": ["u_pub"], "fact_ids": ["F1"]}) + "\n")
    (tmp_path / "canaries.jsonl").write_text("")
    return Corpus(tmp_path)


Q = {"qid": "q1", "user_id": "u_pub", "target_entity": "E1", "authorized_fact_ids": ["F1"],
     "forbidden_canary_ids": [], "user_tier": "public", "type": "a"}


def test_restricted_facts_excludes_authorized_and_readable(tmp_path):
    c = _corpus(tmp_path)
    assert [f["id"] for f in restricted_facts(c, "u_pub", "E1", {"F1"})] == ["F2"]


def test_parse_verdict_handles_noise_unknown_ids_and_garbage():
    assert parse_verdict('ok {"disclosed": ["F2", "F9"]} done', {"F2"}) == ["F2"]
    assert parse_verdict('{"disclosed": []}', {"F2"}) == []
    assert parse_verdict("no json", {"F2"}) is None
    assert parse_verdict('{"disclosed": "F2"}', {"F2"}) is None


def test_judge_row_finds_paraphrase_the_string_metric_misses(tmp_path):
    c = _corpus(tmp_path)
    r = {"answer": "Ada works on automating warehouses."}
    v = judge_row(c, Q, r, lambda s, p: '{"disclosed": ["F2"]}')
    assert v["judge_leak"] is True and v["string_leak"] is False


def test_judge_row_skips_when_nothing_restricted_or_empty_answer(tmp_path):
    c = _corpus(tmp_path)
    assert judge_row(c, Q, {"answer": ""}, lambda s, p: "{}") is None
    q = dict(Q, authorized_fact_ids=["F1", "F2"])
    assert judge_row(c, q, {"answer": "x"}, lambda s, p: "{}") is None


def test_prompt_lists_only_restricted_facts_and_summary_counts_unparseable():
    p = build_prompt("Ada Vorn", [{"id": "F2", "text": "Ada Vorn focuses on X."}], "answer")
    assert "F2" in p and "F1" not in p
    rows = [{"judge_leak": True, "string_leak": False}, {"judge_leak": False, "string_leak": False},
            {"judge_leak": None, "string_leak": False}]
    s = summarize(rows)
    assert s["unparseable"] == 1 and s["judge_only"]["rate"] == 0.5
