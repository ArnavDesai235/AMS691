"""sweep_retrieval: grid construction, scoring and keyword handling (no LightRAG run)."""
from argparse import Namespace
from pathlib import Path

from acl_rag import sweep_retrieval as sw
from acl_rag.metrics import Corpus

ROOT = Path(__file__).resolve().parents[1]


def grid(**kw):
    args = Namespace(top_k=[10, 40], budgets=["6000/8000", "3000/3000"], chunk_top_k=[20], keywords=["orig"])
    for k, v in kw.items():
        setattr(args, k, v)
    return sw.settings(args, {"top_k": 40, "chunk_top_k": 20})


def test_grid_marks_one_baseline():
    g = grid()
    assert len(g) == 4 and sum(s["baseline"] for s in g) == 1
    base = next(s for s in g if s["baseline"])
    assert sw.label(base) == "k40_e6000_r8000_c20_orig"


def test_baseline_added_when_missing():
    g = grid(top_k=[10])
    assert g[0]["baseline"] and sw.label(g[0]) == "k40_e6000_r8000_c20_orig" and len(g) == 3


def test_ll_keywords_replace_high_level():
    kw = {"high_level": ["What do we know"], "low_level": ["Tanla Terlusvi"]}
    s = {**grid()[0], "keywords": "ll"}
    p = sw.query_param({}, s, kw)
    assert p.hl_keywords == ["Tanla Terlusvi"] and p.ll_keywords == ["Tanla Terlusvi"]
    assert sw.query_param({}, {**s, "keywords": "orig"}, kw).hl_keywords == ["What do we know"]


def test_score_matches_metrics_definition():
    corpus = Corpus(ROOT / "corpus" / "dev_cf0.3_s42")
    row = next(r for r in sw.load_rows(ROOT / "corpus" / "dev_cf0.3_s42") if r["authorized_fact_ids"])
    rel = sorted({d for f in row["authorized_fact_ids"] for d in corpus.fact_docs.get(f, ())
                  if row["user_id"] in corpus.doc_users.get(d, ())})
    assert sw.score(corpus, row, rel)["recall@all"] == 1.0
    assert sw.score(corpus, row, ["x"] * 3 + rel)["recall@1"] == 0.0
    assert sw.score(corpus, row, [])["recall@10"] == 0.0


def test_ranked_docs_first_occurrence():
    chunks = [{"file_path": "a"}, {"file_path": "b"}, {"file_path": "a"}, {"file_path": None}]
    assert sw._ranked_docs(chunks, "file_path") == ["a", "b"]


def test_split_is_by_query_and_balanced():
    rows = sw.load_rows(ROOT / "corpus" / "full_cf0.3_s42")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["query_id"], set()).add(sw.split_of(r["query_id"]))
    assert all(len(v) == 1 for v in by_q.values())
    tune = sum(sw.split_of(q) == "tune" for q in by_q)
    assert 0.4 < tune / len(by_q) < 0.6
