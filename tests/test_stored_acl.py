import networkx as nx

from acl_rag.stored_acl import chunk_table, locate_canaries, node_edge_acls, verify

DOC_USERS = {"d-pub": ["u_pub", "u_a"], "d-a": ["u_a"], "d-b": ["u_b"]}
CHUNK_MAP = {"d-pub": ["d-pub-chunk-000"], "d-a": ["d-a-chunk-000"], "d-b": ["d-b-chunk-000"]}
ORACLE = {"user_to_access_set": {"u_pub": "AS0"}, "access_sets": {}}


def _graph():
    g = nx.Graph()
    g.add_node("Ada", source_id="d-pub-chunk-000<SEP>d-a-chunk-000")      # cross-tier: union
    g.add_node("Bob", source_id="d-b-chunk-000")
    g.add_node("Ghost", source_id="d-zzz-chunk-000")                       # unknown doc: fails closed
    g.add_edge("Ada", "Bob", source_id="d-a-chunk-000")
    return g


def test_union_acl_stored_per_node_and_edge_and_fails_closed():
    chunks = chunk_table(CHUNK_MAP | {"d-zzz": ["d-zzz-chunk-000"]}, DOC_USERS)
    nodes, edges = node_edge_acls(_graph(), chunks)
    assert nodes["Ada"] == ["u_a", "u_pub"]
    assert nodes["Bob"] == ["u_b"]
    assert nodes["Ghost"] == []
    assert edges["Ada|Bob"] == ["u_a"]


def test_stored_acl_agrees_with_query_time_acl_and_detects_drift():
    g, chunks = _graph(), chunk_table(CHUNK_MAP | {"d-zzz": ["d-zzz-chunk-000"]}, DOC_USERS)
    nodes, edges = node_edge_acls(g, chunks)
    assert verify(g, chunks, nodes, edges, DOC_USERS, ORACLE) == []
    nodes["Bob"] = ["u_b", "u_pub"]                                         # a stale/leaky stored ACL
    assert verify(g, chunks, nodes, edges, DOC_USERS, ORACLE) == ["node Bob"]


def test_locate_canaries_finds_origin_chunk_and_flags_strays_and_lost():
    cans = [{"canary_id": "C1", "match_regex": r"AB-12", "origin_doc_id": "d-a"},
            {"canary_id": "C2", "match_regex": r"CD-34", "origin_doc_id": "d-a"},
            {"canary_id": "C3", "match_regex": r"EF-56", "origin_doc_id": "d-a"}]
    text = {"d-a-chunk-000": "code AB-12 and CD-34", "d-a-chunk-001": "tail AB-12 overlap",
            "d-b-chunk-000": "stray CD-34 here"}
    doc = {"d-a-chunk-000": "d-a", "d-a-chunk-001": "d-a", "d-b-chunk-000": "d-b"}
    r = locate_canaries(cans, text, doc)
    assert r["origin_chunks"]["C1"] == ["d-a-chunk-000", "d-a-chunk-001"]   # canary in the overlap: both chunks
    assert r["stray"] == {"C2": ["d-b-chunk-000"]}
    assert r["lost"] == ["C3"]
