import json

import pytest

from acl_rag.resultio import config_name, pack, read_rows, result_files


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_pack_roundtrip_and_manifest(tmp_path):
    rows = [{"qid": f"q{i}", "run_id": "r1", "answer": "x" * i} for i in range(50)]
    _write(tmp_path / "C1.jsonl", rows)
    m = pack(tmp_path)
    gz = tmp_path / "C1.jsonl.gz"
    assert read_rows(gz) == rows
    assert m["C1"]["rows"] == 50 and m["C1"]["run_ids"] == ["r1"]
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["C1"]["sha256_gz"] == m["C1"]["sha256_gz"]


def test_pack_is_deterministic(tmp_path):
    _write(tmp_path / "C2.jsonl", [{"qid": "a", "run_id": "r"}])
    first = pack(tmp_path)["C2"]["sha256_gz"]
    assert pack(tmp_path)["C2"]["sha256_gz"] == first


def test_result_files_prefers_raw_then_gz(tmp_path):
    _write(tmp_path / "C1.jsonl", [{"qid": "a"}])
    pack(tmp_path)
    (tmp_path / "C1.jsonl").unlink()
    _write(tmp_path / "C2.jsonl", [{"qid": "b"}])
    pack(tmp_path)
    files = result_files(tmp_path)
    assert [f.name for f in files] == ["C1.jsonl.gz", "C2.jsonl"]
    assert [config_name(f) for f in files] == ["C1", "C2"]
    with pytest.raises(FileNotFoundError):
        result_files(tmp_path, "C3")
