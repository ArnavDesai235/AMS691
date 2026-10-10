"""Per-access-set document splits for the oracle baseline (one LightRAG index per access set).

An access set is a distinct set of groups; users with identical permissions share one
index. A doc is readable by an access set iff acl_groups ∩ access_set_groups != ∅.
Also emits the unfiltered "all" split (the shared index used by C1–C4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .verbalize import yaml_value


def build_oracle_splits(run_dir: Path) -> dict[str, Any]:
    facts = json.loads((run_dir / "facts.json").read_text())
    manifest = [json.loads(line) for line in (run_dir / "manifest.jsonl").open()]
    pr = facts["principals"]
    users_by_as: dict[str, list[str]] = {}
    for u in pr["users"]:
        users_by_as.setdefault(u["access_set"], []).append(u["user_id"])
    splits = {}
    for as_id, groups in pr["access_sets"].items():
        gs = set(groups)
        docs = sorted(m["doc_id"] for m in manifest if gs & set(m["acl_groups"]))
        splits[as_id] = {"groups": groups, "users": sorted(users_by_as.get(as_id, [])),
                         "doc_ids": docs, "n_docs": len(docs)}
    return {
        "access_sets": splits,
        "user_to_access_set": {u["user_id"]: u["access_set"] for u in pr["users"]},
        "all": {"doc_ids": sorted(m["doc_id"] for m in manifest), "n_docs": len(manifest)},
    }


def write_oracle_splits(run_dir: Path) -> dict[str, Any]:
    splits = build_oracle_splits(run_dir)
    (run_dir / "oracle_splits.json").write_text(json.dumps(splits, indent=2))
    return splits


def main() -> None:
    ap = argparse.ArgumentParser(description="Write oracle_splits.json for a generated corpus.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, {k: yaml_value(v) for k, v in (s.split("=", 1) for s in a.set)})
    s = write_oracle_splits(cfg.out_dir)
    print({k: (v["groups"], v["n_docs"]) for k, v in s["access_sets"].items()}, "all:", s["all"]["n_docs"])


if __name__ == "__main__":
    main()
