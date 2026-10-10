"""One command for a full corpus build:
facts -> plan -> docs (LLM) -> queries + gold (LLM) -> oracle splits -> validation.

  python -m corpus_gen.run --config configs/dev.yaml
  python -m corpus_gen.run --config configs/full.yaml --set entities.cross_tier_fraction=0.5

Every LLM response is cached (llm.cache_dir), so re-running is free and reproducible;
only docs/queries whose prompts changed are regenerated. Exit code 1 if any doc fails
generation or validation fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import load_config
from .facts import build_fact_table
from .llm import LLM, check_provider
from .oracle import write_oracle_splits
from .plan import build_plan
from .queries import build_queries, build_rows, summarize, write_queries
from .validate import validate
from .verbalize import verbalize_all, write_corpus, yaml_value


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a complete corpus (docs, queries, gold, oracle splits).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. entities.cross_tier_fraction=0.5")
    a = ap.parse_args()
    t0 = time.time()

    cfg = load_config(a.config, {k: yaml_value(v) for k, v in (s.split("=", 1) for s in a.set)})
    check_provider(cfg.llm)
    print(f"== {cfg.run_name} -> {cfg.out_dir}")
    ft = build_fact_table(cfg)
    plan = build_plan(cfg, ft)
    print(f"facts: {ft.stats['n_by_class']} canaries={ft.stats['n_canaries_by_kind']}")
    print(f"plan:  {plan.stats['docs_per_tier']} shortfalls={plan.stats['n_shortfalls']}")

    llm = LLM(cfg.llm)
    results = verbalize_all(cfg, ft, plan, llm)
    report = write_corpus(cfg, ft, plan, results)
    if report["failed"]:
        print(f"doc generation failed for {report['failed']}; see gen_failures.json", file=sys.stderr)
        sys.exit(1)

    rows = build_rows(cfg, ft, build_queries(cfg, ft), llm)
    qsum = summarize(rows)
    write_queries(cfg, rows, qsum)
    write_oracle_splits(cfg.out_dir)

    val = validate(cfg.out_dir)
    (cfg.out_dir / "validation_report.json").write_text(json.dumps(val, indent=2))
    print(json.dumps({"docs_ok": report["n_docs_ok"], "query_rows": qsum["n_rows"],
                      "validation_ok": val["ok"], "n_errors": val["n_errors"],
                      "llm_cost_usd_this_run": round(llm.spent_usd, 2),
                      "minutes": round((time.time() - t0) / 60, 1)}, indent=2))
    for w in val["warnings"]:
        print("WARN ", w)
    sys.exit(0 if val["ok"] else 1)


if __name__ == "__main__":
    main()
