"""Read and pack per-row result files (<config>.jsonl or <config>.jsonl.gz).

Runs append to plain ``C?.jsonl`` (resumable). Before committing, pack them:

    python -m acl_rag.resultio results/full_cf0.3_s42

which writes ``C?.jsonl.gz`` next to each raw file (byte-identical gzip for the
same input) and ``MANIFEST.json`` with row counts and sha256 of the raw and
compressed bytes. Raw ``C?.jsonl`` is git-ignored; the ``.gz`` files are
committed. Scorers read either form; raw wins when both exist.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

GZ_WARN_BYTES = 50 * 1024 * 1024    # GitHub warns above 50 MB
GZ_LIMIT_BYTES = 95 * 1024 * 1024   # GitHub rejects above 100 MB


def config_name(path: Path) -> str:
    """'C1' for C1.jsonl and C1.jsonl.gz."""
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def result_files(rdir: Path, config: str | None = None) -> list[Path]:
    """One file per config, raw .jsonl preferred over .jsonl.gz."""
    names = [config.upper()] if config else sorted(
        {config_name(p) for p in rdir.glob("C?.jsonl")} | {config_name(p) for p in rdir.glob("C?.jsonl.gz")})
    out = []
    for n in names:
        raw, gz = rdir / f"{n}.jsonl", rdir / f"{n}.jsonl.gz"
        if raw.exists():
            out.append(raw)
        elif gz.exists():
            out.append(gz)
        else:
            raise FileNotFoundError(f"neither {raw} nor {gz} exists")
    return out


def iter_rows(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_rows(path: Path) -> list[dict]:
    return list(iter_rows(path))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack(rdir: Path) -> dict:
    manifest_path = rdir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for raw in sorted(rdir.glob("C?.jsonl")):
        gz = raw.with_name(raw.name + ".gz")
        # filename="" and mtime=0 keep the gzip bytes deterministic, so an
        # unchanged run does not show up as a modified binary in git.
        with open(raw, "rb") as src, open(gz, "wb") as dst_raw, \
                gzip.GzipFile(filename="", mode="wb", fileobj=dst_raw, compresslevel=9, mtime=0) as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                dst.write(block)
        n_raw = sum(1 for _ in iter_rows(raw))
        n_gz = sum(1 for _ in iter_rows(gz))
        if n_raw != n_gz:
            raise RuntimeError(f"{gz}: {n_gz} rows after packing, expected {n_raw}")
        size = gz.stat().st_size
        if size > GZ_LIMIT_BYTES:
            raise RuntimeError(f"{gz} is {size / 2**20:.1f} MB, over GitHub's 100 MB limit")
        if size > GZ_WARN_BYTES:
            print(f"WARNING {gz.name}: {size / 2**20:.1f} MB (GitHub warns above 50 MB)")
        run_ids = {r.get("run_id") for r in iter_rows(gz)}
        manifest[config_name(raw)] = {
            "file": gz.name, "rows": n_raw,
            "run_ids": sorted(x for x in run_ids if x),
            "raw_bytes": raw.stat().st_size, "gz_bytes": size,
            "sha256_raw": _sha256(raw), "sha256_gz": _sha256(gz),
        }
        print(f"{raw.name}: {n_raw} rows, {raw.stat().st_size / 2**20:.1f} MB -> {size / 2**20:.1f} MB")
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dirs", nargs="+", type=Path, help="e.g. results/full_cf0.3_s42")
    args = ap.parse_args()
    for d in args.results_dirs:
        pack(d)


if __name__ == "__main__":
    main()
