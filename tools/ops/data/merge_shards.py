#!/usr/bin/env python3
"""Merge a campaign's shard corpora into ONE globally shuffled .shuf.bin.

WHY A REAL SHUFFLE AND NOT `cat`
    Each shard is internally shuffled, but a concatenation is not: records from
    shard 0 all precede records from shard 1, and a shard is ~1.5 hours of
    consecutive games from one contiguous slice of the opening book. Any reader
    that walks the file in order — or reads it in large sequential blocks —
    would see long runs of correlated positions from the same openings, which
    is exactly the correlation the per-game sampling cap exists to avoid.

    Two-pass external shuffle: scatter every record into one of K random
    buckets, then shuffle each bucket in memory and concatenate. Each bucket is
    sized to fit the memory budget, so this handles corpora far larger than RAM
    on a box that has been OOM-killed repeatedly.

WHY BUCKET COUNT IS DERIVED, NOT FIXED
    K = ceil(total_bytes / BUCKET_MB). Too few buckets and pass 2 OOMs; too
    many and pass 1 thrashes on file handles. Deriving it from the actual byte
    count means the tool scales from a 3 MB smoke run to a 30 GB campaign with
    no knob to get wrong.

USAGE
    python3 tools/ops/data/merge_shards.py <campaign-dir> --out merged.shuf.bin
    ... --bucket-mb 512     memory budget per pass-2 bucket (default 512)
    ... --seed 12345        reproducible shuffle (default 0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import tempfile

REC = 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--bucket-mb", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required (use .venv/bin/python)", file=sys.stderr)
        return 2

    shards = sorted(a.campaign.glob("shard-*/corpus.shuf.bin"))
    if not shards:
        print(f"FATAL: no shard-*/corpus.shuf.bin under {a.campaign}", file=sys.stderr)
        return 2

    total = 0
    for s in shards:
        n = s.stat().st_size
        if n % REC:
            print(f"FATAL: {s} is {n} B, not a multiple of {REC}", file=sys.stderr)
            return 3
        total += n
    nrec = total // REC
    K = max(1, math.ceil(total / (a.bucket_mb * 1024 * 1024)))
    print(f"{len(shards)} shards, {nrec:,} records, {total/1e9:.2f} GB -> {K} buckets")

    rng = np.random.default_rng(a.seed)
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="merge_", dir=a.out.parent))
    bucket_paths = [tmpdir / f"b{i:04d}" for i in range(K)]

    # --- pass 1: scatter ----------------------------------------------------
    handles = [p.open("wb") for p in bucket_paths]
    try:
        for s in shards:
            with s.open("rb") as f:
                while True:
                    # 1 MB of records at a time keeps peak memory flat
                    blob = f.read(REC * 32768)
                    if not blob:
                        break
                    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, REC)
                    which = rng.integers(0, K, size=arr.shape[0])
                    for b in range(K):
                        sel = arr[which == b]
                        if sel.size:
                            handles[b].write(sel.tobytes())
            print(f"  scattered {s.parent.name}")
    finally:
        for h in handles:
            h.close()

    # --- pass 2: shuffle each bucket, concatenate ---------------------------
    h = hashlib.sha256()
    written = 0
    with a.out.open("wb") as out:
        for i, p in enumerate(bucket_paths):
            n = p.stat().st_size
            if not n:
                p.unlink()
                continue
            arr = np.fromfile(p, dtype=np.uint8).reshape(-1, REC)
            rng.shuffle(arr, axis=0)
            blob = arr.tobytes()
            out.write(blob)
            h.update(blob)
            written += arr.shape[0]
            del arr, blob
            p.unlink()
            if (i + 1) % 8 == 0 or i == K - 1:
                print(f"  merged bucket {i+1}/{K} ({written:,} records)")
    tmpdir.rmdir()

    if written != nrec:
        print(f"FATAL: wrote {written} records, expected {nrec}", file=sys.stderr)
        return 4

    meta = dict(records=written, bytes=written * REC, sha256=h.hexdigest(),
                shards=[str(s.parent.name) for s in shards], seed=a.seed,
                artifact=str(a.out))
    a.out.with_suffix(a.out.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"\n{a.out}  {written:,} records  sha256:{h.hexdigest()[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
