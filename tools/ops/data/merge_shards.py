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
    ap.add_argument("--dedupe", action="store_true",
                    help="Drop byte-identical records before shuffling. Timed "
                         "replays of one opening converge on a small set of "
                         "lines, so a high-replay campaign carries real "
                         "duplication; this measures and removes it.")
    a = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required (use .venv/bin/python)", file=sys.stderr)
        return 2

    # Local shards are shuffled per-shard (corpus.shuf.bin); CLOUD shards are
    # not (corpus.bin), because the global shuffle below subsumes it and
    # dropping it removes numpy from the cloud worker entirely. Either is valid
    # input — this pass reshuffles everything regardless.
    shards = []
    for d in sorted(a.campaign.glob("shard-*")):
        for name in ("corpus.shuf.bin", "corpus.bin"):
            if (d / name).is_file():
                shards.append(d / name)
                break
    if not shards:
        print(f"FATAL: no shard-*/corpus[.shuf].bin under {a.campaign}", file=sys.stderr)
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

    # --- optional pass 0: dedupe ------------------------------------------
    # BYTE-identical records only, which is exactly the right key. Two rows for
    # the SAME position with different results are not duplicates — that is the
    # WDL distribution and a value net needs it. What carries zero information
    # is a record identical in position, score AND result, which is what a
    # replayed game produces.
    #
    # Partition by HASH so identical records always land together; a random
    # scatter would spread them across buckets and see only a fraction of the
    # collisions. The shuffle passes below then run on the deduped output, so
    # hash-partitioning here does not bias the final order.
    if a.dedupe:
        dd = tmpdir / "dedupe"; dd.mkdir()
        DK = max(1, math.ceil(total / (a.bucket_mb * 1024 * 1024)))
        dparts = [dd / f"d{i:04d}" for i in range(DK)]
        dh = [p.open("wb") for p in dparts]
        try:
            for sfile in shards:
                with sfile.open("rb") as f:
                    while True:
                        blob = f.read(REC * 32768)
                        if not blob:
                            break
                        arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, REC)
                        # cheap stable hash over the whole record
                        hv = (arr.astype(np.uint64) *
                              np.arange(1, REC + 1, dtype=np.uint64)).sum(axis=1)
                        which = (hv % DK).astype(np.int64)
                        for b in range(DK):
                            sel = arr[which == b]
                            if sel.size:
                                dh[b].write(sel.tobytes())
        finally:
            for h_ in dh:
                h_.close()
        kept_parts = []
        n_before = nrec
        n_after = 0
        for i, p_ in enumerate(dparts):
            if not p_.stat().st_size:
                p_.unlink(); continue
            arr = np.fromfile(p_, dtype=np.uint8).reshape(-1, REC)
            uniq = np.unique(arr, axis=0)
            q = dd / f"u{i:04d}"
            uniq.tofile(q)
            kept_parts.append(q)
            n_after += uniq.shape[0]
            del arr, uniq
            p_.unlink()
        print(f"dedupe: {n_before:,} -> {n_after:,} records "
              f"({100*(n_before-n_after)/max(n_before,1):.2f}% duplicate)")
        shards = kept_parts
        total = n_after * REC
        nrec = n_after
        K = max(1, math.ceil(total / (a.bucket_mb * 1024 * 1024)))
        bucket_paths = [tmpdir / f"b{i:04d}" for i in range(K)]
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
                seed=a.seed, artifact=str(a.out), deduped=bool(a.dedupe))
    a.out.with_suffix(a.out.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"\n{a.out}  {written:,} records  sha256:{h.hexdigest()[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
