#!/usr/bin/env python3
"""Global external shuffle for 32-byte bulletformat records (in.bin -> out.shuf.bin).

data.py's loader contract (jax-workstream-plan queue B): training reads
contiguous windows at shuffled offsets over a PRE-SHUFFLED corpus, so the
corpus itself must be a uniform record-level permutation. The D40_CONVERT dump
is only window-shuffled (the loader's 2M-row buffer), which leaves epoch-scale
ordering intact — this pass removes it.

Two-pass external shuffle (RAM ~= filesize/buckets + IO buffers):
  pass 1: stream input, scatter each record to one of K bucket files via a
          seeded PCG64 (bucket choice i.i.d. uniform);
  pass 2: load each bucket, permute in RAM, append to the output.
A multiset checksum (per-column u64 sums over the 4 words of each record,
permutation-invariant) is computed on both passes and must match, and the
byte size must match, or the output is deleted and the run fails.

  python3 tools/nnue/shuffle_records.py in.bin out.shuf.bin [--seed N] [--buckets K]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

RECORD = 32  # bytes; bulletformat ChessBoard
WORDS = RECORD // 8
CHUNK_RECORDS = 8 << 20  # 8M records = 256 MiB per read


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed", type=int, default=0x00D40FEA)
    ap.add_argument("--buckets", type=int, default=64)
    args = ap.parse_args()

    size = args.input.stat().st_size
    if size % RECORD:
        sys.exit(f"input size {size} is not a multiple of {RECORD}")
    total = size // RECORD
    rng = np.random.default_rng(args.seed)
    print(f"{total:,} records, {args.buckets} buckets, seed {args.seed:#x}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    in_sums = np.zeros(WORDS, dtype=np.uint64)
    out_sums = np.zeros(WORDS, dtype=np.uint64)

    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".shufbkt-") as tmp:
        buckets = [open(os.path.join(tmp, f"b{i:03d}"), "wb") for i in range(args.buckets)]
        try:
            with args.input.open("rb") as src:
                done = 0
                while True:
                    raw = src.read(CHUNK_RECORDS * RECORD)
                    if not raw:
                        break
                    rows = np.frombuffer(raw, dtype=np.uint64).reshape(-1, WORDS)
                    in_sums += rows.sum(axis=0, dtype=np.uint64)
                    ids = rng.integers(0, args.buckets, size=len(rows))
                    order = np.argsort(ids, kind="stable")
                    sorted_rows, sorted_ids = rows[order], ids[order]
                    starts = np.searchsorted(sorted_ids, np.arange(args.buckets))
                    ends = np.append(starts[1:], len(sorted_rows))
                    for b, (s, e) in enumerate(zip(starts, ends)):
                        if e > s:
                            buckets[b].write(sorted_rows[s:e].tobytes())
                    done += len(rows)
                    print(f"scatter: {done:,}/{total:,}", flush=True)
        finally:
            for handle in buckets:
                handle.close()

        with args.output.open("wb") as dst:
            for i in range(args.buckets):
                data = np.fromfile(os.path.join(tmp, f"b{i:03d}"), dtype=np.uint64)
                rows = data.reshape(-1, WORDS)
                out_sums += rows.sum(axis=0, dtype=np.uint64)
                dst.write(rows[rng.permutation(len(rows))].tobytes())
                print(f"permute: bucket {i + 1}/{args.buckets} ({len(rows):,} rows)", flush=True)

    out_size = args.output.stat().st_size
    if out_size != size or not np.array_equal(in_sums, out_sums):
        args.output.unlink(missing_ok=True)
        sys.exit(f"VERIFY FAILED: size {out_size} vs {size}, checksum match "
                 f"{np.array_equal(in_sums, out_sums)} — output deleted")
    print(f"OK: {total:,} records -> {args.output} (multiset checksum + size verified)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
