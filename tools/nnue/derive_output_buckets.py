#!/usr/bin/env python3
"""Derive ARCHON's output-bucket layout from ARCHON's own training data.

2026-08 originality audit, item 1 (docs/audit/2026-08-originality/
00-synthesis.md §2): the piece-count -> output-bucket map the SOTA head ships
was copied verbatim from Reckless (AGPLv3). This tool replaces the hand-made
boundaries with a DERIVED partition: equal-mass buckets over the piece-count
distribution of our own corpus.

Derivation rule (deterministic, no tuning):

    Let h[c] be the number of sampled positions with popcount(occupancy) = c,
    F[c] the cumulative sum, and T the sample total. With B buckets, bucket
    ceiling c_k (k = 1..B, inclusive upper bound in piece count) is

        c_k = min{ c : F[c] >= k * T / B },        c_B forced to 32.

    A position with piece count c lands in the first bucket whose ceiling is
    >= c. Every bucket then holds as close to T/B positions as a contiguous
    piece-count partition allows.

Input format: 32-byte bulletformat ChessBoard records (the D46 pipeline
format; layout pinned in jax/archon_jax/spec.py RECORD_DTYPE and
docs/research/data-recipe-contract.md). Only the first field is read:
bytes 0..8 = occupancy bitboard (little-endian u64); the piece count is its
popcount. The intended input is a GLOBALLY SHUFFLED corpus, so a prefix
sample is a uniform sample.

Memory: streamed in fixed chunks; RSS stays under ~300 MB regardless of
corpus size.

Usage:
    python3 tools/nnue/derive_output_buckets.py \
        data/nnue/jaxcorpora/fishpack32-sf5k.shuf.bin \
        [--records 100000000] [--buckets 8] [--chunk 2000000]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

RECORD_BYTES = 32
PIECE_COUNT_BINS = 33  # popcount(occ) in 0..=32


def stream_histogram(path: Path, max_records: int, chunk_records: int) -> np.ndarray:
    """Piece-count histogram over the first `max_records` records of `path`."""
    hist = np.zeros(PIECE_COUNT_BINS, dtype=np.int64)
    remaining = max_records
    with open(path, "rb") as f:
        while remaining > 0:
            want = min(remaining, chunk_records)
            buf = f.read(want * RECORD_BYTES)
            if not buf:
                break
            n = len(buf) // RECORD_BYTES
            if n == 0:
                break
            raw = np.frombuffer(buf, dtype="<u8", count=n * 4).reshape(n, 4)
            occ = np.ascontiguousarray(raw[:, 0])
            counts = np.unpackbits(occ.view(np.uint8).reshape(n, 8), axis=1).sum(
                axis=1, dtype=np.int64
            )
            hist += np.bincount(counts, minlength=PIECE_COUNT_BINS)[:PIECE_COUNT_BINS]
            remaining -= n
    return hist


def equal_mass_ceilings(hist: np.ndarray, buckets: int) -> list[int]:
    total = int(hist.sum())
    cum = np.cumsum(hist)
    ceilings: list[int] = []
    for k in range(1, buckets + 1):
        target = k * total / buckets
        c = int(np.searchsorted(cum, target))  # min c with cum[c] >= target
        ceilings.append(min(c, PIECE_COUNT_BINS - 1))
    ceilings[-1] = PIECE_COUNT_BINS - 1  # the last bucket always reaches 32
    for a, b in zip(ceilings, ceilings[1:]):
        if b <= a:
            raise SystemExit(
                f"degenerate partition {ceilings}: bucket ceiling {b} <= {a}; "
                "the distribution is too concentrated for this bucket count"
            )
    return ceilings


def layout_from_ceilings(ceilings: list[int]) -> list[int]:
    layout = []
    for count in range(PIECE_COUNT_BINS):
        for b, c in enumerate(ceilings):
            if count <= c:
                layout.append(b)
                break
    return layout


LEGACY_CEILINGS = [8, 12, 16, 19, 22, 25, 28, 32]  # the pre-2026-08 map


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--records", type=int, default=100_000_000)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    args = ap.parse_args()

    size = args.corpus.stat().st_size
    with open(args.corpus, "rb") as f:
        head_sha = hashlib.sha256(f.read(1 << 20)).hexdigest()

    hist = stream_histogram(args.corpus, args.records, args.chunk)
    total = int(hist.sum())

    print(f"corpus        {args.corpus}")
    print(f"file size     {size:,} B ({size // RECORD_BYTES:,} records)")
    print(f"first-MiB sha {head_sha}")
    print(f"sampled       {total:,} records (prefix of a globally shuffled file)")
    print()
    print("piece-count histogram:")
    cum = 0
    for c in range(PIECE_COUNT_BINS):
        if hist[c] == 0:
            continue
        cum += int(hist[c])
        print(
            f"  {c:2d}  {int(hist[c]):>12,}  {hist[c] / total:7.4%}  "
            f"cum {cum / total:8.4%}"
        )

    ceilings = equal_mass_ceilings(hist, args.buckets)
    layout = layout_from_ceilings(ceilings)
    legacy_layout = layout_from_ceilings(LEGACY_CEILINGS)

    print()
    print(f"equal-mass ceilings (inclusive): {ceilings}")
    print("per-bucket mass:")
    lo = 0
    for b, c in enumerate(ceilings):
        mass = int(hist[lo : c + 1].sum())
        print(f"  bucket {b}: counts {lo:2d}..={c:2d}  {mass:>12,}  {mass / total:7.4%}")
        lo = c + 1

    print()
    if ceilings == LEGACY_CEILINGS:
        print(
            "NOTE: the derived ceilings coincide with the legacy (Reckless) "
            "boundaries — record the derivation, but the optics of the "
            "coincidence must be flagged wherever the table ships."
        )
    else:
        same = sum(a == b for a, b in zip(layout, legacy_layout))
        print(f"derived layout agrees with legacy on {same}/33 piece counts")

    print()
    print("Rust (sota_head.rs ceilings):")
    print(f"    [{', '.join(str(c) for c in ceilings)}]")
    print("Rust (expanded [u8; 33]):")
    print(f"    {layout}")
    print("Python (jax/archon_jax/spec.py):")
    print(f"    ARCHON_BUCKET_CEILINGS = {tuple(ceilings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
