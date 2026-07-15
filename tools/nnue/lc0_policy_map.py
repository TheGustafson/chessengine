#!/usr/bin/env python3
"""Canonical Lc0 1858-policy-index mapping.

Lc0's action space is not something to infer statistically from observed legal
move sets.  Indices 0..1791 are, for each source square in a1..h8 order, every
queen-like or knight destination in destination-square order.  The remaining
66 indices are explicit queen/rook/bishop promotions; knight promotions use the
ordinary from-to index.

This reproduces ``kMoveStrs`` in Lc0 ``src/neural/encoder.cc``.  The campaign's
old inferred JSON omitted 87 indices (all explicit promotions among them) and
contained one wrong duplicate castling mapping, causing the corpus builder to
discard roughly 16% of otherwise usable positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FILES = "abcdefgh"
RANKS = "12345678"


def square_name(square: int) -> str:
    return FILES[square % 8] + RANKS[square // 8]


def queen_or_knight(from_square: int, to_square: int) -> bool:
    if from_square == to_square:
        return False
    ff, fr = from_square % 8, from_square // 8
    tf, tr = to_square % 8, to_square // 8
    df, dr = abs(tf - ff), abs(tr - fr)
    return df == 0 or dr == 0 or df == dr or (df, dr) in {(1, 2), (2, 1)}


def canonical_moves() -> list[str]:
    moves = [
        square_name(src) + square_name(dst)
        for src in range(64)
        for dst in range(64)
        if queen_or_knight(src, dst)
    ]
    assert len(moves) == 1792

    for src_file in range(8):
        src = 6 * 8 + src_file
        for dst_file in range(max(0, src_file - 1), min(7, src_file + 1) + 1):
            dst = 7 * 8 + dst_file
            for promotion in "qrb":
                moves.append(square_name(src) + square_name(dst) + promotion)

    assert len(moves) == 1858
    assert len(set(moves)) == 1858
    return moves


def from_to_index(move: str) -> int:
    """Collapse promotion identity for the current 64x64 policy heads."""
    src = FILES.index(move[0]) + 8 * RANKS.index(move[1])
    dst = FILES.index(move[2]) + 8 * RANKS.index(move[3])
    return src * 64 + dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, help="write the canonical index->UCI JSON")
    parser.add_argument("--check", type=Path, help="compare an existing JSON with canonical")
    args = parser.parse_args()

    expected = {str(i): move for i, move in enumerate(canonical_moves())}
    if args.check:
        actual = json.loads(args.check.read_text())
        missing = sorted(set(expected) - set(actual), key=int)
        wrong = sorted(
            (int(i), actual[i], expected[i])
            for i in set(actual) & set(expected)
            if actual[i] != expected[i]
        )
        print(f"entries={len(actual)} missing={len(missing)} wrong={len(wrong)}")
        if missing:
            print("missing indices:", " ".join(missing))
        for index, got, want in wrong:
            print(f"wrong {index}: {got} != {want}")
        if missing or wrong or len(actual) != len(expected):
            raise SystemExit(1)

    if args.write:
        args.write.write_text(json.dumps(expected, indent=2) + "\n")

    if not args.check and not args.write:
        print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
