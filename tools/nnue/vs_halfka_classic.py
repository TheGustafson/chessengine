#!/usr/bin/env python3
"""Independent quantized CENN-v3 parity oracle over a FEN corpus."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import chess
import numpy as np

DIMS, ACC = 64 * 768, 256

def index(perspective: bool, piece: chess.Piece, square: int, king: int) -> int:
    # python-chess colors: WHITE=True, BLACK=False; use p=0 white / 1 black.
    p = 0 if perspective == chess.WHITE else 1
    flip = 56 * p
    opponent = int(piece.color != perspective)
    classic = (opponent * 6 + piece.piece_type - 1) * 64 + (square ^ flip)
    return (king ^ flip) * 768 + classic

def trunc_div(a: int, b: int) -> int:
    return int(a / b)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("quant", type=Path); ap.add_argument("cenn", type=Path)
    ap.add_argument("engine", type=Path); ap.add_argument("fens", type=Path)
    args = ap.parse_args()
    raw = args.quant.read_bytes()
    off = 0
    ft = np.frombuffer(raw, dtype="<i2", count=DIMS * ACC, offset=off).reshape(DIMS, ACC)
    off += 2 * DIMS * ACC
    bias = np.frombuffer(raw, dtype="<i2", count=ACC, offset=off).astype(np.int64); off += 2 * ACC
    out = np.frombuffer(raw, dtype="<i2", count=2 * ACC, offset=off).astype(np.int64); off += 4 * ACC
    out_bias = int(np.frombuffer(raw, dtype="<i2", count=1, offset=off)[0])
    fens = [x.strip() for x in args.fens.read_text().splitlines() if x.strip() and not x.startswith("#")]
    expected = []
    for fen in fens:
        board = chess.Board(fen)
        accumulators = []
        for perspective in (chess.WHITE, chess.BLACK):
            king = board.king(perspective)
            ids = [index(perspective, piece, square, king)
                   for square, piece in board.piece_map().items()]
            accumulators.append(bias + ft[ids].astype(np.int64).sum(axis=0))
        stm = 0 if board.turn == chess.WHITE else 1
        values = np.concatenate((accumulators[stm], accumulators[stm ^ 1]))
        act = np.clip(values, 0, 255)
        dot = int(np.dot(act * act, out))
        expected.append(trunc_div(trunc_div(dot, 255) + out_bias, 255 * 64) * 400)
        # Runtime multiplies before its final division, so correct the ordering.
        expected[-1] = trunc_div((trunc_div(dot, 255) + out_bias) * 400, 255 * 64)
    proc = subprocess.run([str(args.engine.resolve()), "nnue-eval", str(args.cenn.resolve())],
                          input="\n".join(fens) + "\n", text=True, capture_output=True)
    if proc.returncode: raise SystemExit(proc.stderr)
    got = [int(x) for x in proc.stdout.split()]
    diffs = [abs(a-b) for a,b in zip(expected, got)]
    print(f"positions={len(fens)} matches={sum(a==b for a,b in zip(expected,got))} max_abs={max(diffs, default=0)}")
    if expected != got: raise SystemExit("CENN-v3 parity failure")

if __name__ == "__main__": main()
