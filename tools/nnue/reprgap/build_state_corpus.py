"""Representation-gap study: full-state training corpus builder.

TSV (fast_fens, full FENs) -> bulletformat .bin whose extra[2] byte carries
the state the pipeline otherwise drops:

  bit0 own-castle-K   bit1 own-castle-Q   bit2 opp-castle-K   bit3 opp-castle-Q
  bit4 ep-available   bits5-7 rule50 bucket = min(rule50, 63) >> 3

The byte is 0 in every legacy corpus, so a net trained with the state-atom
seam stays bit-parity on legacy data at sb0 (zero-init rows x silent atoms).
Input TSV must already be SHUFFLED (shuf the source first).

  .venv/bin/python tools/nnue/reprgap/build_state_corpus.py <in.tsv> <out.bin> [procs]
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "jax"))
sys.path.insert(0, str(ROOT / "tools/nnue/reprgap"))


def _chunk(lines: list[str]) -> bytes:
    import chess

    from archon_jax.spec import RECORD_DTYPE
    from encode import board_to_record
    recs = np.zeros(len(lines), dtype=RECORD_DTYPE)
    k = 0
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        fen, res, _ply, s5k = parts[:4]
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        rec, st = board_to_record(board)
        res_w, s_w = float(res), float(s5k)
        if board.turn == chess.BLACK:
            res_w, s_w = 1.0 - res_w, -s_w
        rec["score"] = np.int16(int(max(-2499.0, min(2499.0, s_w / 1.9))))
        rec["result"] = np.uint8(int(res_w * 2))
        byte = (st["castle_own_k"] | (st["castle_own_q"] << 1)
                | (st["castle_opp_k"] << 2) | (st["castle_opp_q"] << 3)
                | (int(st["ep_file"] >= 0) << 4)
                | ((min(st["rule50"], 63) >> 3) << 5))
        rec["extra"][0][2] = byte
        recs[k] = rec[0]
        k += 1
    return recs[:k].tobytes()


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    CH = 20000
    total = 0
    with open(src) as fh, open(dst, "wb") as out, Pool(procs) as pool:
        def chunks():
            buf = []
            for line in fh:
                buf.append(line)
                if len(buf) == CH:
                    yield buf
                    buf = []
            if buf:
                yield buf
        for blob in pool.imap(_chunk, chunks()):
            out.write(blob)
            total += len(blob) // 32
            if total % 1_000_000 < CH:
                print(f"  {total} records", flush=True)
    print(f"wrote {total} records -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
