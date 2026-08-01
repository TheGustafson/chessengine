"""Representation-gap study: full-state FEN corpus -> canonical records + state.

Reads TSV rows (``FEN\\tresult\\tply\\tscore5k``, White-relative — the
fast_fens format of data/nnue/binpacks/uho_frs*.tsv), canonicalizes each
position to the side-to-move frame exactly like the bulletformat pipeline
(black to move: vertical mirror + color swap + label sign flip), and writes

  recs   (N,) RECORD_DTYPE      what the network sees (castling/ep DROPPED,
                                 rule50 DROPPED — bulletformat has no field)
  state  sidecar arrays          the state the pipeline drops, stm frame:
         castle_own_k/q, castle_opp_k/q (u8), ep_file (i8, -1 = none),
         rule50 (u8), ply (u16), score_ctl (i16, score5k/1.9 control units,
         stm-rel), result2 (u8, 0/1/2 stm-rel), fenidx (i64 row in sample)

VALIDITY GATES:
  roundtrip   decode 20k real heldout records to boards, re-encode, require
              BYTE-IDENTICAL records (pins occ/nibble/ksq/opp_ksq conventions
              against the real pipeline's output)
  flip        my white-frame record of the mirrored position must equal my
              black-to-move record of the original (canonicalization is an
              involution); engine cross-check happens in check_engine.sh
              (jax forward vs `chessengine nnue-eval` on mixed-stm FENs).

Run (.venv has python-chess; features.py/spec.py are numpy-pure):
  .venv/bin/python tools/nnue/reprgap/encode.py roundtrip
  .venv/bin/python tools/nnue/reprgap/encode.py tsv <sample.tsv> <out.npz>
  .venv/bin/python tools/nnue/reprgap/encode.py fens <fens.txt> <out.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "jax"))

import chess  # noqa: E402

from archon_jax.spec import RECORD_DTYPE  # noqa: E402

PIECE_CODE = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
              chess.QUEEN: 4, chess.KING: 5}


def board_to_record(board: chess.Board) -> tuple[np.ndarray, dict]:
    """stm-canonical record + state dict. Handles both sides to move."""
    flip = board.turn == chess.BLACK
    rec = np.zeros(1, dtype=RECORD_DTYPE)
    pieces = []
    for sq, pc in board.piece_map().items():
        csq = sq ^ 56 if flip else sq
        # color 0 = side to move ("white" in canonical frame)
        color = 0 if (pc.color == board.turn) else 1
        pieces.append((csq, color, PIECE_CODE[pc.piece_type]))
    occ = 0
    ksq = opp_ksq = 0
    for i, (sq, color, pt) in enumerate(sorted(pieces)):
        occ |= 1 << sq
        rec["pcs"][0][i // 2] |= np.uint8(((color << 3) | pt) << (4 * (i % 2)))
        if pt == 5:
            if color == 0:
                ksq = sq
            else:
                opp_ksq = sq ^ 56  # enemy king stored in ITS OWN frame
    rec["occ"] = occ
    rec["ksq"] = ksq
    rec["opp_ksq"] = opp_ksq
    rec["extra"][0] = (64, 64, 0)

    def has(color: bool, kingside: bool) -> int:
        rights = board.castling_rights
        mask = (chess.BB_H1 if kingside else chess.BB_A1) if color == chess.WHITE \
            else (chess.BB_H8 if kingside else chess.BB_A8)
        return int(bool(rights & mask))

    us, them = board.turn, not board.turn
    ep = -1
    if board.ep_square is not None:
        ep = chess.square_file(board.ep_square)
    state = {
        "castle_own_k": has(us, True), "castle_own_q": has(us, False),
        "castle_opp_k": has(them, True), "castle_opp_q": has(them, False),
        "ep_file": ep,
        "rule50": min(board.halfmove_clock, 255),
    }
    return rec, state


def record_to_board(rec: np.ndarray) -> chess.Board:
    """Canonical record -> board (white to move, no castling/ep — what the
    network sees)."""
    board = chess.Board(None)
    occ = int(rec["occ"])
    pcs = rec["pcs"]
    i = 0
    sq = occ
    while sq:
        s = (sq & -sq).bit_length() - 1
        nib = (int(pcs[i // 2]) >> (4 * (i % 2))) & 0xF
        color = chess.WHITE if (nib >> 3) == 0 else chess.BLACK
        pt = (nib & 7) + 1  # chess.PAWN == 1
        board.set_piece_at(s, chess.Piece(pt, color))
        sq &= sq - 1
        i += 1
    board.turn = chess.WHITE
    board.castling_rights = 0
    board.ep_square = None
    return board


def cmd_roundtrip() -> int:
    recs = np.memmap(ROOT / "data/nnue/heldout/uho-band0-5.bin",
                     dtype=RECORD_DTYPE, mode="r")[:20000]
    bad = 0
    for i in range(len(recs)):
        b = record_to_board(recs[i])
        r2, _ = board_to_record(b)
        for f in ("occ", "pcs", "ksq", "opp_ksq"):
            if not np.array_equal(recs[i][f], r2[0][f]):
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH row {i} field {f}: {recs[i][f]} vs {r2[0][f]}")
                break
    print(f"roundtrip: {len(recs) - bad}/{len(recs)} byte-identical "
          f"(occ/pcs/ksq/opp_ksq)")
    return 1 if bad else 0


def cmd_flip_selftest() -> int:
    """Canonicalization involution: record(fen) == record(mirror(fen))."""
    import random
    rng = random.Random(7)
    recs = np.memmap(ROOT / "data/nnue/heldout/uho-band0-5.bin",
                     dtype=RECORD_DTYPE, mode="r")
    bad = 0
    for i in rng.sample(range(len(recs)), 2000):
        b = record_to_board(recs[i])            # white to move
        m = b.mirror()                          # black to move, colors swapped
        m.halfmove_clock = b.halfmove_clock
        r1, s1 = board_to_record(b)
        r2, s2 = board_to_record(m)
        if not (np.array_equal(r1[0]["occ"], r2[0]["occ"])
                and np.array_equal(r1[0]["pcs"], r2[0]["pcs"])
                and r1[0]["ksq"] == r2[0]["ksq"]
                and r1[0]["opp_ksq"] == r2[0]["opp_ksq"] and s1 == s2):
            bad += 1
    print(f"flip involution: {2000 - bad}/2000 identical")
    return 1 if bad else 0


def cmd_tsv(src: str, out: str, limit: int = 0) -> int:
    rows = []
    with open(src) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                rows.append(parts)
            if limit and len(rows) >= limit:
                break
    n = len(rows)
    recs = np.zeros(n, dtype=RECORD_DTYPE)
    st = {k: np.zeros(n, dt) for k, dt in [
        ("castle_own_k", np.uint8), ("castle_own_q", np.uint8),
        ("castle_opp_k", np.uint8), ("castle_opp_q", np.uint8),
        ("ep_file", np.int8), ("rule50", np.uint8), ("ply", np.uint16),
        ("score_ctl", np.int16), ("result2", np.uint8),
        ("fenidx", np.int64)]}
    kept = 0
    for i, (fen, res, ply, s5k) in enumerate(rows):
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        rec, state = board_to_record(board)
        res_w, s_w = float(res), float(s5k)
        if board.turn == chess.BLACK:
            res_w, s_w = 1.0 - res_w, -s_w
        s_ctl = max(-2499.0, min(2499.0, s_w / 1.9))
        recs[kept] = rec[0]
        recs[kept]["score"] = np.int16(int(s_ctl))
        recs[kept]["result"] = np.uint8(int(res_w * 2))
        for k, v in state.items():
            st[k][kept] = v
        st["ply"][kept] = min(int(ply), 65535)
        st["score_ctl"][kept] = np.int16(int(s_ctl))
        st["result2"][kept] = np.uint8(int(res_w * 2))
        st["fenidx"][kept] = i
        kept += 1
        if kept % 50000 == 0:
            print(f"  {kept}/{n}", flush=True)
    recs = recs[:kept]
    st = {k: v[:kept] for k, v in st.items()}
    np.savez(out, recs=recs.view(np.uint8).reshape(kept, 32), **st)
    print(f"tsv: {kept}/{n} encoded -> {out}")
    return 0


def cmd_fens(src: str, out: str) -> int:
    """Plain FEN file -> records npz (state sidecar included)."""
    with open(src) as fh:
        fens = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    rows = [(f, "0.5", "0", "0") for f in fens]
    tmp = Path(out).with_suffix(".tmp.tsv")
    tmp.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    rc = cmd_tsv(str(tmp), out)
    tmp.unlink()
    return rc


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "roundtrip":
        rc = cmd_roundtrip() or cmd_flip_selftest()
    elif cmd == "tsv":
        rc = cmd_tsv(*sys.argv[2:4])
    elif cmd == "fens":
        rc = cmd_fens(*sys.argv[2:4])
    else:
        rc = 2
    sys.exit(rc)
