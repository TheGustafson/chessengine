"""Representation-gap study: probe-target battery over canonical records.

Every target is a PROBE LABEL (diagnostic ground truth), not a feature —
hand-choosing what to TEST for is legitimate; hand-choosing what to FEED the
net is the failed shape #4. Targets are computed on the state-blind board
(exactly what the network sees: placement + stm, no castling/ep/rule50).

Controls:
  material   stm-relative material balance, pawn units  [MUST be decodable]
  pcount     piece count                                 [MUST be decodable]
  hashbit    multiplicative hash bit of occ              [MUST be chance]

Function-class targets (deterministic functions of placement the linear+
640-product architecture may or may not be able to express):
  has_capture     stm has at least one legal capture
  see_best        max SEE over stm legal captures (cp, VAL scale; -1 if none)
  see_pos         see_best > 0 (stm wins material with best capture)
  opp_see_pos     opponent (null-move) has a winning capture — "stm leaves a
                  piece en prise"; valid_opp=0 when stm in check
  in_check        stm in check
  hang_cnt        stm non-king pieces attacked and undefended
  atk_lower       stm pieces attacked by a strictly cheaper attacker
  pin_own/opp     absolutely pinned piece counts
  passed_own/opp  passed pawn counts
  pawn_end        kings+pawns only (validity flag for the two below)
  sqrule_win      stm passed pawn unstoppable by the square rule (approx:
                  path clear, own king off path; noted in doc)
  opposition      kings in direct opposition (gap 2, same file/rank) — with
                  stm to move the OPPONENT holds the opposition
Run:
  .venv/bin/python tools/nnue/reprgap/targets.py heldout 500000
  .venv/bin/python tools/nnue/reprgap/targets.py mix 350000
  .venv/bin/python tools/nnue/reprgap/targets.py recs <recs.npz>
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "jax"))
sys.path.insert(0, str(ROOT / "tools/nnue/reprgap"))

import chess  # noqa: E402

from archon_jax.spec import RECORD_DTYPE  # noqa: E402
from encode import record_to_board  # noqa: E402

OUT = ROOT / "data/nnue/nets/screens/reprgap"
VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
SOURCES = {
    "mix": (ROOT / "data/nnue/jaxcorpora/uho-band75-100.shuf.bin", 5_000_000),
    "heldout": (ROOT / "data/nnue/heldout/uho-band0-5.bin", 0),
}

FIELDS = ["material", "pcount", "hashbit", "has_capture", "see_best",
          "see_pos", "opp_see_pos", "valid_opp", "in_check", "hang_cnt",
          "atk_lower", "pin_own", "pin_opp", "passed_own", "passed_opp",
          "pawn_end", "sqrule_win", "opposition"]


def _attackers(board: chess.Board, color: bool, sq: int, occ: int) -> int:
    """Attackers of sq by `color` under custom occupancy (x-ray aware)."""
    r = chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
    f = chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
    d = chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
    att = ((chess.BB_KING_ATTACKS[sq] & board.kings)
           | (chess.BB_KNIGHT_ATTACKS[sq] & board.knights)
           | ((r | f) & (board.rooks | board.queens))
           | (d & (board.bishops | board.queens))
           | (chess.BB_PAWN_ATTACKS[not color][sq] & board.pawns))
    return att & board.occupied_co[color] & occ


def see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation of a capture (swap algorithm, x-ray aware).

    Promotion gains ignored (a promoting capture is scored as the capture
    only — conservative; rare in the battery)."""
    to_sq, from_sq = move.to_square, move.from_square
    occ = board.occupied
    if board.is_en_passant(move):
        captured = chess.PAWN
        cap_sq = to_sq + (-8 if board.turn == chess.WHITE else 8)
        occ ^= chess.BB_SQUARES[cap_sq]
    else:
        captured = board.piece_type_at(to_sq)
    gain = [VAL[captured]]
    cur = board.piece_type_at(from_sq)
    occ ^= chess.BB_SQUARES[from_sq]
    side = not board.turn
    while True:
        att = _attackers(board, side, to_sq, occ)
        if not att:
            break
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
                   chess.QUEEN, chess.KING):
            sub = att & board.pieces_mask(pt, side)
            if sub:
                fsq = chess.lsb(sub)
                break
        if pt == chess.KING and _attackers(board, not side, to_sq,
                                           occ ^ chess.BB_SQUARES[fsq]):
            break  # king recapture into a defended square is illegal
        gain.append(VAL[cur] - gain[-1])
        cur = pt
        occ ^= chess.BB_SQUARES[fsq]
        side = not side
    for d in range(len(gain) - 1, 0, -1):
        gain[d - 1] = -max(-gain[d - 1], gain[d])
    return gain[0]


def _passed(board: chess.Board, color: bool) -> int:
    cnt = 0
    ahead = range if color == chess.WHITE else lambda a, b: range(a, b, -1)
    for sq in chess.scan_forward(board.pieces_mask(chess.PAWN, color)):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        rr = range(r + 1, 8) if color == chess.WHITE else range(r - 1, -1, -1)
        opp_p = board.pieces_mask(chess.PAWN, not color)
        for r2 in rr:
            for f2 in (f - 1, f, f + 1):
                if 0 <= f2 <= 7 and opp_p & chess.BB_SQUARES[chess.square(f2, r2)]:
                    blocked = True
                    break
            if blocked:
                break
        cnt += not blocked
    return cnt


def _sqrule(board: chess.Board) -> int:
    """stm (white) has a square-rule-unstoppable passer. Pawn endgames only."""
    ok = board.kings | board.pawns
    if board.occupied != ok:
        return 0
    bk = board.king(chess.BLACK)
    wk = board.king(chess.WHITE)
    for sq in chess.scan_forward(board.pieces_mask(chess.PAWN, chess.WHITE)):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        # passed?
        opp_p = board.pieces_mask(chess.PAWN, chess.BLACK)
        clear = True
        for r2 in range(r + 1, 8):
            for f2 in (f - 1, f, f + 1):
                if 0 <= f2 <= 7 and opp_p & chess.BB_SQUARES[chess.square(f2, r2)]:
                    clear = False
                    break
            if not clear:
                break
        if not clear:
            continue
        # path clear of ANY piece (incl. own king), own king not on path
        path_clear = all(
            not (board.occupied & chess.BB_SQUARES[chess.square(f, r2)])
            for r2 in range(r + 1, 8))
        if not path_clear:
            continue
        promo = chess.square(f, 7)
        pdist = 7 - r - (1 if r == 1 else 0)
        if chess.square_distance(bk, promo) > pdist and wk != promo:
            return 1
    return 0


def _one(rec_bytes: bytes) -> tuple:
    rec = np.frombuffer(rec_bytes, dtype=RECORD_DTYPE)[0]
    b = record_to_board(rec)
    occ = int(rec["occ"])
    material = sum(VAL[p.piece_type] * (1 if p.color else -1)
                   for p in b.piece_map().values()
                   if p.piece_type != chess.KING) / 100.0
    pcount = chess.popcount(occ)
    hashbit = int(((occ * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 63)
    in_check = int(b.is_check())

    caps = [m for m in b.legal_moves if b.is_capture(m)]
    sees = [see(b, m) for m in caps]
    see_best = max(sees) if sees else -1
    has_capture = int(bool(caps))
    see_pos = int(see_best > 0)

    valid_opp = int(not in_check)
    opp_see_pos = 0
    if valid_opp:
        nb = b.copy(stack=False)
        nb.turn = chess.BLACK
        try:
            osees = [see(nb, m) for m in nb.legal_moves if nb.is_capture(m)]
            opp_see_pos = int(bool(osees) and max(osees) > 0)
        except Exception:
            valid_opp = 0

    hang_cnt = 0
    atk_lower = 0
    for sq, pc in b.piece_map().items():
        if not pc.color or pc.piece_type == chess.KING:
            continue
        att = _attackers(b, chess.BLACK, sq, b.occupied)
        if not att:
            continue
        if not _attackers(b, chess.WHITE, sq, b.occupied):
            hang_cnt += 1
        cheapest = min(VAL[b.piece_type_at(a)] for a in chess.scan_forward(att))
        if cheapest < VAL[pc.piece_type]:
            atk_lower += 1
    pin_own = sum(b.is_pinned(chess.WHITE, sq)
                  for sq in chess.scan_forward(b.occupied_co[chess.WHITE]))
    pin_opp = sum(b.is_pinned(chess.BLACK, sq)
                  for sq in chess.scan_forward(b.occupied_co[chess.BLACK]))
    passed_own = _passed(b, chess.WHITE)
    passed_opp = _passed(b, chess.BLACK)
    pawn_end = int(b.occupied == (b.kings | b.pawns))
    sqrule_win = _sqrule(b) if pawn_end else 0
    opposition = 0
    if pawn_end:
        wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
        df = abs(chess.square_file(wk) - chess.square_file(bk))
        dr = abs(chess.square_rank(wk) - chess.square_rank(bk))
        opposition = int((df == 0 and dr == 2) or (dr == 0 and df == 2))
    return (material, pcount, hashbit, has_capture, see_best, see_pos,
            opp_see_pos, valid_opp, in_check, hang_cnt, atk_lower, pin_own,
            pin_opp, passed_own, passed_opp, pawn_end, sqrule_win, opposition)


def _chunk(args) -> list[tuple]:
    blob, n = args
    return [_one(blob[i * 32:(i + 1) * 32]) for i in range(n)]


def selftest() -> None:
    b = chess.Board("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1")
    s = see(b, chess.Move.from_uci("e1e5"))
    assert s == 100, f"SEE classic 1 (RxP undefended): {s}"
    b = chess.Board("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1")
    s = see(b, chess.Move.from_uci("d3e5"))
    assert s < 0, f"SEE classic 2 (NxP defended twice): {s}"
    b = chess.Board("k7/8/8/4p3/3P4/8/8/K7 w - - 0 1")
    s = see(b, chess.Move.from_uci("d4e5"))
    assert s == 100, f"SEE pawn takes free pawn: {s}"
    b = chess.Board("k2r4/8/8/3p4/8/3R4/3R4/K7 w - - 0 1")
    s = see(b, chess.Move.from_uci("d3d5"))
    assert s == 100, f"SEE xray battery RxP: {s}"
    print("SEE selftest OK")


def main() -> int:
    selftest()
    src = sys.argv[1]
    if src == "recs":
        path = Path(sys.argv[2])
        recs = np.load(path)["recs"]
        blob = recs.tobytes()
        n = len(recs)
        tag = path.stem
    else:
        path, base = SOURCES[src]
        count = int(sys.argv[2])
        recs = np.memmap(path, dtype=RECORD_DTYPE, mode="r")[base:base + count]
        blob = np.array(recs).tobytes()
        n = count
        tag = f"{src}-{base}-{count}"
    CH = 8192
    jobs = [(blob[i * 32:(i + min(CH, n - i)) * 32], min(CH, n - i))
            for i in range(0, n, CH)]
    rows: list[tuple] = []
    with Pool(4) as pool:
        for k, out in enumerate(pool.imap(_chunk, jobs)):
            rows.extend(out)
            if (k + 1) % 8 == 0:
                print(f"  {len(rows)}/{n}", flush=True)
    arr = np.asarray(rows, dtype=np.float32)
    np.savez(OUT / f"targets-{tag}.npz",
             **{f: arr[:, i] for i, f in enumerate(FIELDS)})
    print(f"targets: {n} rows -> targets-{tag}.npz")
    for i, f in enumerate(FIELDS):
        c = arr[:, i]
        print(f"  {f:12s} mean {c.mean():8.3f}  nonzero {np.mean(c != 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
