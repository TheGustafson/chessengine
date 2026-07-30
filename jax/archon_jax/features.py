"""Vectorized numpy feature extraction for bulletformat ChessBoard batches.

Reimplements, over whole record batches at once:

- HalfKAv2_hm index math      (src/eval/nnue/sota_features.rs lines 60-125)
- FullThreats tables + index  (sota_features.rs lines 130-348)
- threat enumeration           (sota_features.rs for_each_threat, lines 376-434;
                                identical victim masks to the trainer's
                                emit_features, d40_features.rs lines 340-413)
- the 1858-class policy vocab  (src/eval/nnue/policy_vocab.rs)

Everything is index-exact against the Rust sources; the unit tests in
``jax/tests/test_features.py`` pin the hand-computed vectors from the Rust
test modules.

Conventions (bulletformat stored frame): the mover is color 0 ("White") on a
vertically-normalized board, so ``perspective 0`` = stm and ``perspective 1``
= ntm, exactly like the trainer's (stm, ntm) sparse pairs.
"""

from __future__ import annotations

import numpy as np

from .spec import (
    HALFKA_DIMS,
    HALFKA_MAX_ACTIVE,
    PS_KING,
    PS_NB,
    THREAT_DIMS,
    THREAT_MAX_ACTIVE,
    GEOMETRIC_CLASSES,
    POLICY_VOCAB,
    OUTPUT_BUCKETS_LAYOUT,
)

U64 = np.uint64

# ===========================================================================
# Static square tables
# ===========================================================================

# KingBuckets (sota_features.rs:60-70), square index a1=0 ... h8=63.
KING_BUCKETS = np.array([
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
], dtype=np.int64)

_SQ = np.arange(64)
_FILE = _SQ & 7
_RANK = _SQ >> 3


def _bb(squares) -> np.uint64:
    out = np.uint64(0)
    for s in squares:
        out |= U64(1) << U64(s)
    return out


def _leaper_table(deltas) -> np.ndarray:
    """64-entry attack table for a fixed-offset leaper."""
    table = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        acc = 0
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                acc |= 1 << (nr * 8 + nf)
        table[sq] = acc
    return table


KNIGHT_ATTACKS = _leaper_table(
    [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])
KING_ATTACKS = _leaper_table(
    [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])
# PAWN_ATTACKS[color][sq]: diagonal capture squares (color 0 = toward rank 8).
PAWN_ATTACKS = np.stack([
    _leaper_table([(-1, 1), (1, 1)]),
    _leaper_table([(-1, -1), (1, -1)]),
])

# Slider rays: RAYS[d][sq] excludes sq itself, runs to the board edge.
# Directions 0..3 are "positive" (target square > from square): N, E, NE, NW.
# Directions 4..7 are "negative": S, W, SW, SE.
_DIRS = [(0, 1), (1, 0), (1, 1), (-1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1)]
RAYS = np.zeros((8, 64), dtype=np.uint64)
for d, (df, dr) in enumerate(_DIRS):
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        acc = 0
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            acc |= 1 << (nr * 8 + nf)
            nf += df
            nr += dr
        RAYS[d][sq] = acc

BISHOP_DIRS = (2, 3, 6, 7)
ROOK_DIRS = (0, 1, 4, 5)

# Empty-board pseudo attacks for sliders (used by the threat geometry tables).
BISHOP_PSEUDO = RAYS[2] | RAYS[3] | RAYS[6] | RAYS[7]
ROOK_PSEUDO = RAYS[0] | RAYS[1] | RAYS[4] | RAYS[5]
QUEEN_PSEUDO = BISHOP_PSEUDO | ROOK_PSEUDO


def _msb_u64(x: np.ndarray) -> np.ndarray:
    """Isolate the most significant set bit of each nonzero u64 (0 stays 0)."""
    x = x.copy()
    for shift in (1, 2, 4, 8, 16, 32):
        x |= x >> U64(shift)
    return x ^ (x >> U64(1))


def slider_attacks(dirs, from_sq: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Classic ray attacks with blockers, vectorized over instances.

    ``from_sq`` int64 (M,), ``occ`` uint64 (M,) -> uint64 (M,) attack sets.
    A blocker square is included in the attack set (its occupant is a target),
    matching magic-bitboard semantics (attacks() in src/attacks/).
    """
    att = np.zeros(from_sq.shape, dtype=np.uint64)
    one = U64(1)
    for d in dirs:
        ray = RAYS[d][from_sq]
        blockers = ray & occ
        if d < 4:  # positive ray: first blocker = LSB
            first = blockers & (~blockers + one)
            # squares <= first blocker: (first << 1) - 1 ; no blocker -> full ray
            below = np.where(blockers != 0, (first << one) - one, ~U64(0))
            att |= ray & below
        else:      # negative ray: first blocker = MSB
            msb = _msb_u64(blockers)
            above = np.where(blockers != 0, ~(msb - one), ~U64(0))
            att |= ray & above
    return att


def bits_to_pairs(bb: np.ndarray):
    """(M,) uint64 -> (owner_idx, square) arrays over all set bits.

    Order: owner-major, square-ascending (numpy nonzero row-major order),
    identical to Bitboard iteration order (trailing_zeros) in Rust.
    """
    bytes_ = bb.view(np.uint8).reshape(-1, 8)
    bits = np.unpackbits(bytes_, axis=1, bitorder="little")  # (M, 64)
    owner, sq = np.nonzero(bits)
    return owner, sq


# ===========================================================================
# Record decode
# ===========================================================================

class DecodedBatch:
    """Piece lists + bitboards for a batch of bulletformat records.

    Attributes (P = total piece instances across the batch):
      row, sq, color, pt : (P,) int64  -- piece list, row-major/square-ascending
      ordinal            : (P,) int64  -- per-row piece ordinal (nibble index)
      piece_bb           : (N, 2, 6) uint64
      occ                : (N,) uint64
      kings              : (N, 2) int64 -- [white(=stm), black] king squares
      mailbox_code       : (N, 64) int8 -- SF piece code 1..6 / 9..14, 0 = empty
                           (sota_features.rs sf_code, line 138)
    """

    def __init__(self, recs: np.ndarray):
        n = len(recs)
        occ = np.ascontiguousarray(recs["occ"])
        bits = np.unpackbits(occ.view(np.uint8).reshape(n, 8), axis=1,
                             bitorder="little")
        row, sq = np.nonzero(bits)
        counts = bits.sum(axis=1)
        starts = np.zeros(n, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        ordinal = np.arange(len(row)) - starts[row]

        pcs = recs["pcs"]
        nib = (pcs[row, ordinal >> 1] >> ((ordinal & 1) << 2)) & 0xF
        color = (nib >> 3).astype(np.int64)          # 0 = stm("white"), 1 = opp
        pt = (nib & 7).astype(np.int64)              # 0=P 1=N 2=B 3=R 4=Q 5=K

        piece_bb = np.zeros((n, 2, 6), dtype=np.uint64)
        np.bitwise_or.at(piece_bb, (row, color, pt), U64(1) << sq.astype(np.uint64))

        kings = np.zeros((n, 2), dtype=np.int64)
        kmask = pt == 5
        kings[row[kmask], color[kmask]] = sq[kmask]

        mailbox = np.zeros((n, 64), dtype=np.int8)
        mailbox[row, sq] = ((color << 3) | (pt + 1)).astype(np.int8)

        self.n = n
        self.row, self.sq, self.color, self.pt = row, sq, color, pt
        self.ordinal = ordinal
        self.piece_bb = piece_bb
        self.occ = occ.astype(np.uint64)
        self.kings = kings
        self.mailbox_code = mailbox
        self.piece_count = counts.astype(np.int64)


# ===========================================================================
# HalfKAv2_hm (sota_features.rs:77-125)
# ===========================================================================

def _orient_halfka(ksq: np.ndarray) -> np.ndarray:
    """7 (file mirror) for a king on files a-d, else 0 (sota:77-79)."""
    return np.where((ksq & 7) <= 3, 7, 0).astype(np.int64)


def halfka_batch(dec: DecodedBatch) -> np.ndarray:
    """(N, 2, 32) int32 HalfKA indices, -1 padded; axis 1 = perspective
    (0 = stm/White, 1 = ntm/Black). One feature per piece, in piece-list order
    (the padding position is the piece ordinal, capacity 32)."""
    out = np.full((dec.n, 2, HALFKA_MAX_ACTIVE), -1, dtype=np.int32)
    for persp in (0, 1):
        flip = 56 * persp
        ksq = dec.kings[:, persp]                       # raw king square
        orient = _orient_halfka(ksq)                    # file-mirror by RAW file
        bucket = KING_BUCKETS[ksq ^ flip]
        o_row, b_row = orient[dec.row], bucket[dec.row]
        oriented_sq = dec.sq ^ o_row ^ flip
        # piece_square_index (sota:99-110): king -> 640; else pt*128 (+64 enemy)
        psq = np.where(dec.pt == 5, PS_KING,
                       dec.pt * 128 + np.where(dec.color == persp, 0, 64))
        idx = oriented_sq + psq + b_row * PS_NB
        out[dec.row, persp, dec.ordinal] = idx.astype(np.int32)
    return out


# ===========================================================================
# FullThreats tables (sota_features.rs:230-312), built once at import
# ===========================================================================

ALL_PIECES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
NUM_VALID_TARGETS = np.array(
    [0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0], dtype=np.int64)
_MAP = np.array([
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
], dtype=np.int64)


def _geom_attacks_code(code: int, sq: int) -> int:
    color = 0 if code < 8 else 1
    kind = code & 7
    if kind == 1:  # pawn: push-or-attacks
        bb = int(PAWN_ATTACKS[color][sq])
        push = sq + 8 if color == 0 else sq - 8
        if 0 <= push < 64:
            bb |= 1 << push
        return bb
    if kind == 2:
        return int(KNIGHT_ATTACKS[sq])
    if kind == 3:
        return int(BISHOP_PSEUDO[sq])
    if kind == 4:
        return int(ROOK_PSEUDO[sq])
    if kind == 5:
        return int(QUEEN_PSEUDO[sq])
    if kind == 6:
        return int(KING_ATTACKS[sq])
    return 0


def _build_threat_tables():
    offsets = np.zeros((16, 64), dtype=np.int64)
    piece_offset = np.zeros(16, dtype=np.int64)
    cumulative_offset = np.zeros(16, dtype=np.int64)
    lut2 = np.zeros((16, 64, 64), dtype=np.int64)

    cumulative = 0
    for code in ALL_PIECES:
        is_pawn = (code & 7) == 1
        cum_piece = 0
        for frm in range(64):
            offsets[code][frm] = cum_piece
            att = _geom_attacks_code(code, frm)
            for to in range(64):
                lut2[code][frm][to] = bin(att & ((1 << to) - 1)).count("1")
            if not is_pawn or 8 <= frm <= 55:
                cum_piece += bin(att).count("1")
        piece_offset[code] = cum_piece
        cumulative_offset[code] = cumulative
        cumulative += NUM_VALID_TARGETS[code] * cum_piece
    assert cumulative == THREAT_DIMS, cumulative

    # lut1[att][atd][from_o < to_o]; sentinel THREAT_DIMS = excluded.
    lut1 = np.full((16, 16, 2), THREAT_DIMS, dtype=np.int64)
    for att in ALL_PIECES:
        for atd in ALL_PIECES:
            att_t, atd_t = att & 7, atd & 7
            m = _MAP[att_t - 1][atd_t - 1]
            if m < 0:
                continue
            same_type_diff_color = (att ^ atd) == 8
            semi_excluded = att_t == atd_t and (same_type_diff_color or att_t != 1)
            atd_color = atd >> 3
            num_types = NUM_VALID_TARGETS[att] // 2
            sub = atd_color * num_types + m
            feature = cumulative_offset[att] + sub * piece_offset[att]
            lut1[att][atd][0] = feature
            lut1[att][atd][1] = THREAT_DIMS if semi_excluded else feature
    return offsets, piece_offset, cumulative_offset, lut2, lut1


THREAT_OFFSETS, THREAT_PIECE_OFFSET, THREAT_CUM_OFFSET, THREAT_LUT2, THREAT_LUT1 = (
    _build_threat_tables())


def _orient_threats(ksq: np.ndarray) -> np.ndarray:
    """OPPOSITE polarity to HalfKA (sota:169-176): 0 for files a-d, else 7."""
    return np.where((ksq & 7) <= 3, 0, 7).astype(np.int64)


def enumerate_threat_relations(dec: DecodedBatch):
    """All directed attacker->victim relations on each board, perspective-free.

    Returns (row, att_code, from_sq, to_sq, atd_code) int64 arrays. The victim
    masks are exactly for_each_threat's (sota:380-386 == trainer d40:358-365):
      pawn targets: P/N/R (+ the push-square pawn), knight/queen: +B+Q,
      bishop/rook: +B. Kings never attack.
    """
    bb = dec.piece_bb
    both = bb[:, 0, :] | bb[:, 1, :]
    all_pawns = both[:, 0]
    pawn_targets = both[:, 0] | both[:, 1] | both[:, 3]
    minor_slider_targets = pawn_targets | both[:, 2]
    queen_targets = minor_slider_targets | both[:, 4]

    rows, att_codes, froms, tos = [], [], [], []

    def emit(row_idx, code, from_sq, att_bb):
        owner, to = bits_to_pairs(att_bb)
        rows.append(row_idx[owner])
        att_codes.append(code[owner] if isinstance(code, np.ndarray)
                         else np.full(len(owner), code, dtype=np.int64))
        froms.append(from_sq[owner])
        tos.append(sq_from := to.astype(np.int64))
        return sq_from

    for color in (0, 1):
        # Pawns: diagonal captures on pawn_targets + the push-threat square.
        mask = (dec.color == color) & (dec.pt == 0)
        prow, psq = dec.row[mask], dec.sq[mask]
        code = 1 + 8 * color
        att = PAWN_ATTACKS[color][psq] & pawn_targets[prow]
        push = psq + (8 if color == 0 else -8)
        ok = (push >= 0) & (push < 64)
        push_bb = np.zeros(len(psq), dtype=np.uint64)
        push_bb[ok] = U64(1) << push[ok].astype(np.uint64)
        att |= push_bb & all_pawns[prow]
        emit(prow, code, psq, att)

        # Knights / bishops / rooks / queens.
        for pt, dirs, tgt in ((1, None, queen_targets),
                              (2, BISHOP_DIRS, minor_slider_targets),
                              (3, ROOK_DIRS, minor_slider_targets),
                              (4, BISHOP_DIRS + ROOK_DIRS, queen_targets)):
            mask = (dec.color == color) & (dec.pt == pt)
            prow, psq = dec.row[mask], dec.sq[mask]
            code = pt + 1 + 8 * color
            if pt == 1:
                att = KNIGHT_ATTACKS[psq] & tgt[prow]
            else:
                att = slider_attacks(dirs, psq, dec.occ[prow]) & tgt[prow]
            emit(prow, code, psq, att)

    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    att = np.concatenate(att_codes) if att_codes else np.zeros(0, dtype=np.int64)
    frm = np.concatenate(froms) if froms else np.zeros(0, dtype=np.int64)
    to = np.concatenate(tos) if tos else np.zeros(0, dtype=np.int64)
    atd = dec.mailbox_code[row, to].astype(np.int64)
    return row, att, frm, to, atd


def threat_batch(dec: DecodedBatch, relations=None) -> np.ndarray:
    """(N, 2, 160) int32 FullThreats indices, -1 padded, per perspective.

    Matches for_each_threat (sota:376-434): every relation is tested under each
    perspective; excluded / semi-excluded directions return the sentinel and
    are dropped. threat_index formula: sota:324-348.
    """
    if relations is None:
        relations = enumerate_threat_relations(dec)
    row, att, frm, to, atd = relations

    out = np.full((dec.n, 2, THREAT_MAX_ACTIVE), -1, dtype=np.int32)
    order = None
    for persp in (0, 1):
        ksq = dec.kings[:, persp]
        orientation = _orient_threats(ksq) ^ (56 * persp)   # sota:334
        o = orientation[row]
        from_o = frm ^ o
        to_o = to ^ o
        swap = 8 * persp
        att_p = att ^ swap
        atd_p = atd ^ swap
        direction = (from_o < to_o).astype(np.int64)
        base = THREAT_LUT1[att_p, atd_p, direction]
        valid = base < THREAT_DIMS
        idx = np.where(
            valid,
            base + THREAT_OFFSETS[att_p, from_o] + THREAT_LUT2[att_p, from_o, to_o],
            -1)
        # Scatter valid indices into per-row slots. Relations are grouped by
        # (color, piece-type) category, not by row, so sort by row (stable).
        if order is None:
            order = np.argsort(row, kind="stable")
        r_s, i_s, v_s = row[order], idx[order], valid[order]
        r_v, i_v = r_s[v_s], i_s[v_s]
        # per-row running ordinal
        counts = np.bincount(r_v, minlength=dec.n)
        starts = np.zeros(dec.n, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        slot = np.arange(len(r_v)) - starts[r_v]
        if len(slot) and slot.max() >= THREAT_MAX_ACTIVE:
            raise ValueError("FullThreats sparse capacity exceeded")
        out[r_v, persp, slot] = i_v.astype(np.int32)
    return out


# ===========================================================================
# Policy vocabulary (policy_vocab.rs)
# ===========================================================================

def _build_geom_table() -> np.ndarray:
    table = np.full(64 * 64, -1, dtype=np.int64)
    idx = 0
    for frm in range(64):
        ff, fr = frm % 8, frm // 8
        for to in range(64):
            if to == frm:
                continue
            tf, tr = to % 8, to // 8
            df, dr = tf - ff, tr - fr
            queen_like = df == 0 or dr == 0 or abs(df) == abs(dr)
            knight_like = (abs(df) == 1 and abs(dr) == 2) or (abs(df) == 2 and abs(dr) == 1)
            if queen_like or knight_like:
                table[frm * 64 + to] = idx
                idx += 1
    assert idx == GEOMETRIC_CLASSES
    return table


GEOM_TABLE = _build_geom_table()

# Underpromotion pair index (policy_vocab.rs:83-105): from rank 7 -> rank 8,
# |df| <= 1, enumerated from-major/to-ascending; 22 pairs.
_UNDERPROMO_PAIR = np.full(64 * 64, -1, dtype=np.int64)
_pair = 0
for _f in range(48, 56):
    for _t in range(56, 64):
        if abs((_t % 8) - (_f % 8)) <= 1:
            _UNDERPROMO_PAIR[_f * 64 + _t] = _pair
            _pair += 1
assert _pair == 22


def policy_move_id(from_c: np.ndarray, to_c: np.ndarray,
                   promo_code: np.ndarray) -> np.ndarray:
    """Vectorized policy_move_id (policy_vocab.rs:137-155).

    promo_code: 0/4+ = none-or-queen (geometric class), 1=N 2=B 3=R
    (underpromo tail, id = 1792 + pair*3 + (code-1)). Returns -1 for
    out-of-geometry pairs.
    """
    in_range = (from_c < 64) & (to_c < 64)
    fromto = np.where(in_range, from_c.astype(np.int64) * 64 + to_c.astype(np.int64), 0)
    geom = np.where(in_range, GEOM_TABLE[fromto], -1)
    pair = _UNDERPROMO_PAIR[fromto]
    under = (promo_code >= 1) & (promo_code <= 3)
    under_id = np.where(pair >= 0, GEOMETRIC_CLASSES + pair * 3 + (promo_code - 1), -1)
    return np.where(under, under_id, geom)


def canonical_orient(ksq_mover_frame: np.ndarray) -> np.ndarray:
    """policy_vocab.rs canonical_orient for the stored frame: the vertical flip
    is already baked in (stm is White), leaving only the HalfKA file mirror
    decided by the STORED king file (d40_features.rs:913-919)."""
    return np.where((ksq_mover_frame & 7) <= 3, 7, 0).astype(np.int64)


# ===========================================================================
# One-call batch mapper (the ModelInputsMapper equivalent, d40:835-946)
# ===========================================================================

def map_batch(recs: np.ndarray, *, wdl: float, score_multiplier: float,
              scale: float, policy: bool = False,
              threats: bool = True, target_offset: float = 0.0,
              piece_rebalance: bool = False,
              wdl_filter: bool = False,
              wdl_filter_floor: float = 0.0,
              threat_remap: np.ndarray | None = None) -> dict:
    """records -> the dense/sparse training tensors, bullet-mapper-parity.

    Returns dict with:
      hk_stm/hk_ntm  (N,32) int32, th_stm/th_ntm (N,160) int32 (all -1 padded)
      bucket         (N,) int32
      target         (N,) float32   wdl*result + (1-wdl)*transform(score)

    ``target_offset`` selects the eval->expected-score transform (D47 target
    calibration, 2026-07-26):
      0.0  (default)  plain  sigmoid(s/scale)      — the historical recipe
      a>0             draw-aware DOUBLE sigmoid
                        0.5*(1 + sig((s-a)/scale) - sig((-s-a)/scale))
                      i.e. P(win)=sig((s-a)/scale), P(loss)=sig((-s-a)/scale),
                      remainder draw; the value is the EXPECTED SCORE, which
                      is the same quantity as ``result``, so the wdl blend
                      finally mixes like-with-like.
    MLE fits on our own corpora (tools/nnue/fit_target_transform.py, held
    out, 1.5M each): uho-band75-100 a=175.5 scale=71.9 (brier .0354 vs
    .0532 for the current sigmoid(s/400)); fishpack32 a=160.2 scale=33.2
    (.0233 vs .0499). SF's own 270/340 is no better for us than our 400.
    NOTE the corpora disagree (scale 72 vs 33) — a single global transform
    over a mix is itself a compromise.
      pol_target_id  (N,) int32     policy class id or -1 (no packed move)
    The legal-move mask is NOT produced here (needs movegen); see data.py.
    """
    dec = DecodedBatch(recs)
    hk = halfka_batch(dec)
    out = {
        "hk_stm": hk[:, 0], "hk_ntm": hk[:, 1],
        "bucket": OUTPUT_BUCKETS_LAYOUT[
            np.minimum(dec.piece_count, 32)].astype(np.int32),
    }
    if threats:
        th = threat_batch(dec)
        if threat_remap is not None:
            # Recode POC (learned-feature-code study §6.1): re-code the raw
            # 60,720-space indices onto the candidate row space. -1 padding is
            # preserved; the embedding table is (threat_rows, hl). The engine
            # applies the SAME u16 table at its resolver chokepoint (TRM1
            # section), so trainer and engine cannot disagree by construction.
            th = np.where(th >= 0,
                          threat_remap[np.maximum(th, 0)].astype(np.int32), th)
        out["th_stm"], out["th_ntm"] = th[:, 0], th[:, 1]

    score = recs["score"].astype(np.float32)
    result = recs["result"].astype(np.float32) / 2.0
    s = score * score_multiplier
    # target_offset/scale may be SCALARS (one global transform) or per-record
    # ARRAYS (D48Y: one fitted draw-aware transform PER CORPUS, since the
    # score->outcome relation is a property of the labelling process and the
    # position population, not of chess).
    if np.ndim(target_offset) > 0 or target_offset > 0.0:
        # clip the exponent: float32 exp overflows past ~88.7, which happens
        # for |score| > 88.7*scale - offset (~6.7k at scale 76.9). The overflow
        # is BENIGN (inf -> 1/(1+inf) -> 0, the correct limit, verified finite
        # and monotone to +/-30000) but it warns on every batch, and log noise
        # masks real problems.
        pw = 1.0 / (1.0 + np.exp(np.clip(-(s - target_offset) / scale, -80.0, 80.0)))
        pl = 1.0 / (1.0 + np.exp(np.clip(-(-s - target_offset) / scale, -80.0, 80.0)))
        eval_t = 0.5 * (1.0 + pw - pl)
    else:
        eval_t = 1.0 / (1.0 + np.exp(-s / scale))
    out["target"] = (wdl * result + (1.0 - wdl) * eval_t).astype(np.float32)

    # --- SF-style distribution SHAPING (2026-07-29) -------------------------
    # Our filter is a pure accept/reject predicate with no shaping at all; SF's
    # loader carries three probabilistic mechanisms on top (study §2.3), and
    # they are REWEIGHTING, not discarding. Emitted here as a per-record weight
    # so a fixed-size batch pipeline gets the same effect in expectation as
    # their probabilistic skip.
    w = None

    # (a) PIECE-COUNT REBALANCING — SF PR #4942, "an attempt to reduce the
    # skewed dataset piece count distribution where there are much more
    # positions with less than 16 pieces". MEASURED on our mix: mean piece
    # count 15.6 against SF's target 19.5, and the 31-32 bucket gets 4.25% of
    # our data where SF targets 13.95% — a ~3x endgame skew. Alexandria
    # measured +5.08 Elo STC / +2.28 LTC for fixing exactly this, as a
    # data-loader change alone.
    if piece_rebalance:
        # popcount of the occupancy bitboard. NOT `.view(np.uint8)` — a
        # structured-array field is not contiguous and that raises.
        npc = np.bitwise_count(np.ascontiguousarray(recs["occ"])).astype(np.float32)
        # SF's 5-point spline over piece counts 0/8/16/24/32. The -0.20 at 0 is
        # a control point, not a probability — clamp after interpolating.
        knots = np.array([0.0, 8.0, 16.0, 24.0, 32.0], dtype=np.float32)
        vals = np.array([-0.20, 0.45, 1.00, 0.95, 0.75], dtype=np.float32)
        w = np.clip(np.interp(npc, knots, vals), 0.0, 1.0).astype(np.float32)

    # (b) WDL FILTERING — SF skips with probability 1 - score_result_prob(),
    # i.e. down-weights positions whose EVAL DISAGREES WITH THE GAME RESULT.
    # NOTE: this is OUR formulation of that shape, not a transcription — the
    # exact SF predicate lives in a GPL loader we do not copy, and its precise
    # constants are unverified here. We already have a calibrated draw-aware
    # model per corpus, so agreement is scored under it: `eval_t` is the
    # expected score implied by the eval, `result` is what happened.
    if wdl_filter:
        agree = 1.0 - np.abs(eval_t - result)
        aw = np.clip(agree, wdl_filter_floor, 1.0).astype(np.float32)
        w = aw if w is None else (w * aw).astype(np.float32)

    if w is not None:
        out["weight"] = w

    if policy:
        pf = recs["extra"][:, 0].astype(np.int64)
        pt = recs["extra"][:, 1].astype(np.int64)
        promo = recs["extra"][:, 2].astype(np.int64)
        has = (pf < 64) & (pt < 64) & (pf != pt)
        pid = policy_move_id(pf, pt, promo)
        out["pol_target_id"] = np.where(has, pid, -1).astype(np.int32)
    return out
