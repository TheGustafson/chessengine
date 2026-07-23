"""Feature-math parity tests: pin the hand-computed vectors from the Rust test
modules (sota_features.rs tests, policy_vocab.rs tests) against the vectorized
numpy implementation, plus a slow-scalar cross-check of the threat batch path.

Run:  cd jax && JAX_PLATFORMS=cpu uv run --no-project --with numpy --with pytest \
        -- python -m pytest tests/test_features.py -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archon_jax import features as F  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, THREAT_DIMS  # noqa: E402

# piece type codes used below: 0=P 1=N 2=B 3=R 4=Q 5=K; color 0=White(stm)


def make_record(pieces, score=0, result=1, ksq=None, extra=(64, 64, 0)):
    """Build a bulletformat record from [(sq, color, pt), ...]."""
    rec = np.zeros(1, dtype=RECORD_DTYPE)
    occ = 0
    by_sq = sorted(pieces)
    for i, (sq, color, pt) in enumerate(by_sq):
        occ |= 1 << sq
        nib = (color << 3) | pt
        rec["pcs"][0][i // 2] |= nib << (4 * (i % 2))
    rec["occ"] = occ
    rec["score"] = score
    rec["result"] = result
    if ksq is None:
        ksq = next(sq for sq, c, pt in by_sq if c == 0 and pt == 5)
    rec["ksq"] = ksq
    rec["extra"][0] = extra
    return rec


def halfka_of(persp, piece_color, piece_pt, sq, ksq):
    """Single-feature probe through the batch path."""
    # Put kings for both sides; if the probed piece IS a king reuse it.
    pieces = [(sq, piece_color, piece_pt)]
    if not (piece_pt == 5 and piece_color == 0):
        pieces.append((ksq, 0, 5) if persp == 0 else (ksq, 0, 5))
    # ensure a black king exists somewhere out of the way
    if not any(c == 1 and pt == 5 for _, c, pt in pieces):
        spare = next(s for s in (63, 62, 55) if s not in [p[0] for p in pieces])
        pieces.append((spare, 1, 5))
    dec = F.DecodedBatch(make_record(pieces))
    # override king square for the probed perspective (tests use hypothetical kings)
    dec.kings[0, persp] = ksq
    hk = F.halfka_batch(dec)
    # find the probed piece's ordinal
    for i, (s, c, pt) in enumerate(sorted(pieces)):
        if (s, c, pt) == (sq, piece_color, piece_pt):
            return int(hk[0, persp, i])
    raise AssertionError


def test_halfka_hand_computed():
    # sota_features.rs halfka_hand_computed (lines 530-551)
    assert halfka_of(0, 0, 5, 4, 4) == 4 + 640 + 31 * 704       # WK on e1, own king
    assert halfka_of(0, 0, 0, 0, 0) == 19719                     # WP a1, WK a1
    # Black perspective: WHITE pawn a2 (sq 8), BLACK king e8 (sq 60)
    pieces = [(8, 0, 0), (4, 0, 5), (60, 1, 5)]
    dec = F.DecodedBatch(make_record(pieces))
    hk = F.halfka_batch(dec)
    ordinal = sorted(pieces).index((8, 0, 0))
    assert int(hk[0, 1, ordinal]) == 21936


def test_halfka_mirror_symmetry():
    # sota_features.rs halfka_mirror_symmetry: WN c1 with WK b1 == WN f1 with WK g1
    qs = halfka_of(0, 0, 1, 2, 1)
    ks = halfka_of(0, 0, 1, 5, 6)
    assert qs == ks
    assert halfka_of(0, 0, 1, 2, 1) != halfka_of(0, 0, 1, 3, 1)


def test_threat_tables_geometry():
    # sota_features.rs threats_geometry_and_tiling
    for base in (0, 8):
        assert F.THREAT_PIECE_OFFSET[base + 1] == 132
        assert F.THREAT_PIECE_OFFSET[base + 2] == 336
        assert F.THREAT_PIECE_OFFSET[base + 3] == 560
        assert F.THREAT_PIECE_OFFSET[base + 4] == 896
        assert F.THREAT_PIECE_OFFSET[base + 5] == 1456
        assert F.THREAT_PIECE_OFFSET[base + 6] == 420
    last = 14  # B_KING: NUM_VALID_TARGETS = 0, so its block base == Dimensions
    end = (F.THREAT_CUM_OFFSET[last]
           + F.NUM_VALID_TARGETS[last] * F.THREAT_PIECE_OFFSET[last])
    assert end == THREAT_DIMS


def test_threat_hand_computed():
    # sota_features.rs threats_hand_computed: WN b1 -> BP c3, WK a1, White view
    # = lut1 + offsets + lut2 = 2472 + 2 + 2 = 2476.
    pieces = [(1, 0, 1), (18, 1, 0), (0, 0, 5), (63, 1, 5)]
    dec = F.DecodedBatch(make_record(pieces))
    th = F.threat_batch(dec)
    white_set = set(th[0, 0][th[0, 0] >= 0].tolist())
    assert 2476 in white_set, white_set


def test_threat_exclusions_via_sets():
    # rook->rook symmetric threat: exactly ONE direction survives per perspective
    # (sota_features.rs threats_exclusions); kings never attack.
    pieces = [(0, 0, 3), (56, 1, 3), (4, 0, 5), (60, 1, 5)]  # WRa1 BRa8 WKe1 BKe8
    dec = F.DecodedBatch(make_record(pieces))
    th = F.threat_batch(dec)
    for persp in (0, 1):
        s = th[0, persp][th[0, persp] >= 0]
        assert len(s) == 1, s  # one direction of the rook pair, nothing else


def slow_threat_index(persp, att_color, att_pt, frm, to, atd_color, atd_pt, ksq):
    """Direct scalar transcription of sota_features.rs threat_index (324-348)."""
    orientation = (0 if (ksq & 7) <= 3 else 7) ^ (56 * persp)
    from_o = frm ^ orientation
    to_o = to ^ orientation
    swap = 8 * persp
    att = ((att_color << 3) | (att_pt + 1)) ^ swap
    atd = ((atd_color << 3) | (atd_pt + 1)) ^ swap
    d = 1 if from_o < to_o else 0
    base = F.THREAT_LUT1[att, atd, d]
    if base >= THREAT_DIMS:
        return None
    return int(base + F.THREAT_OFFSETS[att, from_o] + F.THREAT_LUT2[att, from_o, to_o])


def slow_for_each_threat(dec, i, persp):
    """Scalar transcription of for_each_threat (sota 376-434) for record i."""
    out = []
    ksq = int(dec.kings[i, persp])
    pieces = [(int(s), int(c), int(p)) for r, s, c, p in
              zip(dec.row, dec.sq, dec.color, dec.pt) if r == i]
    occ = int(dec.occ[i])
    mb = {s: (c, p) for s, c, p in pieces}
    both = lambda pt: [s for s, c, p in pieces if p == pt]
    pawn_t = set(both(0)) | set(both(1)) | set(both(3))
    minor_t = pawn_t | set(both(2))
    queen_t = minor_t | set(both(4))
    for s, c, p in pieces:
        if p == 5:
            continue
        if p == 0:
            targets = [t for t in _bits(int(F.PAWN_ATTACKS[c][s])) if t in pawn_t]
            push = s + 8 if c == 0 else s - 8
            if 0 <= push < 64 and push in set(both(0)):
                targets.append(push)
        elif p == 1:
            targets = [t for t in _bits(int(F.KNIGHT_ATTACKS[s])) if t in queen_t]
        else:
            dirs = {2: F.BISHOP_DIRS, 3: F.ROOK_DIRS,
                    4: F.BISHOP_DIRS + F.ROOK_DIRS}[p]
            att = int(F.slider_attacks(dirs, np.array([s]), np.array([occ], dtype=np.uint64))[0])
            tgt = queen_t if p in (1, 4) else minor_t
            targets = [t for t in _bits(att) if t in tgt]
        for t in targets:
            vc, vp = mb[t]
            idx = slow_threat_index(persp, c, p, s, t, vc, vp, ksq)
            if idx is not None:
                out.append(idx)
    return sorted(out)


def _bits(x):
    out = []
    while x:
        out.append((x & -x).bit_length() - 1)
        x &= x - 1
    return out


def test_threat_batch_matches_slow_scalar_on_random_boards():
    rng = np.random.default_rng(1234)
    recs = []
    for _ in range(40):
        squares = rng.permutation(64)[: rng.integers(6, 28)]
        pieces = [(int(squares[0]), 0, 5), (int(squares[1]), 1, 5)]
        for s in squares[2:]:
            c = int(rng.integers(0, 2))
            pt = int(rng.choice([0, 0, 0, 1, 2, 3, 4]))
            r = s >> 3
            if pt == 0 and r in (0, 7):
                pt = 1
            pieces.append((int(s), c, pt))
        recs.append(make_record(pieces))
    recs = np.concatenate(recs)
    dec = F.DecodedBatch(recs)
    th = F.threat_batch(dec)
    for i in range(len(recs)):
        for persp in (0, 1):
            fast = sorted(th[i, persp][th[i, persp] >= 0].tolist())
            slow = slow_for_each_threat(dec, i, persp)
            assert fast == slow, (i, persp, fast, slow)
            assert len(fast) == len(set(fast))


def test_policy_vocab_vectors():
    # policy_vocab.rs tests
    assert F.GEOM_TABLE[1] == 0                        # a1->b1 first class
    pid = F.policy_move_id
    assert pid(np.array([48]), np.array([56]), np.array([1]))[0] == 1792
    assert pid(np.array([48]), np.array([56]), np.array([2]))[0] == 1793
    assert pid(np.array([48]), np.array([56]), np.array([3]))[0] == 1794
    assert pid(np.array([55]), np.array([63]), np.array([3]))[0] == 1857
    # queen promo rides the geometric class
    q = pid(np.array([48]), np.array([56]), np.array([0]))[0]
    assert 0 <= q < 1792
    # e1g1 castling geometry; knight b1c3; non-geometric a1->b4 = -1
    assert pid(np.array([4]), np.array([6]), np.array([0]))[0] >= 0
    assert pid(np.array([1]), np.array([18]), np.array([0]))[0] >= 0
    assert pid(np.array([0]), np.array([25]), np.array([0]))[0] == -1
    # bijection
    ids = F.GEOM_TABLE[F.GEOM_TABLE >= 0]
    assert len(ids) == 1792 and len(np.unique(ids)) == 1792


def test_startpos_counts():
    # startpos: 32 halfka features, both perspectives; buckets from popcount
    back = [3, 1, 2, 4, 5, 2, 1, 3]
    pieces = [(i, 0, back[i]) for i in range(8)]
    pieces += [(8 + i, 0, 0) for i in range(8)]
    pieces += [(48 + i, 1, 0) for i in range(8)]
    pieces += [(56 + i, 1, back[i]) for i in range(8)]
    rec = make_record(pieces)
    from archon_jax.features import map_batch
    m = map_batch(rec, wdl=0.3, score_multiplier=1.0, scale=400.0, policy=True)
    assert (m["hk_stm"] >= 0).sum() == 32
    assert (m["hk_ntm"] >= 0).sum() == 32
    assert m["bucket"][0] == 7
    n_stm = (m["th_stm"] >= 0).sum()
    n_ntm = (m["th_ntm"] >= 0).sum()
    assert n_stm == n_ntm and 0 < n_stm <= 160
    assert m["pol_target_id"][0] == -1  # extra = (64, 64, 0) -> no packed move


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
