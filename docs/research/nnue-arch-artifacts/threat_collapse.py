#!/usr/bin/env python3
"""How much of the trained FullThreats table survives collapsing the relation?

FullThreats spends 60,720 rows x 1280 lanes (77.7 MB, 56% of the net) on the FULL
relation (attacker piece, from-square, victim piece, to-square). A much cheaper
encoding -- used by Vine -- folds "is this piece attacked / defended" into the
piece-square feature itself, costing a few thousand rows instead of 60,720.

The question this prices: how much of the LEARNED signal in our threat table is
explained by coarser groupings of the same rows?

Method: re-derive the exact index construction from src/eval/nnue/sota_features.rs
(empty-board attack geometry -> prefix-sum offsets -> dense to-ordinal), enumerate
every (attacker, from, victim, to) tuple forward to get index -> tuple, then run an
ANOVA over the 1280-dim rows for several groupings:

   R^2(grouping) = 1 - SS_within / SS_total     (rows treated as vectors)

NULL PANEL (rule 1, run first): the same statistic for a RANDOM grouping with the
same number of groups and the same group-size distribution. A grouping that scores
no better than a random grouping of equal cardinality explains nothing -- the
apparent R^2 of a fine partition is mostly a degrees-of-freedom artifact.

Pure numpy over the .cenn file. No engine, no search.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from net_anatomy import parse  # noqa: E402

RNG = np.random.default_rng(20260728)

# ---------------------------------------------------------------- geometry ---
FILES = lambda s: s & 7
RANKS = lambda s: s >> 3
ALL_PIECES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]   # sota_features.rs:150
NUM_VALID_TARGETS = [0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0]
MAP = [[0, 1, -1, 2, -1, -1],
       [0, 1, 2, 3, 4, -1],
       [0, 1, 2, 3, -1, -1],
       [0, 1, 2, 3, -1, -1],
       [0, 1, 2, 3, 4, -1],
       [-1, -1, -1, -1, -1, -1]]
THREAT_DIMS = 60720


def rays(sq, deltas):
    bb = 0
    f0, r0 = FILES(sq), RANKS(sq)
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            bb |= 1 << (r * 8 + f)
            f += df
            r += dr
    return bb


def steps(sq, deltas):
    bb = 0
    f0, r0 = FILES(sq), RANKS(sq)
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        if 0 <= f < 8 and 0 <= r < 8:
            bb |= 1 << (r * 8 + f)
    return bb


DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]
KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING = DIAG + ORTH


def geom(code, sq):
    """Empty-board attack set; pawns use push-or-attacks (sota_features.rs:225-236)."""
    t = code & 7
    white = code < 8
    if t == 1:
        d = 1 if white else -1
        return steps(sq, [(1, d), (-1, d), (0, d)])
    if t == 2:
        return steps(sq, KNIGHT)
    if t == 3:
        return rays(sq, DIAG)
    if t == 4:
        return rays(sq, ORTH)
    if t == 5:
        return rays(sq, DIAG + ORTH)
    if t == 6:
        return steps(sq, KING)
    return 0


def build_index_map():
    """-> arrays of length THREAT_DIMS: attacker code, from, victim code, to."""
    offsets = [[0] * 64 for _ in range(16)]
    piece_offset = [0] * 16
    cumulative_offset = [0] * 16
    lut2 = {}
    cumulative = 0
    for code in ALL_PIECES:
        is_pawn = (code & 7) == 1
        cum = 0
        for f in range(64):
            offsets[code][f] = cum
            bb = geom(code, f)
            lut2[(code, f)] = {t: bin(bb & ((1 << t) - 1)).count("1")
                               for t in range(64) if (bb >> t) & 1}
            if (not is_pawn) or (8 <= f <= 55):
                cum += bin(bb).count("1")
        piece_offset[code] = cum
        cumulative_offset[code] = cumulative
        cumulative += NUM_VALID_TARGETS[code] * cum
    assert cumulative == THREAT_DIMS, cumulative

    lut1 = {}
    for a in ALL_PIECES:
        for v in ALL_PIECES:
            at, vt = a & 7, v & 7
            m = MAP[at - 1][vt - 1]
            same_type_diff_color = (a ^ v) == 8
            semi = (at == vt) and (same_type_diff_color or at != 1)
            if m < 0:
                lut1[(a, v)] = (THREAT_DIMS, THREAT_DIMS)
                continue
            vc = v >> 3
            nt = NUM_VALID_TARGETS[a] // 2
            base = cumulative_offset[a] + (vc * nt + m) * piece_offset[a]
            lut1[(a, v)] = (THREAT_DIMS if semi else base, base)

    att = np.full(THREAT_DIMS, -1, np.int16)
    frm = np.full(THREAT_DIMS, -1, np.int16)
    vic = np.full(THREAT_DIMS, -1, np.int16)
    to = np.full(THREAT_DIMS, -1, np.int16)
    for a in ALL_PIECES:
        is_pawn = (a & 7) == 1
        for f in range(64):
            if is_pawn and not (8 <= f <= 55):
                continue
            for t, ordv in lut2[(a, f)].items():
                for v in ALL_PIECES:
                    base = lut1[(a, v)][0 if f < t else 1]
                    if base >= THREAT_DIMS:
                        continue
                    i = base + offsets[a][f] + ordv
                    if i < THREAT_DIMS:
                        att[i], frm[i], vic[i], to[i] = a, f, v, t
    return att, frm, vic, to


# --------------------------------------------------------------- statistic ---
def r2(W, g):
    """1 - SS_within/SS_total over vector rows, grouped by integer labels g."""
    n, d = W.shape
    mu = W.mean(0)
    sst = float(((W - mu) ** 2).sum())
    ng = int(g.max()) + 1
    cnt = np.bincount(g, minlength=ng).astype(np.float64)
    sums = np.zeros((ng, d), np.float64)
    np.add.at(sums, g, W.astype(np.float64))
    nz = cnt > 0
    gm = np.zeros_like(sums)
    gm[nz] = sums[nz] / cnt[nz, None]
    ssb = float((cnt[:, None] * (gm - mu) ** 2).sum())
    return ssb / sst, ng


def null_r2(W, g, reps=4):
    """Same statistic with the group LABELS shuffled -- same cardinality, same
    group sizes, no structure."""
    out = []
    for _ in range(reps):
        out.append(r2(W, RNG.permutation(g))[0])
    return float(np.mean(out))


def main(path):
    n = parse(path)
    assert n["threat_w"] is not None, "net has no threat table"
    W = n["threat_w"].astype(np.float32)
    att, frm, vic, to = build_index_map()
    live = att >= 0
    print(f"{path.rsplit('/',1)[-1]}: {W.shape[0]} threat rows, "
          f"{live.sum()} resolved by the forward enumeration "
          f"({100*live.mean():.1f}%)")
    W = W[live]
    a, f_, v, t = att[live], frm[live], vic[live], to[live]
    at, vt = (a & 7).astype(np.int64), (v & 7).astype(np.int64)
    enemy = ((a >> 3) != (v >> 3)).astype(np.int64)      # attacker/victim opposite colour
    vcol = (v >> 3).astype(np.int64)

    groupings = {
        "(victim type, victim square, is-enemy-attacker)  <- Vine-style flag": vt * 128 + t.astype(np.int64) * 2 + enemy,
        "(victim type, victim square, victim colour)":                          vt * 128 + t.astype(np.int64) * 2 + vcol,
        "(attacker type, victim type)":                                         at * 6 + vt,
        "(attacker type, victim type, victim square)":                          (at * 6 + vt) * 64 + t.astype(np.int64),
        "(attacker type, attacker square)":                                     at * 64 + f_.astype(np.int64),
        "(victim square only)":                                                 t.astype(np.int64),
        "(attacker square only)":                                              f_.astype(np.int64),
    }
    # norm-weighted view: restrict to the rows that actually carry mass
    nrm = np.linalg.norm(W, axis=1)
    big = nrm >= np.median(nrm)
    print(f"  row |w|_2: mean {nrm.mean():.2f}  median {np.median(nrm):.2f}  "
          f"p99 {np.percentile(nrm,99):.2f}")
    print(f"\n  {'grouping':62s} {'groups':>7s} {'R2':>7s} {'null':>7s} "
          f"{'R2-null':>8s} {'top-half R2':>11s}")
    for name, g in groupings.items():
        g = np.unique(g, return_inverse=True)[1]
        val, ng = r2(W, g)
        nul = null_r2(W, g)
        gb = np.unique(g[big], return_inverse=True)[1]
        vb, _ = r2(W[big], gb)
        nb = null_r2(W[big], gb)
        print(f"  {name:62s} {ng:7d} {100*val:6.1f}% {100*nul:6.1f}% "
              f"{100*(val-nul):7.1f}% {100*(vb-nb):10.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "engines/canonical/net.cenn")
