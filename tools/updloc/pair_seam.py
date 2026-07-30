#!/usr/bin/env python3
"""Learned-representation study: the POSITION-SUCCESSOR PAIR seam, validated.

Every binpack-derived record already carries the move played next in the game
(`extra[0..2]`), and the D46 quiet filter guarantees it is a Normal,
non-capture, non-promotion move — so the successor position is a pure piece
relocation synthesizable in numpy with no movegen. This tool:

  1. synthesizes the successor RECORD (occ / pcs / ksq / opp_ksq, in the
     opponent's canonical frame) — the loader-side derivation a Tier-1
     update-locality penalty needs (data-recipe-contract exposure: none — no
     new corpus, no new filter site, no labels claimed for the successor);
  2. validates it (mover on from-square, empty destination, king bookkeeping);
  3. runs BOTH positions through the trainer's own index mappers
     (`halfka_batch` / `threat_batch`, index-exact vs the Rust sources) and
     measures the DATA-side update locality of the incumbent feature set:
     rows changed per played move, per perspective, split by king-mover —
     the training-distribution complement of the in-tree foldcount census.

Perspective alignment across the move: the successor record is stm-flipped,
so physical player P = perspective 0 at t and perspective 1 at t+1. The
mappers canonicalize internally, so indices are frame-invariant per
(position, player); the delta distribution itself validates the alignment
(a wrong frame would show ~full-set deltas on every pair, not 2-row ones).

Usage:
  .venv/bin/python tools/updloc/pair_seam.py [heldout.bin] [n-records]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
from archon_jax.features import (DecodedBatch, canonical_orient,  # noqa: E402
                                 halfka_batch, threat_batch)
from archon_jax.spec import RECORD_DTYPE, THREAT_DIMS, HALFKA_DIMS  # noqa: E402

MAIN = Path("/home/nick/projects/chessengine")


def synthesize_successor(recs):
    """Successor records for Normal non-capture moves; returns (succ, valid,
    king_move) with succ in the opponent's canonical frame."""
    n = len(recs)
    pf = recs["extra"][:, 0].astype(np.int64)
    pt_sq = recs["extra"][:, 1].astype(np.int64)
    promo = recs["extra"][:, 2].astype(np.int64)
    m = canonical_orient(recs["ksq"].astype(np.int64))
    pf_b, pt_b = pf ^ m, pt_sq ^ m

    dec = DecodedBatch(recs)
    mailbox = dec.mailbox_code.copy()  # (N, 64) 0=empty, (color<<3)|(pt+1)

    valid = (pf < 64) & (pt_sq < 64) & (pf != pt_sq) & (promo == 0)
    mover = mailbox[np.arange(n), pf_b]
    valid &= (mover != 0) & ((mover >> 3) == 0)          # stm piece on from
    valid &= mailbox[np.arange(n), pt_b] == 0            # empty destination
    king_move = valid & (mover == 6)                     # stm king code 0b0110

    # apply the relocation, then flip into the opponent's frame
    mb = mailbox.copy()
    mb[np.arange(n), pt_b] = mover
    mb[np.arange(n), pf_b] = 0
    flipped = np.zeros_like(mb)
    flipped[:, np.arange(64) ^ 56] = np.where(mb != 0, mb ^ 8, 0)

    occ2 = (flipped != 0)
    succ = np.zeros(n, dtype=RECORD_DTYPE)
    succ["occ"] = np.packbits(occ2, axis=1, bitorder="little").view(
        np.uint64).reshape(n)
    # nibble list in ascending-square (= occupancy) order, vectorized:
    rows, sqs = np.nonzero(occ2)
    counts = occ2.sum(axis=1)
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    ordinal = np.arange(len(rows)) - starts[rows]
    nib = (flipped[rows, sqs] - 1) & 0xF  # code-1: (color<<3)|pt exactly
    pcs = np.zeros((n, 32), dtype=np.uint8)
    pcs[rows, ordinal] = nib
    succ["pcs"] = (pcs[:, 0::2] | (pcs[:, 1::2] << 4))
    # Field conventions (verified against decoded boards): ksq = stm king in
    # the RECORD frame; opp_ksq = enemy king in ITS OWN frame (i.e. the
    # record-frame enemy king ^ 56).
    succ["ksq"] = (dec.kings[:, 1] ^ 56).astype(np.uint8)
    own_king_after = np.where(king_move, pt_b, dec.kings[:, 0])
    succ["opp_ksq"] = own_king_after.astype(np.uint8)
    succ["extra"][:, :2] = 64  # no packed move for the successor
    succ["result"] = 2 - recs["result"]
    return succ, valid, king_move


def sym_diff_per_group(idx_a, idx_b, dims):
    """Symmetric-difference sizes between two (N, 2, cap) -1-padded index
    arrays, aligned so perspective p in `a` pairs with 1-p in `b`."""
    n = idx_a.shape[0]

    def keys(ix, swap):
        r, p, _ = np.nonzero(ix >= 0)
        v = ix[ix >= 0].astype(np.int64)
        pp = (1 - p) if swap else p
        return (np.int64(r) * 2 + pp) * dims + v

    ka, kb = keys(idx_a, False), keys(idx_b, True)
    allk = np.concatenate([ka, kb])
    uniq, counts = np.unique(allk, return_counts=True)
    singles = uniq[counts == 1]
    group = singles // dims
    return np.bincount(group, minlength=n * 2).reshape(n, 2)


def stats(name, d, sel):
    x = d[sel] if sel is not None else d.ravel()
    if len(x) == 0:
        print(f"  {name:34s} (empty)")
        return
    q = np.percentile(x, [50, 90, 99])
    print(f"  {name:34s} mean {x.mean():6.3f}  p50 {q[0]:4.0f}  "
          f"p90 {q[1]:4.0f}  p99 {q[2]:4.0f}  max {x.max():4d}")


def main():
    heldout = sys.argv[1] if len(sys.argv) > 1 else str(
        MAIN / "data/nnue/heldout/uho-band0-5.bin")
    n_rec = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    recs = np.fromfile(heldout, dtype=RECORD_DTYPE, count=n_rec)

    succ, valid, king_move = synthesize_successor(recs)
    print(f"records={len(recs):,}  valid pairs={valid.sum():,} "
          f"({valid.mean():.2%})  king-mover pairs={king_move.sum():,} "
          f"({king_move.sum()/max(valid.sum(),1):.2%} of valid)")
    recs, succ, king_move = recs[valid], succ[valid], king_move[valid]

    dec_t, dec_s = DecodedBatch(recs), DecodedBatch(succ)
    # bookkeeping self-checks on the synthesized records
    assert (dec_s.kings[:, 0] == succ["ksq"]).all()
    assert (dec_s.kings[:, 1] == (succ["opp_ksq"] ^ 56)).all()
    assert (dec_s.occ == succ["occ"]).all()

    h_t, h_s = halfka_batch(dec_t), halfka_batch(dec_s)
    t_t, t_s = threat_batch(dec_t), threat_batch(dec_s)

    dh = sym_diff_per_group(h_t, h_s, HALFKA_DIMS)
    dt = sym_diff_per_group(t_t, t_s, THREAT_DIMS)

    act_h = (h_t >= 0).sum(axis=2)
    act_t = (t_t >= 0).sum(axis=2)
    print(f"\nactive rows/perspective (data-side): "
          f"halfka {act_h.mean():.2f}  threat {act_t.mean():.2f}")

    km = king_move
    print("\n== data-side rows changed per played move, per perspective ==")
    print("-- HalfKA --")
    stats("all pairs, both persp", dh, None)
    stats("non-king moves", dh, np.ix_(~km, [0, 1])[0])
    stats("king moves, mover persp", dh[:, 0], km)
    stats("king moves, opponent persp", dh[:, 1], km)
    print("-- FullThreats --")
    stats("all pairs, both persp", dt, None)
    stats("non-king moves", dt, np.ix_(~km, [0, 1])[0])
    stats("king moves, mover persp", dt[:, 0], km)
    stats("king moves, opponent persp", dt[:, 1], km)

    both = dh + dt
    print(f"\ntotal rows changed/move (both channels, both persp): "
          f"mean {both.sum(axis=1).mean():.2f}")


if __name__ == "__main__":
    main()
