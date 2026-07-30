#!/usr/bin/env python3
"""Learned-representation study: value mini-census for the LEARNED `pocket`
king-context bit (own king in {g1, h1, h2}, own-frame — the min-flip
30%-data-mass partition of the measured in-tree king transition matrix).

Computes, on the heldout pack, the same necessary-condition statistics the
richer-representation study's full census used to admit its hand bits:
P(b=1), MI(threat-tuple; bit) in bits, tuples with both cells >= tau, and
the median minority-cell share of the top-K tuples — for `pocket`, `rank1`
(the predecessor's placeholder), and `null` (position-hash control).

CPU-only numpy. Usage:
  .venv/bin/python tools/updloc/pocket_value_census.py [heldout.bin] [n]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
from archon_jax.features import DecodedBatch, threat_batch  # noqa: E402
from archon_jax.spec import RECORD_DTYPE, THREAT_DIMS  # noqa: E402

MAIN = Path("/home/nick/projects/chessengine")
POCKET = {6, 7, 15}  # g1, h1, h2 own-frame


def bit_value(dec, persp, name):
    own = dec.kings[:, persp] ^ (56 * persp)  # own-frame
    if name == "pocket":
        return np.isin(own, list(POCKET)).astype(np.int64)
    if name == "rank1":
        return ((own >> 3) >= 1).astype(np.int64)
    if name == "null":
        h = dec.occ * np.uint64(0x9E3779B97F4A7C15)
        return ((h >> np.uint64(17)) & np.uint64(1)).astype(np.int64)
    raise ValueError(name)


def main():
    heldout = sys.argv[1] if len(sys.argv) > 1 else str(
        MAIN / "data/nnue/heldout/uho-band0-5.bin")
    n_rec = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    bits = ("pocket", "rank1", "null")
    joint = {b: np.zeros((THREAT_DIMS, 2), dtype=np.int64) for b in bits}
    nb1 = {b: 0 for b in bits}
    total = 0

    chunk = 131_072
    with open(heldout, "rb") as f:
        remaining = n_rec
        while remaining > 0:
            recs = np.fromfile(f, dtype=RECORD_DTYPE,
                               count=min(chunk, remaining))
            if len(recs) == 0:
                break
            remaining -= len(recs)
            dec = DecodedBatch(recs)
            tb = threat_batch(dec)  # (N, 2, cap)
            for persp in (0, 1):
                idx = tb[:, persp, :]
                r, _ = np.nonzero(idx >= 0)
                v = idx[idx >= 0].astype(np.int64)
                for b in bits:
                    bv = bit_value(dec, persp, b)
                    np.add.at(joint[b], (v, bv[r]), 1)
                    nb1[b] += int(bv.sum())
            total += 2 * len(recs)

    print(f"samples (record x perspective): {total:,}")
    for b in bits:
        j = joint[b].astype(float)
        fires = j.sum()
        p1 = nb1[b] / total
        # MI(t; b) over the joint fire distribution, as region_census does
        pt = j.sum(axis=1) / fires
        pb = np.array([1 - nb1[b] / total, nb1[b] / total])
        pj = j / fires
        # renormalise pb to the fire distribution for consistency
        pb_f = j.sum(axis=0) / fires
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(pj * np.log2(pj / (pt[:, None] * pb_f[None, :])))
        both100 = int(((j[:, 0] >= 100) & (j[:, 1] >= 100)).sum())
        mass_both = j[(j[:, 0] >= 100) & (j[:, 1] >= 100)].sum() / fires
        top = np.argsort(j.sum(axis=1))[::-1][:8192]
        mins = np.minimum(j[top, 0], j[top, 1])
        share = np.median(mins / np.maximum(j[top].sum(axis=1), 1))
        print(f"  {b:8s} P(b=1)={p1:.3f}  MI={mi:.4f} bits  "
              f"tuples both cells>=100: {both100:,} ({mass_both:.1%} mass)  "
              f"top-8192 median min-cell share: {share:.3f}")


if __name__ == "__main__":
    main()
