"""Tier-0 pre-screen for the REFINEMENT direction (richer-representation study):
the joint (threat-tuple x king-region-bit) census.

The compression POC established that the threat code is compressible but that
compression cannot buy Elo (fold work = active_features x ACC, unchanged).
The refinement question is the opposite direction: can a threat tuple's row be
SPLIT by a slow king-context bit, at constant total rows, funded by merging
rows the data never fires? This census answers the necessary conditions:

  1. CO-OCCUPANCY - for each candidate context bit b and tuple t, the joint
     fire counts n(t, b=0), n(t, b=1). A split only helps if hot tuples
     actually occur under BOTH values of the bit (else the conditioning is
     degenerate and the split learns nothing the unconditioned row lacks).
  2. THE DEAD-ROW BUDGET - rows that never fire on the training mix NOR the
     heldout pack. Merging those funds splits at zero projected eval cost on
     the distributions we can measure (games remain the blunder-tail guard).
  3. MI(t; b | fired) per bit - an upper-bound-flavored summary of how much
     information the bit adds to the code (NOT sufficient for value - the
     gradient probe carries that - but a cheap ranking + a hard zero test).

Region bits are defined per PERSPECTIVE in that perspective's own frame,
exactly as the engine would compute them from (perspective, ksq, enemy ksq):
  adv    own king off its back two ranks (rel_rank >= 2)
  rank1  own king off its back rank      (rel_rank >= 1)
  fpair  own king on the CENTER file pair of its mirror half (oriented file
         c/d, i.e. one bit finer than the existing d/e mirror)
  esame  enemy king on the SAME file-half as own king (attack-race bit)
  eadv   enemy king off ITS back two ranks
  null   content-free position-hash bit (the NULL for every statistic here)

Mix-weighted sample over the D48X training mix + a full pass of the heldout
pack for the never-fire union. CPU-only numpy, nice'd, safe beside the GPU.
"""
import sys
from pathlib import Path

import numpy as np

# THIS checkout's jax/ (region_bit lives beside the code2d loader seam).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
from archon_jax.data import BulletBin                       # noqa: E402
from archon_jax.features import (DecodedBatch, REGION_BIT_NAMES,  # noqa: E402
                                 region_bit, threat_batch)
from archon_jax.spec import THREAT_DIMS                     # noqa: E402

ROOT = "/home/nick/projects/chessengine"
MIX = [  # the D48X / recode-t1 training mix (recode_t1_kmeans8k.json)
    ("uho-band75-100.shuf.bin", 0.251234),
    ("fishpack32.shuf.bin", 0.158256),
    ("farseerT74-band75-100.shuf.bin", 0.121036),
    ("datapv2-band75-88.shuf.bin", 0.097139),
    ("farseerT75-band75-83.shuf.bin", 0.090499),
    ("multinet-band75-88.shuf.bin", 0.089567),
    ("t60t70-band75-85.shuf.bin", 0.082276),
    ("pylon-band75-88.shuf.bin", 0.046534),
    ("wisr-band75-95.shuf.bin", 0.034833),
    ("wrongnnue-band75-95.shuf.bin", 0.025796),
    ("canon-tight-frame.shuf.bin", 0.002701),
    ("hardneg-frame.shuf.bin", 0.000129),
]
HELDOUT = f"{ROOT}/data/nnue/heldout/uho-band0-5.bin"
N_TOTAL = 8_000_000
CHUNK = 65_536
BITS = list(REGION_BIT_NAMES)
OUT = f"{ROOT}/data/nnue/nets/screens/recode/region-census-mix8M.npz"


def region_bits(dec: DecodedBatch, persp: int) -> np.ndarray:
    """(NBITS, N) int64 region bits for one perspective.

    Thin stack over the CANONICAL per-bit definitions in
    ``archon_jax.features.region_bit`` — the census, the probe, and the
    training-time code2d path all read the same arithmetic, so they cannot
    drift. (The defs were born here and moved there verbatim when the
    trainer seam landed; the saved census artifacts predate the move but are
    bit-identical.)
    """
    return np.stack([region_bit(dec, persp, n) for n in BITS])


def accumulate(recs, joint, fired):
    dec = DecodedBatch(recs)
    th = threat_batch(dec)                       # (N, 2, 160), -1 padded
    for persp in (0, 1):
        bits = region_bits(dec, persp)           # (NBITS, N)
        t = th[:, persp, :]
        valid = t >= 0
        rows_idx, slot = np.nonzero(valid)
        tv = t[rows_idx, slot].astype(np.int64)
        fired[tv] = True
        for b in range(len(BITS)):
            bv = bits[b][rows_idx]
            counts = np.bincount(tv + bv * THREAT_DIMS, minlength=2 * THREAT_DIMS)
            joint[b] += counts.reshape(2, THREAT_DIMS).T


def main():
    joint = np.zeros((len(BITS), THREAT_DIMS, 2), dtype=np.int64)
    fired_mix = np.zeros(THREAT_DIMS, dtype=bool)
    done_total = 0
    for name, w in MIX:
        bin_ = BulletBin(f"{ROOT}/data/nnue/jaxcorpora/{name}")
        n = min(int(round(N_TOTAL * w)), len(bin_))
        done = 0
        while done < n:
            recs = bin_.slice(done, min(CHUNK, n - done))
            accumulate(recs, joint, fired_mix)
            done += len(recs)
        done_total += done
        print(f"pack {name}: {done} positions", flush=True)

    fired_held = np.zeros(THREAT_DIMS, dtype=bool)
    joint_held = np.zeros((len(BITS), THREAT_DIMS, 2), dtype=np.int64)
    bin_ = BulletBin(HELDOUT)
    done = 0
    while done < len(bin_):
        recs = bin_.slice(done, min(CHUNK, len(bin_) - done))
        accumulate(recs, joint_held, fired_held)
        done += len(recs)
    print(f"heldout: {done} positions", flush=True)

    np.savez(OUT, joint=joint, joint_held=joint_held, fired_mix=fired_mix,
             fired_held=fired_held, bits=np.array(BITS), n_mix=done_total,
             n_held=done)

    # ---- summary ----------------------------------------------------------
    tot = joint[0].sum(axis=1)                    # per-tuple mass (bit 0 view)
    mass = tot / tot.sum()
    order = np.argsort(tot)[::-1]
    dead = ~(fired_mix | fired_held)
    print(f"\npositions: mix={done_total}  heldout={done}")
    print(f"never-fired (mix sample U heldout): {int(dead.sum())}/{THREAT_DIMS} "
          f"({100.0 * dead.sum() / THREAT_DIMS:.1f}%) <- the split budget")
    print(f"fired mix only: {int((fired_mix & ~fired_held).sum())}, "
          f"heldout only: {int((fired_held & ~fired_mix).sum())}")
    for b, bname in enumerate(BITS):
        j = joint[b].astype(np.float64)
        n0, n1 = j[:, 0], j[:, 1]
        nt = n0 + n1
        p1 = n1.sum() / nt.sum()
        # MI(t; b | fired), in bits
        pt = nt / nt.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pb = np.stack([n0 / nt.sum(), n1 / nt.sum()], axis=1)
            marg = np.array([1 - p1, p1])
            mi = np.nansum(pb * np.log2(pb / (pt[:, None] * marg[None, :])))
        hb = -(p1 * np.log2(max(p1, 1e-12))
               + (1 - p1) * np.log2(max(1 - p1, 1e-12)))
        line = (f"bit {bname:6s} P(b=1)={p1:.3f} H(b)={hb:.3f} "
                f"MI(t;b)={mi:.4f} bits")
        for tau in (30, 100, 300):
            split_ok = (np.minimum(n0, n1) >= tau)
            line += (f" | tau={tau}: {int(split_ok.sum())} tuples, "
                     f"{100.0 * mass[split_ok].sum():.1f}% mass")
        print(line)
        # top-8192 view: how balanced is the bit among the hot tuples?
        hot = order[:8192]
        bal = np.minimum(n0[hot], n1[hot]) / np.maximum(nt[hot], 1)
        print(f"    top-8192 tuples: median min-cell share {np.median(bal):.3f}, "
              f"25th pct {np.percentile(bal, 25):.3f}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
