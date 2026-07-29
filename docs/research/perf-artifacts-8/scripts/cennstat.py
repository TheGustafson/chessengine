#!/usr/bin/env python3
"""perf study 8 — read a CENN-v2 net and report the FT weight distributions.

    python3 cennstat.py <net.cenn>

The question this exists to answer: the threat transformer is quantised at
THREAT_QUANT = 64 while HalfKA is at FT_QUANT = 255, and that mismatch costs
~5 SIMD ops per merge in `activate` (perf study 8 §6.1).  Rescaling the stored
threat weights to the 255 frame — a pure file conversion, NOT a retrain — is
only possible if `round(w * 255/64)` still fits in i8, i.e. if
`max|w| <= 31` (31*255/64 = 123.5 -> 124; 32 -> 127.5 -> 128, which overflows).

So: print the max, the tail, and exactly how many weights would clip.
"""

import struct
import sys
from collections import Counter

HEADER = 64
HALFKA_DIMS = 22528
THREAT_DIMS = 60720

path = sys.argv[1]
raw = open(path, "rb").read()
assert raw[:4] == b"CEN2", raw[:4]
(version, flags) = struct.unpack_from("<HH", raw, 4)
(hk, th, acc, l2, l3, buckets) = struct.unpack_from("<6I", raw, 8)
(ftq, l1q, ftshift, scale) = struct.unpack_from("<4i", raw, 32)
print(f"version={version} flags={flags} halfka={hk} threat={th} acc={acc} "
      f"L2={l2} L3={l3} buckets={buckets}")
print(f"FT_QUANT={ftq} L1_QUANT={l1q} FT_SHIFT={ftshift} network_scale={scale}")

off = HEADER
n_halfka = hk * acc
halfka = memoryview(raw)[off:off + n_halfka * 2].cast("h")
off += n_halfka * 2
bias = memoryview(raw)[off:off + acc * 2].cast("h")
off += acc * 2
n_threat = th * acc
threat = memoryview(raw)[off:off + n_threat].cast("b")
off += n_threat

def stats(name, mv, scale_to=None):
    lo = hi = 0
    hist = Counter()
    nz = 0
    for v in mv:
        if v:
            nz += 1
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        hist[abs(v)] += 1
    n = len(mv)
    print(f"\n{name}: n={n:,}  min={lo}  max={hi}  nonzero={nz / n * 100:.1f}%")
    mag = max(abs(lo), abs(hi))
    print(f"  max |w| = {mag}")
    if scale_to is not None:
        num, den = scale_to
        clip = sum(c for m, c in hist.items() if round(m * num / den) > 127)
        print(f"  rescaling by {num}/{den}: {clip:,} of {n:,} weights would clip "
              f"i8 ({clip / n * 100:.4f}%)")
        # what fraction of the MASS is in the clipping weights
        print(f"  largest |w| that survives: {int(127 * den / num)}")
    # tail
    tail = sorted(hist.items())[-8:]
    print(f"  |w| tail (magnitude: count): {tail}")

stats("HalfKA FT (i16, quant 255)", halfka)
stats("FT bias (i16)", bias)
stats("Threat FT (i8, quant 64)", threat, scale_to=(255, 64))
