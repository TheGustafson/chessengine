#!/usr/bin/env python3
"""How much of the neck->L3 map's energy survives truncating the neck to k dims?

For each output bucket, SVD the L2 weight matrix W2 [L2 x L3] and report the
cumulative fraction of ||W2||_F^2 in the top-k singular directions. This prices
"shrink the neck to k" in the only currency available offline: how much of the
linear map from neck to L3 a rank-k neck could still express.

NULL: the same curve for an i.i.d. matrix of identical shape (resampled entries).
An interpretation is only licensed where obs and null separate.
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from net_anatomy import parse

RNG = np.random.default_rng(20260728)

def curve(W):
    sv = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    e = sv**2
    return np.cumsum(e)/e.sum()

for p in sys.argv[1:]:
    n = parse(p)
    l2, l3, ob = n['l2'], n['l3'], n['buckets']
    obs = np.mean([curve(n['l2_w'][b]) for b in range(ob)], axis=0)
    flat = n['l2_w'].ravel()
    null = np.mean([curve(RNG.choice(flat, size=(l2,l3), replace=True)) for _ in range(32)], axis=0)
    ks = [k for k in (2,4,6,8,10,12,16,24,32,48,64,96,128) if k <= min(l2,l3)]
    print(f"{p.rsplit('/',1)[-1]:42s} L2={l2:4d} L3={l3}")
    print("   k      :", "  ".join(f"{k:5d}" for k in ks))
    print("   obs %  :", "  ".join(f"{100*obs[k-1]:5.1f}" for k in ks))
    print("   null % :", "  ".join(f"{100*null[k-1]:5.1f}" for k in ks))
