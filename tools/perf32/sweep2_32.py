#!/usr/bin/env python3
"""perf study 32 -- second sweep: per-column INTEGER MSE-optimal scales, and
empirical bias correction.

Why integer and not fractional: the merge has to reconstruct the 255-frame
value from the reduced-frame accumulator, and the only single-uop way to do
that in i16 lanes is `vpmullw` by a per-column integer (the product is exactly
today's accumulator value, so it has today's overflow behaviour and no more).
A fractional scale m/2^k would need `pst*m` to survive in i16 BEFORE the shift,
which it does not -- it needs 32-bit lanes and doubles the merge stage.

Bias correction is free at runtime: the mean accumulator error a column picks up
is a constant offset, so it folds into ft_bias.  The marginal feature
activation rates come from the real position set rather than being assumed
uniform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cennq  # noqa: E402

c = cennq.load("engines/canonical/net.cenn")
w = c.halfka().astype(np.float64)
den = np.sqrt((w**2).mean())
aw = np.abs(w)
colmax = aw.max(axis=0)


def rel(wq):
    return np.sqrt(((w - wq) ** 2).mean()) / den


def deq(s):
    return np.clip(np.rint(w / s), -127, 127) * s


print("=== per-column INTEGER scale, chosen by column MSE (clipping allowed) ===")
smax = int(np.ceil(colmax.max() / 127))
best_e = np.full(w.shape[1], np.inf)
best_s = np.ones(w.shape[1], dtype=np.int32)
for s in range(1, smax + 1):
    e = ((w - np.clip(np.rint(w / s), -127, 127) * s) ** 2).mean(axis=0)
    m = e < best_e
    best_e[m] = e[m]
    best_s[m] = s
s_mse = best_s[None, :]
nc = np.maximum(1, np.ceil(colmax / 127)).astype(np.int32)
print(f"  no-clip integer     rel {rel(deq(nc[None,:]))*100:6.3f}%")
print(f"  MSE-optimal integer rel {rel(deq(s_mse))*100:6.3f}%   "
      f"scales {np.bincount(best_s, minlength=9)[1:9].tolist()} (s=1..8)")
print(f"  columns where MSE-optimal < no-clip: {(best_s < nc).sum()}")
clipped = int((np.abs(np.rint(w / s_mse)) > 127).sum())
print(f"  clipped weights {clipped:,} ({clipped/w.size*100:.4f}%)")

np.save("/home/nick/projects/chessengine/data/nnue/nets/perf32/s_mse.npy", best_s)
np.save("/home/nick/projects/chessengine/data/nnue/nets/perf32/s_nc.npy", nc)
print("  (scales saved)")
