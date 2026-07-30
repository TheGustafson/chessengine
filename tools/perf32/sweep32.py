#!/usr/bin/env python3
"""perf study 32 -- weight-domain sweep of the quantisation-scheme family.

A screen, not a verdict: the five schemes already measured end-to-end give
p99(SF cp) = 3.7 x rel_err(%) with a spread of 3.66-3.91, so rel_err is a good
cheap proxy for ranking schemes.  Only the winners get materialised as nets and
scored by the engine.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cennq  # noqa: E402

c = cennq.load(sys.argv[1] if len(sys.argv) > 1 else "engines/canonical/net.cenn")
w = c.halfka().astype(np.float64)  # (22528, 1280), 255-frame
den = np.sqrt((w**2).mean())
aw = np.abs(w)


def rel(wq):
    return np.sqrt(((w - wq) ** 2).mean()) / den


def q(w_, s):
    return np.clip(np.rint(w_ / s), -127, 127) * s


print(f"baseline rms {den:.3f}")
print()
print("=== row structure: max|w| per FEATURE row (22528 rows) ===")
rowmax = aw.max(axis=1)
for p in (0, 10, 50, 90, 99, 100):
    print(f"  p{p:<4} rowmax = {np.percentile(rowmax, p):8.1f}")
print(f"  rows all-zero: {(rowmax == 0).sum()}")
colmax = aw.max(axis=0)
print(f"  rowmax p90/p10 = {np.percentile(rowmax,90)/max(1,np.percentile(rowmax,10)):.2f}"
      f"   colmax p90/p10 = {np.percentile(colmax,90)/np.percentile(colmax,10):.2f}")

print()
print("=== scale families (rel err %) ===")
print(f"  global s=4                       {rel(q(w,4))*100:6.3f}")
s_int = np.maximum(1, np.ceil(colmax / 127))[None, :]
print(f"  per-col integer no-clip          {rel(q(w,s_int))*100:6.3f}")
s_frac = np.maximum(1e-9, colmax / 127)[None, :]
print(f"  per-col fractional no-clip       {rel(q(w,s_frac))*100:6.3f}")

grid = np.linspace(0.30, 1.0, 36)
best = np.full(w.shape[1], np.inf)
best_s = np.zeros(w.shape[1])
for g in grid:
    s = np.maximum(1e-9, colmax * g / 127)[None, :]
    e = ((w - q(w, s)) ** 2).mean(axis=0)
    m = e < best
    best[m] = e[m]
    best_s[m] = (colmax * g / 127)[m]
s_opt = np.maximum(1e-9, best_s)[None, :]
print(f"  per-col fractional MSE-optimal   {rel(q(w,s_opt))*100:6.3f}   "
      f"(median clip fraction {np.median(best_s*127/np.maximum(1,colmax)):.2f})")

print()
print("=== per-ROW power-of-two shift x per-column scale ===")
for name, sc in (("col-frac", s_frac), ("col-opt", s_opt)):
    r = np.maximum(1e-9, (aw / sc).max(axis=1))
    k = np.clip(np.floor(np.log2(127.0 / r)), 0, 6)
    kk = (2.0**k)[:, None]
    wq = np.clip(np.rint(w / sc * kk), -127, 127) * sc / kk
    print(f"  {name} + per-row shift          {rel(wq)*100:6.3f}   "
          f"k hist {[int((k==i).sum()) for i in range(7)]}")

# per-row shift chosen by MSE rather than by no-clip
r = np.maximum(1e-9, (aw / s_opt).max(axis=1))
kbase = np.clip(np.floor(np.log2(127.0 / r)), 0, 6)
bestrow = None
for extra in (0, 1, 2):
    k = np.clip(kbase + extra, 0, 6)
    kk = (2.0**k)[:, None]
    wq = np.clip(np.rint(w / s_opt * kk), -127, 127) * s_opt / kk
    e = rel(wq)
    print(f"  col-opt + per-row shift +{extra}      {e*100:6.3f}")
    if bestrow is None or e < bestrow[0]:
        bestrow = (e, wq)

print()
print(f"best in family {bestrow[0]*100:6.3f}%   "
      f"target for p99 <= 5 SF cp is <= 1.35%")
