#!/usr/bin/env python3
"""Neck-capacity probe with its NULL.

Question: is ARCHON's L2 ("neck") width a binding constraint?

Statistic: the participation ratio (exp of the entropy of the squared singular
spectrum) of (a) the per-bucket L1 weight matrix [L2 x ACC] and (b) the per-bucket
L2 weight matrix [L2 x L3].

NULL PANEL (run FIRST, per CLAUDE.md rule 1): the same statistic on i.i.d. random
matrices of identical shape, with entries drawn to match the empirical marginal.
A statistic that cannot distinguish a trained net from noise cannot answer the
question. Reported side by side; only the RATIO observed/null is interpreted.

Pure numpy over the .cenn files. No engine, no search.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from net_anatomy import eff_rank, parse  # noqa: E402

RNG = np.random.default_rng(20260728)


def null_eff_rank(shape, sample, reps=32):
    """eff_rank of random matrices of `shape` whose entries are RESAMPLED (with
    replacement) from the observed entries — same marginal, no structure."""
    flat = sample.ravel()
    out = []
    for _ in range(reps):
        m = RNG.choice(flat, size=shape, replace=True).astype(np.float64)
        out.append(eff_rank(np.linalg.svd(m, compute_uv=False)))
    return float(np.mean(out)), float(np.std(out))


def probe(path):
    n = parse(path)
    l2, l3, ob, acc = n["l2"], n["l3"], n["buckets"], n["acc"]
    r1 = [eff_rank(np.linalg.svd(n["l1_w"][b].astype(np.float64), compute_uv=False))
          for b in range(ob)]
    r2 = [eff_rank(np.linalg.svd(n["l2_w"][b].astype(np.float64), compute_uv=False))
          for b in range(ob)]
    n1m, n1s = null_eff_rank((l2, acc), n["l1_w"][0], reps=8)
    n2m, n2s = null_eff_rank((l2, l3), n["l2_w"][0], reps=32)
    name = path.rsplit("/", 1)[-1]
    print(f"{name:42s} acc={acc:5d} L2={l2:4d} L3={l3:3d}")
    print(f"   L1 [L2 x ACC] eff_rank  obs {np.mean(r1):6.2f} / {l2:3d}   "
          f"NULL {n1m:6.2f}±{n1s:.2f}   obs/null {np.mean(r1)/n1m:.3f}")
    print(f"   L2 [L2 x L3]  eff_rank  obs {np.mean(r2):6.2f} / {min(l2,l3):3d}   "
          f"NULL {n2m:6.2f}±{n2s:.2f}   obs/null {np.mean(r2)/n2m:.3f}")
    return dict(name=name, acc=acc, l2=l2, l3=l3,
                r1=float(np.mean(r1)), r1null=n1m,
                r2=float(np.mean(r2)), r2null=n2m)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        probe(p)
