#!/usr/bin/env python3
"""Frequency-weighted k-means over the incumbent's threat rows (Recode R1).

The T0b/Tier-1 learned-compression candidate: cluster the trained
l0_threatw rows (60720, hl) into R' classes, weighted by census fire
frequency (+1 smoothed), init at the R' highest-frequency rows (the
learned-sketch heavy-hitter allocation). Outputs:

  <out>-remap.npy      int32[60720] class assignment (the code)
  <out>-centroids.npy  f32[R', hl]  weighted class means (the tied warm init)

Metric caveat (study §8.4, stated up front): weighted L2 in row space is the
obvious metric and may be the wrong one — rows interact through the shared
nonlinear head. If Tier-1 verdicts look noisy, this is the first suspect.

Usage: recode_kmeans.py <raw_ckpt> <census.npz> <rows> <out_prefix>
       [--iters 20] [--seed 1] [--tensor l0_threatw]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
from archon_jax import checkpoint as ck  # noqa: E402


def kmeans(x: np.ndarray, w: np.ndarray, k: int, iters: int, seed: int):
    n, d = x.shape
    rng = np.random.default_rng(seed)
    # Heavy-hitter init: the k highest-weight rows.
    centroids = x[np.argsort(w)[::-1][:k]].copy()
    assign = np.zeros(n, dtype=np.int64)
    chunk = 8192
    for it in range(iters):
        c2 = (centroids * centroids).sum(1)  # (k,)
        changed = 0
        sse = 0.0
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            # argmin ||x-c||^2 = argmin c2 - 2 x.c
            scores = c2[None, :] - 2.0 * (x[lo:hi] @ centroids.T)
            a = scores.argmin(1)
            changed += int((a != assign[lo:hi]).sum())
            assign[lo:hi] = a
            sse += float((scores[np.arange(hi - lo), a]
                          + (x[lo:hi] * x[lo:hi]).sum(1)).sum())
        # Weighted update.
        num = np.zeros((k, x.shape[1]), dtype=np.float64)
        den = np.zeros(k, dtype=np.float64)
        np.add.at(num, assign, x.astype(np.float64) * w[:, None])
        np.add.at(den, assign, w)
        empty = den == 0.0
        if empty.any():
            # Reseed empty clusters at random high-weight rows.
            seeds = rng.choice(n, size=int(empty.sum()),
                               p=w / w.sum(), replace=False)
            num[empty] = x[seeds].astype(np.float64) * w[seeds, None]
            den[empty] = w[seeds]
        centroids = (num / den[:, None]).astype(np.float32)
        print(f"iter {it:>2}: changed {changed:>6}  sse {sse:.4e}  "
              f"empty {int(empty.sum())}", flush=True)
        if changed == 0:
            break
    return assign, centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("census")
    ap.add_argument("rows", type=int)
    ap.add_argument("out_prefix")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tensor", default="l0_threatw")
    args = ap.parse_args()

    flat = ck.open_raw(args.ckpt, shapes=None, mmap=False)
    x = np.asarray(flat[args.tensor], dtype=np.float32)
    x = x.reshape(60720, -1)
    w = np.load(args.census)["th"].astype(np.float64) + 1.0
    print(f"rows {x.shape}, weights sum {w.sum():.3e}, k {args.rows}")

    assign, centroids = kmeans(x, w, args.rows, args.iters, args.seed)
    np.save(args.out_prefix + "-remap.npy", assign.astype(np.int32))
    np.save(args.out_prefix + "-centroids.npy", centroids)
    sizes = np.bincount(assign, minlength=args.rows)
    print(f"class sizes: min {sizes.min()} med {int(np.median(sizes))} "
          f"max {sizes.max()}; wrote {args.out_prefix}-remap/-centroids")


if __name__ == "__main__":
    main()
