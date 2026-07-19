#!/usr/bin/env python3
"""Calibrate the label-WRONGNESS selector on a 5k/50k answer-key.

The targeted-relabel flywheel (docs/research/data-quality-curriculum.md §0) hinges
on cheaply predicting which shallow (5k-node) labels are IMPACTFULLY wrong WITHOUT
deep-labeling first. c1 gives us the answer key: positions with BOTH 5k and 50k
labels. This tool measures how well cheap, net-FREE features predict impactful
disagreement, so we know whether targeting is viable (good recall in a small
top-k%) or needs a medium-depth screen.

Inputs:
  --deep    corpus.txt lines "FEN | cp50k | wdl"   (the SF18@50k answer)
  --shallow TSV "FEN\\tresult\\tply\\tcp5k"           (fast_fens --score output)
Join by FEN, compute net-free features + impactful-wrongness labels, report each
feature's Pearson r with |Δcp| and its recall@k for the wrong set vs the random
k% baseline (lift = how much better than blind sampling).
"""
from __future__ import annotations
import argparse, sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", required=True, help="corpus.txt (FEN | cp50k | wdl)")
    ap.add_argument("--shallow", required=True, help="TSV FEN\\tresult\\tply\\tcp5k")
    ap.add_argument("--flip-thresh", type=int, default=50, help="|cp| to count as 'decisive' for a winner-flip")
    args = ap.parse_args()

    deep = {}
    for line in open(args.deep):
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 3:
            try:
                deep[p[0]] = (float(p[1]), float(p[2]))
            except ValueError:
                pass
    print(f"deep(50k) labels: {len(deep)}", file=sys.stderr)

    rows = []  # (cp5, cp50, wdl, ply)
    for line in open(args.shallow):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        d = deep.get(f[0])
        if d is None:
            continue
        try:
            rows.append((float(f[3]), d[0], d[1], int(f[2])))
        except ValueError:
            continue
    n = len(rows)
    if n == 0:
        sys.exit("no matched positions (shallow extract and deep corpus don't overlap)")

    # --- impactful-wrongness targets ---
    T = args.flip_thresh
    def is_flip(cp5, cp50):
        return (cp5 > T and cp50 < -T) or (cp5 < -T and cp50 > T)
    def is_band_err(cp5, cp50):   # big correction in the near-balanced band (training-sensitive)
        return abs(cp5 - cp50) > 100 and abs(cp50) < 250
    wrong = [is_flip(c5, c50) or is_band_err(c5, c50) for c5, c50, _, _ in rows]
    nflip = sum(is_flip(c5, c50) for c5, c50, _, _ in rows)
    nwrong = sum(wrong)
    print(f"matched: {n}   winner-flips: {nflip} ({100*nflip/n:.2f}%)   impactful-wrong: {nwrong} ({100*nwrong/n:.2f}%)")

    # --- cheap net-free features (higher = more suspect, by hypothesis) ---
    def feat_outcome_mismatch(cp5, cp50, wdl, ply):
        # eval predicts a winner opposite to the game outcome
        s = 1 if cp5 > T else (-1 if cp5 < -T else 0)
        o = 1 if wdl > 0.6 else (-1 if wdl < 0.4 else 0)
        return 1.0 if (s and o and s != o) else 0.0
    def feat_near_zero(cp5, cp50, wdl, ply):
        return 1.0 / (1.0 + abs(cp5) / 100.0)      # peaks near balanced positions
    def feat_abs_cp5(cp5, cp50, wdl, ply):
        return abs(cp5)                             # extreme shallow evals (over-confident?)
    def feat_ply(cp5, cp50, wdl, ply):
        return float(ply)
    feats = {
        "outcome_mismatch": feat_outcome_mismatch,
        "near_zero_eval":   feat_near_zero,
        "abs_cp5k":         feat_abs_cp5,
        "ply":              feat_ply,
    }

    absd = [abs(c5 - c50) for c5, c50, _, _ in rows]
    def pearson(xs, ys):
        m = len(xs); mx = sum(xs)/m; my = sum(ys)/m
        cov = sum((a-mx)*(b-my) for a, b in zip(xs, ys))/m
        sx = (sum((a-mx)**2 for a in xs)/m)**.5; sy = (sum((b-my)**2 for b in ys)/m)**.5
        return cov/(sx*sy) if sx*sy else 0.0

    print(f"\n{'feature':<18} {'r(|Δcp|)':>9}   recall@k of impactful-wrong (lift vs random)")
    for name, fn in feats.items():
        vals = [fn(*r) for r in rows]
        r = pearson(vals, absd)
        order = sorted(range(n), key=lambda i: vals[i], reverse=True)
        line = f"{name:<18} {r:>9.3f}   "
        for k in (5, 10, 20):
            topk = order[: max(1, n*k//100)]
            hit = sum(wrong[i] for i in topk)
            recall = hit / max(nwrong, 1)
            lift = recall / (k/100)
            line += f"k={k}%:{100*recall:4.0f}%({lift:.1f}x)  "
        print(line)
    print("\nread: recall@k = fraction of impactful-wrong positions captured in the top-k% by that feature;")
    print("lift = recall / (k/100). lift>>1 in a small k => cheap targeting is viable. lift~1 => feature is blind.")


if __name__ == "__main__":
    main()
