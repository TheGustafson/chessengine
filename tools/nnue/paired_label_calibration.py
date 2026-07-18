#!/usr/bin/env python3
"""Calibration of cheap vs deep cp labels on the SAME line-aligned FEN set.

Label-value pilot corroborator + pre-gate. Reads two `FEN | cp | result` files
that MUST be line-aligned (same positions, same order). Reports Pearson,
Spearman, RMS, MAE overall and stratified by |deep cp| decision band. A tiny
gap (Pearson>0.98, RMS<25cp) => cheap ~ deep => deeper labels can't help.
"""
import math
import sys

import numpy as np


def col(path):
    fens, cps = [], []
    for l in open(path):
        if not l.strip():
            continue
        parts = l.split(" | ")
        fens.append(parts[0])
        cps.append(int(parts[1]))
    return fens, np.array(cps)


def main():
    fa, cheap = col(sys.argv[1])
    fb, deep = col(sys.argv[2])
    n = min(len(cheap), len(deep))
    cheap, deep = cheap[:n], deep[:n]
    if fa[:n] != fb[:n]:
        aligned = sum(1 for x, y in zip(fa[:n], fb[:n]) if x == y)
        print(f"WARNING: files not fully line-aligned ({aligned}/{n} FENs match)", file=sys.stderr)
    r = float(np.corrcoef(cheap, deep)[0, 1])
    rho = float(np.corrcoef(np.argsort(np.argsort(cheap)), np.argsort(np.argsort(deep)))[0, 1])
    rms = math.sqrt(float(np.mean((cheap - deep) ** 2)))
    mae = float(np.mean(np.abs(cheap - deep)))
    flip = float(np.mean(np.sign(cheap) != np.sign(deep)))
    print(f"n={n} Pearson={r:.4f} Spearman={rho:.4f} RMS={rms:.1f}cp MAE={mae:.1f}cp sign_flip={flip:.3f}")
    for lo, hi in [(0, 50), (50, 150), (150, 400), (400, 2500)]:
        m = (np.abs(deep) >= lo) & (np.abs(deep) < hi)
        if m.sum():
            band_rms = math.sqrt(float(np.mean((cheap[m] - deep[m]) ** 2)))
            print(f"  |deep|in[{lo},{hi}): n={int(m.sum())} RMS={band_rms:.1f}cp")


if __name__ == "__main__":
    main()
