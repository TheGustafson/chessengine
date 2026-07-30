#!/usr/bin/env python3
"""perf study 32 phase 1a -- the HalfKA FT weight distribution.

Answers, for the canonical net: what is the dynamic range of the HalfKA feature
transformer, globally and per accumulator column, and what scale would an i8
representation need?  No engine, no build.

    python3 tools/perf32/census32.py [net.cenn]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cennq  # noqa: E402


def main() -> int:
    net_path = sys.argv[1] if len(sys.argv) > 1 else "engines/canonical/net.cenn"
    c = cennq.load(net_path)
    print(f"net            {net_path}")
    print(f"arch           halfka_dims={c.halfka_dims} threat_dims={c.threat_dims} "
          f"acc={c.acc} buckets={c.buckets}")
    print(f"quant          ft_quant={c.ft_quant} l1_quant={c.l1_quant} "
          f"ft_shift={c.ft_shift} scale={c.network_scale}")

    w = c.halfka().astype(np.int32)      # (22528, 1280), 255-frame integers
    n = w.size
    aw = np.abs(w)
    print()
    print("=== HalfKA FT weights, 255-frame integer units "
          f"({c.halfka_dims} rows x {c.acc} cols = {n:,}) ===")
    print(f"  mean {w.mean():+.4f}  std {w.std():.4f}  "
          f"mean|w| {aw.mean():.4f}  rms {np.sqrt((w.astype(np.float64)**2).mean()):.4f}")
    qs = [50, 90, 99, 99.9, 99.99, 99.999, 100]
    pv = np.percentile(aw, qs)
    for q, v in zip(qs, pv):
        print(f"  p{q:<8} |w| = {v:9.2f}   (float {v/c.ft_quant:+.5f})")
    mx = int(aw.max())
    print(f"  max|w| = {mx} = {mx / c.ft_quant:.5f} in float units")
    print(f"  zero weights: {(w == 0).sum():,} ({(w == 0).mean()*100:.2f}%)")

    # --- global symmetric scale --------------------------------------------
    print()
    print("=== global symmetric i8 scale (w_i8 = round(w/s), clipped to +-127) ===")
    print(f"  no-clip scale needs s >= ceil({mx}/127) = {-(-mx // 127)}")
    for s in (2, 3, 4, 5, 6, 8, 12, 16):
        q = np.round(w / s)
        clipped = int((np.abs(q) > 127).sum())
        q = np.clip(q, -127, 127)
        err = w - q * s
        rel = np.sqrt((err.astype(np.float64) ** 2).mean()) / np.sqrt(
            (w.astype(np.float64) ** 2).mean()
        )
        snr = -20 * np.log10(rel) if rel > 0 else float("inf")
        print(f"  s={s:<3d} clipped {clipped:>9,} ({clipped/n*100:7.4f}%)  "
              f"rms_err {np.sqrt((err.astype(np.float64)**2).mean()):7.3f}  "
              f"rel {rel*100:6.3f}%  SNR {snr:5.1f} dB")

    # --- per-column (per accumulator lane) ---------------------------------
    colmax = aw.max(axis=0)
    print()
    print("=== per-column max|w| over the 1280 accumulator lanes ===")
    for q, v in zip([0, 10, 50, 90, 100], np.percentile(colmax, [0, 10, 50, 90, 100])):
        print(f"  p{q:<4} colmax = {v:8.1f}")
    need = np.maximum(1, np.ceil(colmax / 127)).astype(np.int32)
    vals, cnts = np.unique(need, return_counts=True)
    print("  per-column no-clip integer scale histogram: "
          + "  ".join(f"s={v}:{cn}" for v, cn in zip(vals, cnts)))

    def report(tag, q, s_row):
        """s_row broadcasts over columns."""
        deq = q * s_row
        err = w - deq
        rel = np.sqrt((err.astype(np.float64) ** 2).mean()) / np.sqrt(
            (w.astype(np.float64) ** 2).mean()
        )
        snr = -20 * np.log10(rel)
        clipped = int((np.abs(np.round(w / s_row)) > 127).sum())
        print(f"  {tag:<34s} clipped {clipped:>9,} ({clipped/n*100:7.4f}%)  "
              f"rel {rel*100:6.3f}%  SNR {snr:5.1f} dB  "
              f"max|err| {int(np.abs(err).max())}")

    print()
    print("=== per-column integer scales ===")
    s_nc = need.astype(np.int32)[None, :]
    report("no-clip ceil(colmax/127)", np.clip(np.round(w / s_nc), -127, 127), s_nc)
    for pct in (99.9, 99.99, 99.999):
        thr = np.percentile(aw, pct, axis=0)
        s = np.maximum(1, np.ceil(thr / 127)).astype(np.int32)[None, :]
        report(f"clip at per-col p{pct}", np.clip(np.round(w / s), -127, 127), s)

    # threat channel for reference (already i8 at scale 64)
    t = c.threats().astype(np.int32)
    print()
    print("=== threat FT weights (already i8, 64-frame) for reference ===")
    print(f"  max|t| {np.abs(t).max()}  rms {np.sqrt((t.astype(np.float64)**2).mean()):.3f}"
          f"  zeros {(t == 0).mean()*100:.2f}%")
    print(f"  in FLOAT units: max {np.abs(t).max()/64:.4f}  "
          f"rms {np.sqrt((t.astype(np.float64)**2).mean())/64:.5f}")
    print(f"  HalfKA in FLOAT units: max {mx/255:.4f}  "
          f"rms {np.sqrt((w.astype(np.float64)**2).mean())/255:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
