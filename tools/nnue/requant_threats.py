#!/usr/bin/env python3
"""Re-quantise a raw checkpoint's THREAT channel at a different scale.

Why this exists (2026-07-29). The threat channel is 72.9% of our input space and
carries **86% of all accumulator quantisation noise** — 5.58% relative error at
25.1 dB SNR, against HalfKA's 0.51% / 45.8 dB. Two causes compound: threats are
quantised 4x coarser (scale 64 vs 255) AND their weights are 3x smaller in
magnitude, so the same absolute step is a proportionally larger error.

**Scale 64 is not a tuned choice — it is the largest value that does not
overflow.** `checkpoint.quantise` RAISES on overflow rather than clipping, and
our max threat weight is 1.9778, so 127/1.9778 = 64.2. One outlier out of
77.7 M weights is setting the precision for all of them.

Measured sweep on d48y2-sb170 (exact, using the real `_round_half_away`):

    scale    clipped     rel err   SNR      vs 64
       64          0       5.58%   25.1 dB  1.00x   <- today
      112        735       3.49%   29.1 dB  1.60x
      128      1,512       3.39%   29.4 dB  1.65x   <- optimum, 0.0019% clipped
      160      4,613       3.77%   28.5 dB  1.48x
      255     37,791       6.80%   23.3 dB  0.82x   <- Stockfish's, WORSE for us

**Stockfish's 255 is worse for OUR weights** because their trainer produces a
distribution that fits ±0.5 (that is what QAT buys them); ours spans ±1.98, so
255 just clips our tail. 128 is the optimum for the distribution we actually
have, and needs no retraining.

Engine side (NOT done by this script): the 255/64 frame ratio is carried as a
literal `<< 2` at `sota_head.rs:248` and `sota_simd.rs:423`. At scale 128 that
becomes `<< 1` (255/128 = 1.992, the same 0.39% gain approximation as today's
3.984 -> 4). The loader also validates quant constants, so both sides must move
together, and the NNUE gate pins move with them — intentionally.

Usage:
    python3 tools/nnue/requant_threats.py <raw.bin> <out-quant.bin> --scale 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jax"))
from archon_jax import checkpoint  # noqa: E402
from archon_jax import spec as spec_mod  # noqa: E402
from archon_jax.spec import ArchSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--scale", type=int, default=128,
                    help="threat quantisation scale (default 128 = measured optimum)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the error/clip statistics and write nothing")
    a = ap.parse_args()

    params = checkpoint.open_raw(a.raw, None, mmap=True)
    params = {k: np.asarray(v) for k, v in params.items()}
    w = params["l0_threatw"].astype(np.float32)

    lim = 127.0 / a.scale
    n_clip = int((np.abs(w) > lim).sum())
    wr = float(np.sqrt((w.astype(np.float64) ** 2).mean()))

    def err_at(scale):
        qf = checkpoint._round_half_away(np.float64(scale) * w.astype(np.float64))
        q = np.clip(qf, -127, 127) / scale
        return float(np.sqrt(((w - q).astype(np.float64) ** 2).mean()))

    e_old, e_new = err_at(64), err_at(a.scale)
    print(f"threat weights: n={w.size:,}  RMS {wr:.5f}  max|w| {np.abs(w).max():.4f}")
    print(f"  scale  64 : err {e_old:.6f}  ({e_old/wr*100:.2f}%, {20*np.log10(wr/e_old):.1f} dB)")
    print(f"  scale {a.scale:3d} : err {e_new:.6f}  ({e_new/wr*100:.2f}%, {20*np.log10(wr/e_new):.1f} dB)"
          f"   -> {e_old/e_new:.2f}x better")
    print(f"  clipped at scale {a.scale}: {n_clip:,} weights ({n_clip/w.size*100:.4f}%)")

    if a.dry_run:
        print("(dry run — nothing written)")
        return 0

    # CLIP before handing to the exporter: `checkpoint.quantise` RAISES on
    # overflow rather than clipping, which is exactly why scale 64 was forced.
    params["l0_threatw"] = np.clip(w, -lim, lim)

    old = spec_mod.QUANT_THREAT
    spec_mod.QUANT_THREAT = a.scale          # read at call time by param_tensors()
    try:
        n = checkpoint.write_quant(a.out, params, ArchSpec.full())
    finally:
        spec_mod.QUANT_THREAT = old
    print(f"wrote {a.out} ({n:,} bytes) with threat scale {a.scale}")
    print("REMEMBER: the engine's `<< 2` at sota_head.rs:248 and sota_simd.rs:423 "
          f"must become `<< {int(round(255/a.scale)).bit_length()-1}` "
          f"(255/{a.scale} = {255/a.scale:.3f}), and the CENN quant constants must match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
