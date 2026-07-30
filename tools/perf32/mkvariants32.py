#!/usr/bin/env python3
"""perf study 32 phase 1b -- write i8-simulating CENN nets.

Quantising a HalfKA weight to i8 with scale `s` and multiplying by `s` on the
way into the accumulator is numerically IDENTICAL to storing `s*round(w/s)` in
today's i16 table.  So each scheme below is written out as an ordinary CENN-v2
net whose HalfKA weights lie on the i8 grid, and the shipped engine scores it.
No engine change is needed to get the fidelity number, and the number is exact
rather than a model.

Schemes (all symmetric; the FT weight mean is -1.5/255, so a zero point buys
almost nothing and costs a per-row correction):

  g<s>        global scale s, exact bias  (bias added at merge, accumulator
              seeded 0)          -> merge cost: unchanged when s is 4
  g<s>b       global scale s, bias rounded onto the grid and left in the
              accumulator        -> merge cost: unchanged, one fewer add
  col         per-column integer scale s_j = max(1, ceil(colmax_j/127)),
              zero clipping, exact bias  -> merge cost: one vpmullw
  colc<p>     per-column integer scale from the per-column p'th percentile
              (clips the tail), exact bias

    python3 tools/perf32/mkvariants32.py OUTDIR [scheme...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cennq  # noqa: E402

NET = "engines/canonical/net.cenn"


def quantise(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """w: (rows, cols) int32, 255-frame.  s: broadcastable integer scale.
    Returns the DEQUANTISED i16 weights (s * clip(round(w/s), -127, 127))."""
    q = np.clip(np.rint(w / s), -127, 127)
    return (q * s).astype(np.int16)


def col_scales(w: np.ndarray, pct: float | None) -> np.ndarray:
    aw = np.abs(w)
    ref = aw.max(axis=0) if pct is None else np.percentile(aw, pct, axis=0)
    return np.maximum(1, np.ceil(ref / 127)).astype(np.int32)[None, :]


def main() -> int:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    schemes = sys.argv[2:] or ["g4", "g4b", "g8", "col", "colc99.99"]

    c = cennq.load(NET)
    w = c.halfka().astype(np.int32)
    bias = c.ft_bias().copy()

    for name in schemes:
        if name.startswith("g"):
            body = name[1:]
            round_bias = body.endswith("b")
            s = int(body[:-1] if round_bias else body)
            sarr = np.int32(s)
            wq = quantise(w, sarr)
            bq = (np.rint(bias / s) * s).astype(np.int16) if round_bias else None
        elif name == "col":
            sarr = col_scales(w, None)
            wq = quantise(w, sarr)
            bq = None
        elif name.startswith("colc"):
            sarr = col_scales(w, float(name[4:]))
            wq = quantise(w, sarr)
            bq = None
        else:
            raise SystemExit(f"unknown scheme {name}")

        out = outdir / f"d48x-i8-{name}.cenn"
        h = cennq.write_variant(c, out, wq, bq)
        err = (w - wq.astype(np.int32)).astype(np.float64)
        rel = np.sqrt((err**2).mean()) / np.sqrt((w.astype(np.float64) ** 2).mean())
        smin, smax = int(np.min(sarr)), int(np.max(sarr))
        print(f"{name:<12s} s={smin}..{smax}  rel_err {rel*100:6.3f}%  "
              f"max|err| {int(np.abs(err).max()):3d}  bias={'grid' if bq is not None else 'exact'}  "
              f"sha {h}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
