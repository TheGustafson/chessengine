#!/usr/bin/env python3
"""In-place score rescale for 32-byte bulletformat records (D46 unit repair).

The sfbinpack loader applies SFBINPACK_SCORE_DIV at load time
(`score = (f32::from(score) / DIV).round() as i16` — round half AWAY from
zero). Dumps made without the env carry raw sfbinpack units. This applies
the identical mapping post-hoc, bit-exact to the Rust path: the quotient is
computed per possible i16 value as float32 (same IEEE op Rust performs),
then rounded half-away in float64 (every f32 is exact in f64, so this IS
roundf), via a 20001-entry LUT. Idempotence is NOT safe — run exactly once
per corpus; the ledger row must record it.

  python3 tools/nnue/rescale_records.py corpus.bin --div 1.9
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

RECORD = 32
SCORE_OFF = 24  # np.dtype offset of ("score", "<i2") after occ(8)+pcs(16)
CHUNK = 16 << 20  # records per pass (512 MiB)


def build_lut(div: float) -> np.ndarray:
    lut = np.empty(65536, dtype=np.int16)
    fdiv = np.float32(div)
    for s in range(-32768, 32768):
        q = float(np.float32(np.float32(s) / fdiv))  # exact f32 quotient
        r = math.floor(abs(q) + 0.5)  # exact in f64 => roundf semantics
        lut[s & 0xFFFF] = int(math.copysign(r, q)) if q != 0.0 else 0
    return lut


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--div", type=float, required=True)
    args = ap.parse_args()

    size = args.corpus.stat().st_size
    if size % RECORD:
        raise SystemExit(f"size {size} not a multiple of {RECORD}")
    total = size // RECORD
    lut = build_lut(args.div)

    mm = np.memmap(args.corpus, dtype=np.uint8, mode="r+")
    recs = mm.reshape(total, RECORD)
    done = 0
    while done < total:
        n = min(CHUNK, total - done)
        block = recs[done:done + n, SCORE_OFF:SCORE_OFF + 2]
        scores = block.view("<i2").ravel()
        block.view("<i2")[:, 0] = lut[scores.astype(np.int64) & 0xFFFF]
        done += n
        print(f"rescaled {done:,}/{total:,}", flush=True)
    mm.flush()
    print(f"OK: {total:,} records rescaled by /{args.div} (half-away, f32-exact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
