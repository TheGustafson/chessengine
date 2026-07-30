#!/usr/bin/env python3
"""perf study 32 phase 1c -- eval fidelity of each i8 scheme, in SF-frame cp.

Scores every FEN of a position set with the shipped engine under the reference
net and under each variant, and reports the delta distribution in BOTH frames.
Our printed scores are 2.07x Stockfish's PRINTED cp; the conversion goes through
`searchopt/tools/uciprobe.to_sf_frame` so it cannot be hand-fumbled (CLAUDE.md
rule R2).

    python3 tools/perf32/evaldelta32.py <fens> <ref.cenn> <var.cenn>...
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "searchopt" / "tools"))
from uciprobe import to_sf_frame  # noqa: E402

BIN = str(ROOT / "engines" / "bin" / "archon-s25")


def scores(net: str, fens_path: str) -> np.ndarray:
    fens = Path(fens_path).read_bytes()
    out = subprocess.run(
        ["nice", "-n", "19", BIN, "nnue-eval", net],
        input=fens, capture_output=True, check=True,
    ).stdout.decode()
    return np.array([int(x) for x in out.split()], dtype=np.int64)


def main() -> int:
    fens_path, ref_net = sys.argv[1], sys.argv[2]
    variants = sys.argv[3:]
    ref = scores(ref_net, fens_path)
    sf_ref = np.array([to_sf_frame(v) for v in ref])
    print(f"positions {len(ref)}   set {fens_path}")
    print(f"reference {ref_net}")
    print(f"  ours: mean {ref.mean():+8.1f}  sd {ref.std():7.1f}  "
          f"|.| median {np.median(np.abs(ref)):7.1f}")
    print(f"  SF cp: mean {sf_ref.mean():+8.1f}  sd {sf_ref.std():7.1f}")
    print()
    hdr = (f"{'scheme':<26s} {'mean':>7s} {'median|d|':>9s} {'p95':>7s} {'p99':>7s} "
           f"{'max':>7s} {'sign+':>6s} {'spear':>7s}")
    print(hdr + "        (SF-frame cp)")
    print("-" * len(hdr))
    for v in variants:
        s = scores(v, fens_path)
        d = np.array([to_sf_frame(int(x)) for x in (s - ref)])
        ad = np.abs(d)
        # Spearman without scipy: Pearson on ranks.
        r1 = np.argsort(np.argsort(ref)).astype(float)
        r2 = np.argsort(np.argsort(s)).astype(float)
        spear = float(np.corrcoef(r1, r2)[0, 1])
        print(f"{Path(v).stem:<26s} {d.mean():+7.3f} {np.median(ad):9.3f} "
              f"{np.percentile(ad,95):7.3f} {np.percentile(ad,99):7.3f} "
              f"{ad.max():7.3f} {(d>0).mean()*100:5.1f}% {spear:7.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
