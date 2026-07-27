#!/usr/bin/env python3
"""Strip the FullThreats channel out of a d40 quantised.bin.

Purpose is measurement, not deployment. The threat features cost ~30% of all
engine cycles (perf study, 2026-07-27) and the obvious question — do they earn
that at equal time? — needs a HalfKA-only net to answer. Training one is a
GPU-days experiment; this produces one in seconds by deleting the channel from
an existing checkpoint, purely so the SPEED half of the question can be
measured now. The resulting net EVALUATES BADLY (its value head was trained
expecting threat contributions that are now identically zero) — do not read
anything into its play, only into its nodes/second.

Layout being edited (netpack.pack_d40 is the normative reader):
    l0_pstw   i16  HALFKA_DIMS x ACC
    l0_pstb   i16  ACC
    l0_threatw i8  THREAT_DIMS x ACC      <- deleted
    [policy]  i8   POLICY_VOCAB x L2      (when the net carries one)
    l1w       i8   BUCKETS*L2 x ACC
    ... f32 tails ...
    PSQ1      b"PSQ1" u32 buckets u32 rows i32[rows*buckets]
              rows = HALFKA_DIMS + THREAT_DIMS, HalfKA rows FIRST
              -> truncated to HALFKA_DIMS and the header rewritten

    python3 tools/nnue/strip_threats.py in-quant.bin out-quant.bin \
        --acc 1280 --l2 16 --policy

Pack the result with D40_HALFKA_ONLY=1 (plus the same D40_ACC/L2/L3): netpack
re-validates both the stream length and the PSQ1 row count, so a mistake here
fails loudly there rather than producing a plausible-but-wrong net.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

HALFKA_DIMS = 22528
THREAT_DIMS = 60720
POLICY_VOCAB = 1858


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--acc", type=int, required=True)
    ap.add_argument("--l2", type=int, required=True)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--policy", action="store_true")
    args = ap.parse_args()

    raw = Path(args.src).read_bytes()
    acc = args.acc

    head = (HALFKA_DIMS * acc + acc) * 2          # l0_pstw + l0_pstb, i16
    threat = THREAT_DIMS * acc                     # l0_threatw, i8
    if len(raw) < head + threat:
        raise SystemExit(f"{args.src}: too small for HALFKA+THREAT at acc={acc}")

    idx = raw.find(b"PSQ1", head + threat)
    if idx < 0:
        raise SystemExit(f"{args.src}: no PSQ1 section found")
    nb, rows = struct.unpack_from("<II", raw, idx + 4)
    if rows != HALFKA_DIMS + THREAT_DIMS:
        raise SystemExit(f"PSQ1 rows={rows}, expected {HALFKA_DIMS + THREAT_DIMS}")
    psq_body = idx + 12
    keep_psq = HALFKA_DIMS * nb * 4                # HalfKA rows come first
    if psq_body + rows * nb * 4 > len(raw):
        raise SystemExit("PSQ1 section runs past end of file")

    out = bytearray()
    out += raw[:head]                              # HalfKA FT + bias
    out += raw[head + threat:idx]                  # everything after threats,
                                                   # up to the PSQ1 magic
    out += b"PSQ1"
    out += struct.pack("<II", nb, HALFKA_DIMS)     # rewritten row count
    out += raw[psq_body:psq_body + keep_psq]       # HalfKA PSQT rows only
    # ...and EVERYTHING AFTER the PSQ1 body, verbatim. This used to stop here,
    # silently dropping the optional SKP1 output-skip section that follows —
    # and because SKP1 is optional in both netpack and the engine loader
    # (sota_loader.rs), nothing downstream complained. Any full-vs-stripped
    # speed comparison built with the old version therefore differed by the
    # skip lane as well as by the threat tensors, biasing the measured threat
    # cost upward. The docstring's "netpack re-validates, so a mistake here
    # fails loudly there" was never true for this particular mistake.
    tail = raw[psq_body + rows * nb * 4:]
    out += tail

    Path(args.dst).write_bytes(out)
    if tail:
        print(f"carried {len(tail)} trailing bytes through "
              f"(section {tail[:4].decode('ascii', 'replace')!r})")
    print(f"stripped {threat/1e6:.1f}MB of threat weights and "
          f"{(rows - HALFKA_DIMS) * nb * 4 / 1e6:.1f}MB of threat PSQT rows")
    print(f"{args.src}: {len(raw)} bytes -> {args.dst}: {len(out)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
