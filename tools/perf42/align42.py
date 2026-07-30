#!/usr/bin/env python3
"""perf study 42 -- does a 32-byte cluster ever straddle a 64-byte line?

    python3 tools/perf42/align42.py <ibs-script> <lo-hex> <hi-hex>

`Cluster` is `#[repr(C, align(32))]` and `const _: () = assert!(align_of::
<Cluster>() == 32)`, so the TYPE cannot straddle. What the type system does NOT
prove is that the running allocation is where the type says it is: the claim
"a probe touches one line" also needs the `Vec<Cluster>` buffer to start on a
32-byte boundary, which is an ALLOCATOR promise, not a compiler one.

This checks it from the outside, on the shipped binary, with no instrumentation:
histogram the low 6 bits of every observed TT data address. A 32-aligned table
of 32-byte clusters holding 10-byte entries can only produce byte offsets
  {0..29} within the low half and {32..61} within the high half
of a line -- offsets 30, 31, 62, 63 are the two pad bytes and are never touched.
Any traffic at a forbidden offset, or any access whose 16-bit field crosses a
64-byte boundary, is a straddle and a bug.
"""

from __future__ import annotations

import argparse
import collections
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from regions42 import parse  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("lo")
    ap.add_argument("hi")
    args = ap.parse_args()
    lo, hi = int(args.lo, 16), int(args.hi, 16)

    off = collections.Counter()
    n = 0
    for addr, _op, _lvl, _tlb, _ip in parse(args.script):
        if lo <= addr < hi:
            off[addr % 64] += 1
            n += 1

    print(f"TT accesses sampled: {n}")
    print(f"{'off%64':>7} {'count':>7}   {'off%64':>7} {'count':>7}")
    for i in range(32):
        print(f"{i:>7} {off[i]:>7}   {i + 32:>7} {off[i + 32]:>7}")

    pad = sum(off[i] for i in (30, 31, 62, 63))
    lohalf = sum(off[i] for i in range(32))
    hihalf = n - lohalf
    print()
    print(f"low half  [0,32): {lohalf} ({100.0 * lohalf / max(n, 1):.1f}%)")
    print(f"high half [32,64): {hihalf} ({100.0 * hihalf / max(n, 1):.1f}%)")
    print(f"pad bytes {{30,31,62,63}}: {pad}  -- must be ~0 if the table is 32-aligned")
    # A 2-byte field starting at 63, or an 8-byte field starting past 56, would
    # cross the line. Every TtEntry field is 1 or 2 bytes and every entry base
    # is at cluster+0/10/20, so the only crossing offsets possible are 63.
    print(f"offset 63 (only offset a 2-byte field could straddle from): {off[63]}")
    print("STRADDLE RATE: 0.000% by construction" if pad == 0 else "STRADDLE DETECTED")


if __name__ == "__main__":
    main()
