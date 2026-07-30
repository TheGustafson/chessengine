#!/usr/bin/env python3
"""perf study 42 -- the TT's TLB behaviour against the engine's.

    python3 tools/perf42/tlb42.py <ibs-script> <lo-hex> <hi-hex>

THP is `always` on this box and `defrag` is `madvise`, and the engine gets
**zero** transparent huge pages (`tools/perf42/smaps42.sh`): with the machine's
free memory fragmented, a fault that cannot get a 2 MB page instantly falls
back to 4 KB and khugepaged never catches up inside a 4-second bench. So every
TT probe is also a 4 KB page walk candidate -- a 16 MB table is 4096 pages and
a 256 MB one is 65536, against a 3072-entry L2 dTLB.

This cross-tabulates the IBS TLB field inside the TT's address range against
the whole process, so the table's share of page walks can be read off rather
than argued from capacity.
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

    inside: collections.Counter = collections.Counter()
    outside: collections.Counter = collections.Counter()
    for addr, _op, _lvl, tlb, _ip in parse(args.script):
        (inside if lo <= addr < hi else outside)[tlb or "(blank)"] += 1

    ni, no = sum(inside.values()), sum(outside.values())
    print(f"{'TLB outcome':<24} {'TT':>9} {'TT%':>7} {'rest':>9} {'rest%':>7}")
    for k in sorted(set(inside) | set(outside), key=lambda x: -(inside[x] + outside[x])):
        print(
            f"{k:<24} {inside[k]:>9} {100.0 * inside[k] / max(ni, 1):>6.2f}% "
            f"{outside[k]:>9} {100.0 * outside[k] / max(no, 1):>6.2f}%"
        )
    miss_i = sum(v for k, v in inside.items() if "hit" not in k.lower() and k != "(blank)")
    miss_o = sum(v for k, v in outside.items() if "hit" not in k.lower() and k != "(blank)")
    print()
    print(f"TT accesses {ni}, of which non-L1-TLB-hit {miss_i} ({100.0 * miss_i / max(ni, 1):.2f}%)")
    print(f"rest        {no}, of which non-L1-TLB-hit {miss_o} ({100.0 * miss_o / max(no, 1):.2f}%)")
    print(f"TT share of all non-L1-TLB-hits: {100.0 * miss_i / max(miss_i + miss_o, 1):.2f}%")


if __name__ == "__main__":
    main()
