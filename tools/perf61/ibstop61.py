#!/usr/bin/env python3
"""perf study 61 — IBS census by SAMPLED-OP COUNT and by TAG-TO-RETIRE LATENCY.

    python3 tools/perf61/ibstop61.py <ibs-script.txt> [top] [--offsets SYMPAT]

Study 29 §8's rule is that memory work must be ranked by miss LATENCY, not miss
count, and study 60 confirmed it the hard way (a function that is #1 on the
count ranking is 11.8% of the cost ranking).  The same rule applies to CYCLES:
`perf record -e cycles` cannot be trusted per-line on this box, but IBS tags an
op at ISSUE and records its own tag-to-retire latency, so summing that weight
per symbol is a latency-weighted profile with no skid.

Prints both columns side by side, because the DIFFERENCE between them is the
signal: a symbol whose latency share far exceeds its op share is stalling.
"""

import collections
import re
import sys

TAIL = re.compile(r"BLK\s+\S+\s+(\d+)\s+([0-9a-f]+)\s+(\S+)(?:\+(0x[0-9a-f]+))?")


def main() -> None:
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 25
    offpat = None
    if "--offsets" in sys.argv:
        offpat = re.compile(sys.argv[sys.argv.index("--offsets") + 1])

    ops: collections.Counter[str] = collections.Counter()
    lat: collections.Counter[str] = collections.Counter()
    off_ops: collections.Counter[tuple[str, str]] = collections.Counter()
    off_lat: collections.Counter[tuple[str, str]] = collections.Counter()

    with open(path) as fh:
        for line in fh:
            m = TAIL.search(line)
            if not m:
                continue
            w, _ip, sym, off = m.group(1), m.group(2), m.group(3), m.group(4)
            w = int(w)
            ops[sym] += 1
            lat[sym] += w
            if offpat and offpat.search(sym):
                off_ops[(sym, off or "0x0")] += 1
                off_lat[(sym, off or "0x0")] += w

    tot_ops = sum(ops.values())
    tot_lat = sum(lat.values())
    print(f"# sampled ops {tot_ops}  total tag-to-retire {tot_lat}  "
          f"mean {tot_lat / max(tot_ops, 1):.2f} cy/op")
    print(f"{'ops%':>7} {'lat%':>7} {'cy/op':>7}  symbol")
    for sym, n in lat.most_common(top):
        print(f"{100 * ops[sym] / tot_ops:7.3f} {100 * n / tot_lat:7.3f} "
              f"{n / ops[sym]:7.2f}  {sym[:100]}")

    if offpat:
        print()
        print(f"{'ops%':>7} {'lat%':>7} {'cy/op':>7}  symbol+offset")
        for (sym, off), n in off_lat.most_common(top):
            print(f"{100 * off_ops[(sym, off)] / tot_ops:7.3f} {100 * n / tot_lat:7.3f} "
                  f"{n / off_ops[(sym, off)]:7.2f}  {sym[:70]}+{off}")


if __name__ == "__main__":
    main()
