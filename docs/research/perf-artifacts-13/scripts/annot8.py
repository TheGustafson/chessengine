#!/usr/bin/env python3
"""perf study 8 — split a `perf annotate --stdio` dump into contiguous
address blocks so a symbol's internal phases can be costed separately.

    python3 annot8.py <annot.txt> [split-addr ...]

With no split addresses it prints the running cumulative percentage every 5%
plus the 25 hottest instructions, which is how you FIND the phase boundaries.
With split addresses it prints the percentage sum of each [addr_i, addr_i+1)
block — the phase costs.
"""

import re
import sys

path = sys.argv[1]
splits = [int(a, 16) for a in sys.argv[2:]]

ROW = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-f]+):\s+(\S+)(.*)$")
rows = []
for line in open(path):
    m = ROW.match(line)
    if m:
        rows.append((int(m.group(2), 16), float(m.group(1)), m.group(3), m.group(4).strip()))

total = sum(r[1] for r in rows)
print(f"{len(rows)} instructions, total local percent {total:.2f}")

if not splits:
    print("\n--- cumulative every 5% ---")
    run = 0.0
    mark = 5.0
    for addr, pct, op, rest in rows:
        run += pct
        if run >= mark:
            print(f"  {run:6.2f}%  at {addr:x}  {op} {rest[:60]}")
            mark += 5.0
    print("\n--- 25 hottest instructions ---")
    for addr, pct, op, rest in sorted(rows, key=lambda r: -r[1])[:25]:
        print(f"  {pct:6.2f}%  {addr:x}  {op:16s} {rest[:70]}")
    print("\n--- by opcode ---")
    agg = {}
    for _, pct, op, _ in rows:
        agg[op] = agg.get(op, 0) + pct
    for op, pct in sorted(agg.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {pct:6.2f}%  {op}")
else:
    bounds = [rows[0][0]] + splits + [rows[-1][0] + 1]
    for lo, hi in zip(bounds, bounds[1:]):
        s = sum(p for a, p, _, _ in rows if lo <= a < hi)
        n = sum(1 for a, _, _, _ in rows if lo <= a < hi)
        print(f"  [{lo:x},{hi:x})  {s:7.2f}%  ({s / total * 100:5.1f}% of symbol)  {n} static instrs")
