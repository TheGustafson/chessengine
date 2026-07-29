#!/usr/bin/env python3
"""perf study 8 — find the COLD STATIC MASS inside a hot symbol.

    python3 cold8.py <annot.txt> [min-run]

A monolithic PGO-grown function pays register pressure for code it almost
never executes: every cold block that LLVM inlined still competes for
registers across the hot path, which is the mechanism impl-1 proved on
`next_move` (13.5% -> 4.9% stack traffic, -5.2% engine instructions, total
text UNCHANGED).  This lists the longest contiguous runs of near-zero-sample
instructions, i.e. the blocks worth outlining, and reports what share of the
symbol's static size is cold.
"""

import re
import sys

path = sys.argv[1]
MIN_RUN = int(sys.argv[2]) if len(sys.argv) > 2 else 25

ROW = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
rows = []
for line in open(path):
    m = ROW.match(line)
    if m:
        rows.append((int(m.group(2), 16), float(m.group(1)), m.group(3), m.group(4).strip()))

runs = []
start = None
acc = 0.0
for i, (addr, pct, op, rest) in enumerate(rows):
    if pct <= 0.005:
        if start is None:
            start, acc = i, 0.0
        acc += pct
    else:
        if start is not None and i - start >= MIN_RUN:
            runs.append((start, i, acc))
        start = None
if start is not None and len(rows) - start >= MIN_RUN:
    runs.append((start, len(rows), acc))

cold_static = sum(b - a for a, b, _ in runs)
print(f"{len(rows)} static instructions; {cold_static} in cold runs >= {MIN_RUN} "
      f"({cold_static / max(len(rows), 1) * 100:.1f}% of static size)")
print(f"{len(runs)} cold runs\n")
for a, b, acc in sorted(runs, key=lambda r: -(r[1] - r[0]))[:20]:
    print(f"  {b - a:5d} instrs  {rows[a][0]:x}..{rows[b - 1][0]:x}  {acc:.3f}% dynamic")
    print(f"        head: {rows[a][2]} {rows[a][3][:70]}")
    print(f"        tail: {rows[b - 1][2]} {rows[b - 1][3][:70]}")
