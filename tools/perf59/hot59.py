#!/usr/bin/env python3
"""perf study 59 — the hottest INSTRUCTIONS inside one symbol, and a mnemonic
histogram of where its cycles go.

    perf annotate -i <data> --stdio -s <symbol> | python3 tools/perf59/hot59.py [topN]

`perf annotate`'s default output is address-ordered, which for a 7.9 KB
PGO-inlined function is unreadable. Sorting by percent, and bucketing by
mnemonic, is what makes "12.40% of the engine" into "these six loads".
"""
import collections
import re
import sys

top = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows = []
for line in sys.stdin:
    m = re.match(r"\s*([\d.]+)\s*:\s+([0-9a-f]+):\s+(\S+)\s*(.*)", line)
    if not m:
        continue
    rows.append((float(m.group(1)), m.group(2), m.group(3), m.group(4).strip()))

total = sum(r[0] for r in rows)
print(f"{len(rows)} instructions, {total:.2f}% of symbol accounted\n")

by_mn = collections.Counter()
cnt_mn = collections.Counter()
for pct, _, mn, _ in rows:
    by_mn[mn] += pct
    cnt_mn[mn] += 1
print(f"{'mnemonic':<20}{'% of symbol':>12}{'count':>8}")
for mn, pct in by_mn.most_common(22):
    print(f"{mn:<20}{pct:>12.2f}{cnt_mn[mn]:>8}")

print(f"\ntop {top} instructions")
print(f"{'%':>7}  {'addr':<10}{'insn'}")
for pct, addr, mn, ops in sorted(rows, reverse=True)[:top]:
    print(f"{pct:>7.2f}  {addr:<10}{mn} {ops}")
