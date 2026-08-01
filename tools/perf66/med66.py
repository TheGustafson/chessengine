#!/usr/bin/env python3
"""perf study 66 — read `panel66.sh`'s rows and report each arm against the
reference mode, PAIRED WITHIN ROUND.

Per round the ratio arm/ref is formed first and the MEDIAN of ratios is
reported, so a round in which the box was busy scales both sides and cancels.
The sign count is printed beside it: at n rounds a median with a 50/50 sign
count is a coin flip whatever its value.

The last block prints every arm against the arm BEFORE it in the mode list,
which is where the isolated contrasts live (compaction against the
slot-preserving pack; the transpose against an identity map).

  python3 tools/perf66/med66.py <panel.tsv> [reference-label]
"""

import sys
from collections import defaultdict
from statistics import median

rows = defaultdict(dict)  # (round, counter) -> {mode: value}
counters = []
modes = []
for line in open(sys.argv[1]):
    r, m, c, v = line.rstrip("\n").split("\t")
    rows[(int(r), c)][m] = float(v)
    if c not in counters:
        counters.append(c)
    if m not in modes:
        modes.append(m)

REF = sys.argv[2] if len(sys.argv) > 2 else modes[0]


def contrast(counter, arm, ref):
    ratios = [
        v[arm] / v[ref]
        for v in (rows[k] for k in rows if k[1] == counter)
        if arm in v and ref in v and v[ref]
    ]
    if not ratios:
        return None
    neg = sum(1 for x in ratios if x < 1.0)
    return (median(ratios) - 1) * 100, neg, len(ratios)


arms = [m for m in modes if m != REF]
print(f"vs {REF}")
print(f"{'counter':38s} {'ref (median)':>16s}  " + "  ".join(f"{a:>22s}" for a in arms))
for c in counters:
    refs = [v[REF] for v in (rows[k] for k in rows if k[1] == c) if REF in v]
    if not refs:
        continue
    cells = []
    for a in arms:
        got = contrast(c, a, REF)
        cells.append(" " * 22 if got is None else f"{got[0]:+8.3f}% {got[1]:2d}/{got[2]:<2d} neg")
    print(f"{c:38s} {median(refs):16.0f}  " + "  ".join(cells))

for i in range(1, len(modes) - 1):
    a, b = modes[i + 1], modes[i]
    print(f"\n{a} vs {b}:")
    for c in counters:
        got = contrast(c, a, b)
        if got:
            print(f"  {c:36s} {got[0]:+8.3f}% {got[1]:2d}/{got[2]:<2d} neg")
