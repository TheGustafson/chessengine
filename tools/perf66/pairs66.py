#!/usr/bin/env python3
"""perf study 66 — one named PAIRWISE contrast out of a panel file.

`med66.py` quotes everything against one reference. The decomposition this
study needs is a chain of differences (map alone; permutation on top of the
map; packing on top of the map), and each link has to be paired WITHIN ROUND
or the box's load leaks in.

  python3 tools/perf66/pairs66.py <panel.tsv> <arm> <base>
"""

import sys
from collections import defaultdict
from statistics import median

rows = defaultdict(dict)
counters = []
for line in open(sys.argv[1]):
    r, m, c, v = line.rstrip("\n").split("\t")
    rows[(int(r), c)][m] = float(v)
    if c not in counters:
        counters.append(c)

arm, base = sys.argv[2], sys.argv[3]
print(f"{arm} vs {base}")
for c in counters:
    ratios = [
        v[arm] / v[base]
        for v in (rows[k] for k in rows if k[1] == c)
        if arm in v and base in v and v[base]
    ]
    if not ratios:
        continue
    neg = sum(1 for x in ratios if x < 1.0)
    print(f"  {c:36s} {(median(ratios) - 1) * 100:+8.3f}% {neg:2d}/{len(ratios):<2d} neg")
