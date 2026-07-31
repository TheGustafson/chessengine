#!/usr/bin/env python3
"""perf study 58 (harness from 55/53/49) -- paired within-round contrasts off a
blocked Latin panel.

Reads `panel58.sh`'s output: one `### round= pos= label= nodes=` marker per run
followed by that run's `perf stat -x,` CSV lines. `nodes` is carried on the
marker and checked here, because a panel whose node totals are not all equal is
not measuring one search tree and none of its contrasts mean anything.
"""
import math
import os
import re
import statistics
import sys
from collections import defaultdict

NODES = int(os.environ.get("P58_NODES", 5001562))

path, ref = sys.argv[1], sys.argv[2]
labels = sys.argv[3:]

rounds = defaultdict(dict)  # round -> label -> {event: value}
pos = defaultdict(dict)  # label -> round -> position
nodeset = set()
cur = None
for line in open(path):
    line = line.rstrip("\n")
    m = re.match(r"### round=(\d+) pos=(\d+) label=(\S+)(?: nodes=(\S+))?", line)
    if m:
        r, p, lab = int(m.group(1)), int(m.group(2)), m.group(3)
        cur = rounds[r].setdefault(lab, {})
        pos[lab][r] = p
        if m.group(4):
            nodeset.add(m.group(4))
        continue
    if cur is None:
        continue
    f = line.split(",")
    if len(f) >= 3 and re.fullmatch(r"\d+", f[0]) and f[2]:
        cur[f[2]] = int(f[0])

events = sorted({e for r in rounds.values() for d in r.values() for e in d})
good = [r for r, d in sorted(rounds.items()) if all(lb in d and d[lb] for lb in labels + [ref])]
nlab = len(set(labels + [ref]))
good = good[: len(good) // nlab * nlab] if len(good) >= nlab else good
print(f"panel {path}   ref={ref}   complete rounds={len(good)}")
if nodeset:
    print(f"node totals seen: {sorted(nodeset)}"
          + ("" if len(nodeset) == 1 else "   *** NOT NODE-IDENTICAL — contrasts are void ***"))
gset = set(good)
print("mean position over ADJUDICATED rounds (ideal %.2f): " % ((nlab - 1) / 2)
      + "  ".join(f"{lb}={statistics.mean([p for r, p in pos[lb].items() if r in gset]):.2f}"
                  for lb in sorted(pos) if any(r in gset for r in pos[lb])))
print()
base = {e: statistics.median(rounds[r][ref][e] / NODES for r in good) for e in events}
print("REFERENCE per node: " + "  ".join(f"{e}={base[e]:.2f}" for e in events))
print()
hdr = f"{'label':<10}{'event':<34}{'median %':>10}{'trimmed %':>11}{'se':>8}{'neg/n':>9}{'p':>9}"
print(hdr)
for lab in labels:
    for e in events:
        d = []
        for r in good:
            a, b = rounds[r][lab].get(e), rounds[r][ref].get(e)
            if a and b:
                d.append(100.0 * (a - b) / b)
        if not d:
            continue
        n = len(d)
        med = statistics.median(d)
        s = sorted(d)
        k = max(1, n // 10)
        tr = statistics.mean(s[k : n - k]) if n > 2 * k else statistics.mean(s)
        se = statistics.stdev(d) / math.sqrt(n) if n > 1 else 0.0
        neg = sum(1 for x in d if x < 0)
        k2 = min(neg, n - neg)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k2 + 1)) / 2**n)
        print(f"{lab:<10}{e:<34}{med:>10.3f}{tr:>11.3f}{se:>8.3f}{neg:>6}/{n:<3}{p:>9.4f}")
    print()
