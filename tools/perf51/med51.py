#!/usr/bin/env python3
"""perf study 51 -- paired within-round contrasts off a blocked Latin panel."""
import sys, re, statistics, math
from collections import defaultdict
import os
NODES = float(os.environ.get("P51_NODES", 5001562))
path, ref = sys.argv[1], sys.argv[2]
labels = sys.argv[3:]
rounds = defaultdict(dict)   # round -> label -> {event: value}
pos = defaultdict(list)
cur = None
for line in open(path):
    line = line.rstrip("\n")
    m = re.match(r"### round=(\d+) pos=(\d+) label=(\S+)", line)
    if m:
        r, p, lab = int(m.group(1)), int(m.group(2)), m.group(3)
        cur = rounds[r].setdefault(lab, {})
        pos[lab].append(p)
        continue
    if cur is None: continue
    f = line.split(",")
    if len(f) >= 3 and re.fullmatch(r"\d+", f[0]) and f[2]:
        cur[f[2]] = int(f[0])
events = sorted({e for r in rounds.values() for d in r.values() for e in d})
# only rounds where every label present
good = [r for r, d in sorted(rounds.items()) if all(l in d and d[l] for l in labels + [ref])]
# whole-block truncation
nlab = len(set(labels + [ref]))
good = good[: len(good) // nlab * nlab] if len(good) >= nlab else good
print(f"panel {path}   ref={ref}   complete rounds={len(good)}")
print("mean position (ideal %.2f): " % ((nlab - 1) / 2)
      + "  ".join(f"{l}={statistics.mean(pos[l]):.2f}" for l in sorted(pos)))
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
            if a and b: d.append(100.0 * (a - b) / b)
        if not d: continue
        n = len(d); med = statistics.median(d)
        s = sorted(d); k = max(1, n // 10)
        tr = statistics.mean(s[k:n-k]) if n > 2 * k else statistics.mean(s)
        se = statistics.stdev(d) / math.sqrt(n) if n > 1 else 0.0
        neg = sum(1 for x in d if x < 0)
        # two-sided exact binomial on the sign count
        k2 = min(neg, n - neg)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k2 + 1)) / 2**n)
        print(f"{lab:<10}{e:<34}{med:>10.3f}{tr:>11.3f}{se:>8.3f}{neg:>6}/{n:<3}{p:>9.4f}")
    print()
