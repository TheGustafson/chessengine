#!/usr/bin/env python3
"""perf study 8 — parse a pstat8.sh log into per-arm per-node counters.

    python3 parse8.py <log> [ref-arm]

Prints, per arm: median nodes, per-node instructions / cycles / branches /
branch-misses, NPS, and (if a reference arm is named) the per-round ratio
median against it.  instructions:u is deterministic, so its spread is the
instrument's own self-check: anything above ~0.01% means the arms searched
different trees and the comparison is invalid.
"""

import statistics
import sys
from collections import defaultdict

log = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else None

rows = []
for line in open(log):
    if not line.startswith("round="):
        continue
    d = dict(kv.split("=", 1) for kv in line.split())
    rows.append({k: (int(v) if v.isdigit() else v) for k, v in d.items()})

arms = []
for r in rows:
    if r["arm"] not in arms:
        arms.append(r["arm"])

by = defaultdict(list)
for r in rows:
    by[r["arm"]].append(r)

FIELDS = ["instructions", "cycles", "branches", "branch_misses"]

print(f"{'arm':<18}{'nodes':>12}{'rounds':>7}", end="")
for f in FIELDS:
    print(f"{f + '/node':>18}", end="")
print(f"{'NPS(med)':>12}{'instr spread':>14}")

per_node = {}
for a in arms:
    rs = by[a]
    nodes = statistics.median(r["nodes"] for r in rs)
    print(f"{a:<18}{nodes:>12.0f}{len(rs):>7}", end="")
    per_node[a] = {}
    for f in FIELDS:
        vals = [r[f] / r["nodes"] for r in rs]
        per_node[a][f] = statistics.median(vals)
        print(f"{statistics.median(vals):>18.1f}", end="")
    nps = statistics.median(r["nps"] for r in rs)
    ins = [r["instructions"] for r in rs]
    spread = (max(ins) - min(ins)) / statistics.median(ins) if len(ins) > 1 else 0.0
    print(f"{nps:>12.0f}{spread * 100:>13.4f}%")

if ref and ref in per_node:
    print(f"\nratios vs {ref} (per-node medians):")
    for a in arms:
        if a == ref:
            continue
        print(f"  {a:<18}", end="")
        for f in FIELDS:
            print(f"  {f}={per_node[a][f] / per_node[ref][f]:.4f}", end="")
        print()

    # per-round paired NPS ratio (the only load-tolerant way to quote NPS)
    print(f"\nper-round paired NPS ratio vs {ref}:")
    nrounds = min(len(by[a]) for a in arms)
    for a in arms:
        if a == ref:
            continue
        rr = []
        for i in range(nrounds):
            rr.append(by[a][i]["nps"] / by[ref][i]["nps"])
        rr.sort()
        print(f"  {a:<18} median={statistics.median(rr):.4f}  range=[{rr[0]:.4f},{rr[-1]:.4f}]")
