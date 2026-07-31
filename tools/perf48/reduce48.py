#!/usr/bin/env python3
"""perf study 48 -- reduce the load census.

Shares are computed WITHIN each run (study 44 Sec 0.3): every group carries
cycles:u and instructions:u, so each run yields a self-contained ratio measured
at the same slot under the same instantaneous load.  Absolute per-node figures
use the study's node count.
"""
import sys, re, statistics
from collections import defaultdict

NODES = 5001562
path = sys.argv[1] if len(sys.argv) > 1 else "docs/research/perf-artifacts-48/census.txt"

runs = []          # list of dict event -> count
cur = None
for line in open(path):
    line = line.rstrip("\n")
    if line.startswith("### "):
        if cur:
            runs.append(cur)
        cur = {}
        continue
    if cur is None:
        continue
    # perf stat -x, :  value,unit,event,runtime,pct,...
    f = line.split(",")
    if len(f) >= 3 and re.fullmatch(r"\d+", f[0]) and f[2]:
        cur[f[2]] = int(f[0])
if cur:
    runs.append(cur)

per_node = defaultdict(list)
share    = defaultdict(list)
per_inst = defaultdict(list)
for r in runs:
    cyc = r.get("cycles:u")
    ins = r.get("instructions:u")
    for ev, v in r.items():
        per_node[ev].append(v / NODES)
        if cyc:
            share[ev].append(v / cyc)
        if ins:
            per_inst[ev].append(v / ins)

print(f"runs parsed: {len(runs)}   nodes: {NODES}")
print()
print(f"{'event':<62} {'n':>3} {'per node':>12} {'% of cycles':>12} {'% of insn':>10}")
for ev in sorted(per_node):
    n = len(per_node[ev])
    pn = statistics.median(per_node[ev])
    sh = 100 * statistics.median(share[ev]) if share[ev] else float("nan")
    pi = 100 * statistics.median(per_inst[ev]) if per_inst[ev] else float("nan")
    print(f"{ev:<62} {n:>3} {pn:>12.2f} {sh:>12.2f} {pi:>10.2f}")

# derived
def med(ev):
    return statistics.median(per_node[ev]) if ev in per_node else None
ld = med("ls_dispatch.ld_dispatch:u")
st = med("ls_dispatch.store_dispatch:u")
ldst = med("ls_dispatch.ld_st_dispatch:u")
l1 = med("L1-dcache-loads:u")
ins = med("instructions:u")
lnc = med("ex_no_retire.load_not_complete:u")
print()
print("DERIVED")
if ld and st:
    print(f"  memory ops/node                  = {ld+st:10.2f}   ({100*(ld+st)/ins:.2f} % of instructions)")
if ldst:
    print(f"  ld_st_dispatch/node              = {ldst:10.2f}   (ld+st = {ld+st:.2f}, delta {ldst-(ld+st):+.2f})")
if l1:
    print(f"  L1-dcache-loads (ls_dc_accesses) = {l1:10.2f}   (ld+st = {(ld+st):.2f}, ratio {l1/(ld+st):.4f})")
if lnc and ld:
    print(f"  load-stall cycles per ld_dispatch= {lnc/ld:10.4f}")
