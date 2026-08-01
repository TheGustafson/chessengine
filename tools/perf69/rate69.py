#!/usr/bin/env python3
"""perf study 69 — the EXECUTION RATE of named branch sites, from the LBR dump.

  python3 tools/perf69/rate69.py <brstacksym.txt> <branches-total> <nodes> \
      <symbol> <+0xoff> [+0xoff ...]

An LBR stack is a uniform sample of the recent branch stream, so a site's share
of all sampled entries times the frame's `branches:u` total is its absolute
execution count. That converts "22 references in the disassembly" into
"executions per node", which is the only form an arm may be priced in
(study 68 §4.3).

Prints per site: sampled entries, executions/node, mispredict rate; then the
total, which is the arm's budget.
"""
import collections
import sys

path, btot, nodes = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
sym = sys.argv[4]
offs = sys.argv[5:]

seen = collections.Counter()
miss = collections.Counter()
tot = 0
for line in open(path, errors="replace"):
    for ent in line.split():
        f = ent.split("/")
        if len(f) < 3:
            continue
        tot += 1
        seen[f[0]] += 1
        miss[f[0]] += f[2] == "M"

# entries -> real branch executions; the dump may hold several reps, and both
# `btot` and `nodes` are per-rep, so the reps cancel in the ratio.
per_entry = btot / (tot / (tot / tot))  # placeholder, replaced below
scale = btot / tot  # branches per entry, per rep-equivalent
print(f"# {tot} sampled entries <-> {btot:.0f} branches/rep  =>  {scale:.1f} branches/entry")
print(f"{'site':<28}{'entries':>9}{'exec/node':>12}{'misp':>8}")
te = tm = 0.0
for o in offs:
    key = sym + o
    s, m = seen[key], miss[key]
    te += s
    tm += m
    print(f"{o:<28}{s:>9}{s*scale/nodes:>12.3f}{(100.0*m/s if s else 0):>7.1f}%")
print(f"{'TOTAL':<28}{te:>9.0f}{te*scale/nodes:>12.3f}{(100.0*tm/te if te else 0):>7.1f}%")
