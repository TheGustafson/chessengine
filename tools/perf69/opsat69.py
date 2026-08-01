#!/usr/bin/env python3
"""perf study 69 — how often a LIST OF EXACT INSTRUCTION OFFSETS executes.

  python3 tools/perf69/opsat69.py <ibs.txt> <ops-per-node> <sym> <+0xoff ...>
  python3 tools/perf69/opsat69.py <ibs.txt> <ops-per-node> <sym> --stdin

`ibsops67.sh` samples DISPATCHED ops with `cnt_ctl=0`, so its per-rip histogram
is proportional to op EXECUTION COUNT with no skid. A site's share of all
samples times the frame's ops/node is its executions per node.

This is the instrument LBR cannot be: **an LBR stack records only TAKEN
branches**, so a never-taken guard branch -- exactly what an initialised
`LazyLock` check is -- has zero LBR entries and is invisible there. Measured,
not assumed: all ten `attacks::TABLES` guard branches in `search::<NonPv>` have
0 of 418,976 LBR entries.
"""
import collections
import sys

path, opn, sym = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rest = sys.argv[4:]
offs = [l.strip() for l in sys.stdin if l.strip()] if rest == ["--stdin"] else rest

hist = collections.Counter()
tot = 0
for line in open(path, errors="replace"):
    p = line.split(None, 1)
    if len(p) < 2:
        continue
    tot += 1
    hist[p[1].strip()] += 1

per = opn / tot
print(f"# {tot} IBS op samples  <->  {opn:.1f} ops/node   =>  {per:.5f} ops/node per sample")
print(f"{'site':<12}{'samples':>9}{'exec/node':>12}")
ts = 0
for o in offs:
    s = hist[sym + o]
    ts += s
    print(f"{o:<12}{s:>9}{s*per:>12.4f}")
print(f"{'TOTAL':<12}{ts:>9}{ts*per:>12.4f}   ({len(offs)} sites)")
