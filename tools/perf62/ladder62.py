#!/usr/bin/env python3
"""Perf study 62 — price the shape-dispatch ladder rung by rung.

Reads the THR_PAIR histogram and the whole-frame instruction totals and prints
the marginal instructions per converted fold and per converted row, which is
what tells Route A (continue the table) from Route B (fix the generic kernel).
"""
import re
import sys

h = {}
for line in open(sys.argv[1] if len(sys.argv) > 1 else "perfout62/fc.txt"):
    m = re.match(r"FOLDCOUNT thr_pair\.(\d+)a(\d+)s=(\d+)", line)
    if m:
        h[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
TOT = sum(h.values())
NODES = 5001562
base = 27911705229
runs = {"s62a": 27812155259, "s62b": 27717104442, "s62c": 27602585143}


def grid(n):
    return {(a, s) for a in range(n + 1) for s in range(n + 1)} - {(0, 0)}


g3 = grid(3)
prev, prevwin = g3, 0
print("MARGINAL per rung")
print("%-6s %8s %10s %12s %9s %9s %9s" % ("rung", "folds", "rows", "dInsn", "i/node", "i/fold", "i/row"))
for lab, n in [("s62a", 4), ("s62b", 5), ("s62c", 6)]:
    new = grid(n) - prev
    f = sum(h.get(k, 0) for k in new)
    r = sum((k[0] + k[1]) * h.get(k, 0) for k in new)
    win = base - runs[lab]
    d = win - prevwin
    print("%-6s %8d %10d %12d %9.2f %9.1f %9.2f" % (lab, f, r, d, d / NODES, d / f, d / r))
    prev, prevwin = grid(n), win

print()
print("CUMULATIVE vs base")
for lab, n in [("s62a", 4), ("s62b", 5), ("s62c", 6)]:
    new = grid(n) - g3
    f = sum(h.get(k, 0) for k in new)
    r = sum((k[0] + k[1]) * h.get(k, 0) for k in new)
    win = base - runs[lab]
    print(
        "  %-5s folds=%7d (%5.2f%% of channel) rows=%8d  win=%11d = %6.2f i/n (%+.3f%%)"
        % (lab, f, 100 * f / TOT, r, win, win / NODES, -100.0 * win / base)
    )

print()
print("STUDY 61's OWN FAMILIES (for the same fit)")
fams = {
    "one-sided": {(1, 0), (0, 1), (2, 0), (0, 2)},
    "three-row": {(3, 0), (0, 3), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)},
}
for name, fam in fams.items():
    f = sum(h[k] for k in fam)
    r = sum((k[0] + k[1]) * h[k] for k in fam)
    print("  %-10s folds=%8d rows=%9d rows/fold=%.3f" % (name, f, r, r / f))

print()
print("WHAT IS LEFT after grid 0..=6")
left = {k: v for k, v in h.items() if k not in grid(6)}
lf = sum(left.values())
lr = sum((k[0] + k[1]) * v for k, v in left.items())
print("  folds=%d (%.2f%% of channel)  rows>=%d (%.2f%% of all threat rows)" % (
    lf, 100 * lf / TOT, lr, 100 * lr / sum((k[0] + k[1]) * v for k, v in h.items())))
print("  (rows is a LOWER bound: every bucket with a 7 index is capped)")
