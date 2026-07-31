#!/usr/bin/env python3
"""Perf study 62 — the uncapped tail of the threat fold, and the Finny warm
refresh's pair shape (the kernel with no dispatch at all)."""
import re
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "perfout62/fc2.txt"
thr, dual, lens = {}, {}, {"a": {}, "s": {}}
for line in open(src):
    m = re.match(r"FOLDCOUNT thr_pair\.(\d+)a(\d+)s=(\d+)", line)
    if m:
        thr[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    m = re.match(r"FOLDCOUNT dual_pair\.(\d+)a(\d+)s=(\d+)", line)
    if m:
        dual[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    m = re.match(r"FOLDCOUNT thr_lens\.([as])(\d+)=(\d+)", line)
    if m:
        lens[m.group(1)][int(m.group(2))] = int(m.group(3))

TOT = sum(thr.values())
print("threat folds %d" % TOT)
print("\nPER-LIST LENGTH, UNCAPPED — the asymmetry")
print("%4s %10s %8s %10s %8s" % ("n", "adds", "cum%", "subs", "cum%"))
ca = cs = 0
for n in range(0, 17):
    a, s = lens["a"].get(n, 0), lens["s"].get(n, 0)
    ca += a
    cs += s
    print("%4d %10d %7.2f%% %10d %7.2f%%" % (n, a, 100 * ca / TOT, s, 100 * cs / TOT))
print("  adds >= 7: %d = %.2f%%   subs >= 7: %d = %.2f%%" % (
    TOT - sum(lens["a"].get(n, 0) for n in range(7)),
    100 * (TOT - sum(lens["a"].get(n, 0) for n in range(7))) / TOT,
    TOT - sum(lens["s"].get(n, 0) for n in range(7)),
    100 * (TOT - sum(lens["s"].get(n, 0) for n in range(7))) / TOT))

print("\nRECTANGULAR vs SQUARE grids (threat fold), coverage of folds")
for na in (4, 5, 6, 7):
    row = []
    for ns in (4, 5, 6, 7):
        g = {(a, s) for a in range(na + 1) for s in range(ns + 1)} - {(0, 0)}
        c = sum(thr.get(k, 0) for k in g)
        row.append("%5.2f%%(%d)" % (100 * c / TOT, len(g)))
    print("  adds<=%d : %s" % (na, "  ".join(row)))
print("  (columns subs<=4,5,6,7; the >=7 buckets are capped so the last column"
      "\n   is a LOWER bound on what a subs<=7 grid would actually catch)")

DT = sum(dual.values())
print("\nFINNY WARM REFRESH (i8_deltas_dual_inner — NO dispatch at all)")
print("  calls %d" % DT)
for n in (2, 3, 4, 5, 6):
    g = {(a, s) for a in range(n + 1) for s in range(n + 1)} - {(0, 0)}
    c = sum(dual.get(k, 0) for k in g)
    r = sum((k[0] + k[1]) * dual.get(k, 0) for k in g)
    print("  grid 0..=%d (%2d arms): %6.2f%% of calls, %d rows" % (n, len(g), 100 * c / DT, r))
print("  top shapes:")
for k, v in sorted(dual.items(), key=lambda x: -x[1])[:12]:
    print("    (%d,%d) %7d  %5.2f%%" % (k[0], k[1], v, 100 * v / DT))
print("  empty (0,0): %d = %.2f%% — a pure 2560 B copy through the fold" % (
    dual.get((0, 0), 0), 100 * dual.get((0, 0), 0) / DT))
