#!/usr/bin/env python3
"""perf study 67 — paired medians of a panel67 CSV.

  python3 tools/perf67/med67.py <csv> <base-label> [arm-label ...]

Reports, per counter, the WITHIN-ROUND paired delta (arm - base) / base as a
median over rounds, plus the count of rounds in which the delta was negative.
A within-round pairing is what makes a shared box usable: both arms saw the
same neighbours in the same minute. Levels are printed only as the base's
median, never per arm -- single-run levels of the miss-class counters swing by
more than the effects this study measures (study 27 sec 3).
"""
import collections
import csv
import statistics
import sys

path, base = sys.argv[1], sys.argv[2]
arms = sys.argv[3:]
rows = list(csv.DictReader(open(path)))
cols = [c for c in rows[0] if c not in ("round", "label")]
by = collections.defaultdict(dict)
for r in rows:
    by[r["round"]][r["label"]] = r
labels = arms or sorted({r["label"] for r in rows} - {base})

print(f"{'counter':<16}{'base median':>16}  {'arm':<12}{'paired delta':>15}{'neg/n':>8}")
for arm in labels:
    for c in cols:
        d, bv = [], []
        for rd in by.values():
            if base in rd and arm in rd and rd[base][c] and rd[arm][c]:
                b, a = float(rd[base][c]), float(rd[arm][c])
                bv.append(b)
                d.append(100.0 * (a - b) / b)
        if not d:
            continue
        neg = sum(1 for x in d if x < 0)
        print(f"{c:<16}{statistics.median(bv):>16.0f}  {arm:<12}"
              f"{statistics.median(d):>+14.3f}%{neg:>5}/{len(d)}")
    print()
