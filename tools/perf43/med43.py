#!/usr/bin/env python3
"""perf study 43 — reduce a panel43 file: paired within-round deltas, median.

Every label is differenced against the REFERENCE label's run FROM THE SAME
ROUND, then the medians of those per-round percentage deltas are reported with
the full [min, max] range and the SIGN COUNT.  The sign count is the honest
part: study 29's retraction turned on a median that looked clean while the
per-round signs were 5-9 of 16.

    python3 tools/perf43/med43.py <panel.txt> <reference-label>
"""
import re, sys, statistics as st

path, ref = sys.argv[1], sys.argv[2]
rows = {}
pat = re.compile(
    r"r(\d+)\s+(\S+)\s+nodes=(\d+)\s+cyc/n=\s*([\d.]+)\s+i/n=\s*([\d.]+)\s+stall/n=\s*([\d.]+)"
)
for line in open(path):
    m = pat.match(line.strip())
    if m:
        r, lab, n, c, i, s = m.groups()
        rows.setdefault(int(r), {})[lab] = (int(n), float(c), float(i), float(s))

labels, cols = [], ("cyc/n", "i/n", "stall/n")
for r in rows:
    for lab in rows[r]:
        if lab not in labels:
            labels.append(lab)
nodes = {n for r in rows for (n, *_ ) in [rows[r][l] for l in rows[r]]}
print(f"rounds={len(rows)}  node counts seen: {sorted(nodes)}")
print(f"reference = {ref}\n")
print(f"{'label':<14} {'n':>3}  " + "  ".join(f"{c:>34}" for c in cols))
for lab in labels:
    per = {k: [] for k in range(3)}
    for r in sorted(rows):
        if ref not in rows[r] or lab not in rows[r]:
            continue
        base, arm = rows[r][ref], rows[r][lab]
        for k in range(3):
            per[k].append(100.0 * (arm[k + 1] - base[k + 1]) / base[k + 1])
    if not per[0]:
        continue
    cells = []
    for k in range(3):
        v = per[k]
        neg = sum(1 for x in v if x < 0)
        cells.append(f"{st.median(v):+8.4f}% [{min(v):+7.3f},{max(v):+7.3f}] {neg:2d}/{len(v)}-")
    print(f"{lab:<14} {len(per[0]):>3}  " + "  ".join(cells))
print("\nabsolute medians")
for lab in labels:
    for k, name in enumerate(cols):
        v = [rows[r][lab][k + 1] for r in sorted(rows) if lab in rows[r]]
        print(f"  {lab:<14} {name:<9} {st.median(v):12.4f}")
