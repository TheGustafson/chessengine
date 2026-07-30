#!/usr/bin/env python3
"""perf study 29 -- medians and paired deltas from panel.txt.

Reads the round-robin panel and reports, per arm and per metric, the median
over rounds and the delta against a named control. The control comes FIRST in
the output so the byte-identical / arms-off null is read before any arm.
"""

import re
import statistics as st
import sys
from collections import defaultdict

LINE = re.compile(
    r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+i/n=\s*([\d.]+)\s+cyc/n=\s*([\d.]+)\s+"
    r"loads/n=\s*([\d.]+)\s+miss/n=\s*([\d.]+)\s+rate=([\d.]+)%\s+stall/n=\s*([\d.]+)"
)
COLS = ["i/n", "cyc/n", "loads/n", "miss/n", "rate%", "stall/n"]


def main():
    path = sys.argv[1]
    ctrl = sys.argv[2]
    vals = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        m = LINE.match(line.strip())
        if not m:
            continue
        arm = m.group(2)
        for i, c in enumerate(COLS):
            vals[arm][c].append(float(m.group(4 + i)))
    order = [ctrl] + [a for a in vals if a != ctrl]
    n = len(vals[ctrl]["cyc/n"])
    print(f"# rounds n={n}  control={ctrl}")
    head = f"{'arm':10s}" + "".join(f"{c:>13s}{'  d%':>8s}" for c in COLS)
    print(head)
    base = {c: st.median(vals[ctrl][c]) for c in COLS}
    for arm in order:
        row = f"{arm:10s}"
        for c in COLS:
            m = st.median(vals[arm][c])
            d = 100 * (m - base[c]) / base[c]
            row += f"{m:13.3f}{d:+8.3f}"
        print(row)
    # spread of the control, the honest noise floor per metric
    print("\n# control spread (min..max, and half-range as % of median)")
    for c in COLS:
        v = vals[ctrl][c]
        half = 100 * (max(v) - min(v)) / 2 / st.median(v)
        print(f"  {c:8s} {min(v):11.3f} .. {max(v):11.3f}   +/-{half:6.3f}%")


if __name__ == "__main__":
    main()
