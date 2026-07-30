#!/usr/bin/env python3
"""perf study 35 -- reduce the CONCURRENT paired panel (`conc35.sh`).

    python3 tools/perf35/conc35.py <conc.txt> <control>

Each pass in `conc.txt` is one control run and one arm run that executed AT THE
SAME TIME on two physical cores, so their delta is free of the frequency and
memory-phase drift that a sequential panel cannot hold constant. Passes come in
core-swapped twos; this reducer

  1. forms the delta of each pass,
  2. averages the two core assignments of each pair (cancelling core asymmetry),
  3. reports the MEDIAN of those per-pair deltas, the [min,max], and the count.

The control-vs-control arm (a byte-identical copy) is the null: whatever it
reports is the floor below which no arm in the same table means anything.
"""

from __future__ import annotations

import re
import statistics as st
import sys
from collections import defaultdict

LINE = re.compile(
    r"^r(\d+)\s+cpu(\d+)\s+(\S+)\s+nodes=(\d+)\s+i/n=\s*([\d.]+)\s+"
    r"cyc/n=\s*([\d.]+)\s+loads/n=\s*([\d.]+)"
)
COLS = ["i/n", "cyc/n", "loads/n"]


def main():
    path, ctrl = sys.argv[1], sys.argv[2]
    # The file is strictly ordered: ctrl, arm, ctrl, arm, ... two lines per pass.
    lines = []
    for line in open(path):
        m = LINE.match(line.strip())
        if m:
            lines.append((int(m.group(1)), int(m.group(2)), m.group(3),
                          int(m.group(4)),
                          {c: float(m.group(5 + i)) for i, c in enumerate(COLS)}))

    nodes = defaultdict(set)
    for _, _, lab, n, _ in lines:
        nodes[lab].add(n)

    # Walk passes in file order; a pass is (control line, arm line).
    deltas = defaultdict(lambda: defaultdict(list))  # arm -> col -> [d% per pass]
    i = 0
    passes = 0
    while i + 1 < len(lines):
        (_, _, la, _, va), (_, _, lb, _, vb) = lines[i], lines[i + 1]
        i += 2
        if la != ctrl:
            continue
        passes += 1
        for c in COLS:
            deltas[lb][c].append(100 * (vb[c] - va[c]) / va[c])

    print(f"# passes={passes}  control={ctrl}  (each pass = one simultaneous "
          f"control+arm run on two physical cores)")
    for lab in sorted(nodes):
        flag = "" if len(nodes[lab]) == 1 else "   <-- NOT NODE-STABLE"
        print(f"#   {lab:12s} nodes {sorted(nodes[lab])}{flag}")
    print()
    print(f"{'arm':14s}{'n':>4s}" +
          "".join(f"{c + ' d%':>12s}{'[min,max]':>20s}" for c in COLS))
    for arm in sorted(deltas):
        # average the two core assignments of each consecutive pair
        row = f"{arm:14s}"
        n_pairs = 0
        cells = ""
        for c in COLS:
            d = deltas[arm][c]
            paired = [(d[j] + d[j + 1]) / 2 for j in range(0, len(d) - 1, 2)]
            n_pairs = len(paired)
            cells += (f"{st.median(paired):+12.3f}"
                      f"  [{min(paired):+7.3f},{max(paired):+7.3f}]")
        print(row + f"{n_pairs:>4d}" + cells)


if __name__ == "__main__":
    main()
