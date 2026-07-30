#!/usr/bin/env python3
"""perf study 35 -- PAIRED WITHIN-ROUND reduction of panel35.txt.

    python3 tools/perf35/med35.py <panel.txt> <control-label>

Study 29's `med29.py` differenced two independent medians. That is what let a
-1.67% cycles "win" at n=13 survive to the brink of shipping: the effect was
position-in-round, and a median-of-arm minus median-of-control cannot see it.
This reducer works only with deltas computed INSIDE a round, where both arms
saw the same box, the same trainer phase and the same thermal state:

    d_r = 100 * (arm_r - ctrl_r) / ctrl_r        for each round r
    report median(d_r), plus min/max of d_r

so a per-round common-mode shift cancels exactly instead of averaging.

Rounds where a label is missing (an aborted run) are dropped from every label,
so every reported delta comes from a complete round.
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
    path, ctrl = sys.argv[1], sys.argv[2]
    # rows[round][label] = {col: value}; nodes[label] = set of node totals
    rows = defaultdict(dict)
    nodes = defaultdict(set)
    for line in open(path):
        m = LINE.match(line.strip())
        if not m:
            continue
        r, arm = int(m.group(1)), m.group(2)
        nodes[arm].add(int(m.group(3)))
        rows[r][arm] = {c: float(m.group(4 + i)) for i, c in enumerate(COLS)}

    labels = sorted({a for r in rows.values() for a in r})
    if ctrl not in labels:
        sys.exit(f"control {ctrl!r} not in {labels}")
    complete = [r for r in sorted(rows) if all(a in rows[r] for a in labels)]
    if not complete:
        sys.exit("no complete rounds")

    print(f"# complete rounds n={len(complete)}  control={ctrl}  "
          f"(dropped {len(rows) - len(complete)} partial)")
    print("# node totals per arm (MUST be a single value each for a "
          "node-identical refactor):")
    for a in labels:
        flag = "" if len(nodes[a]) == 1 else "   <-- NOT NODE-STABLE"
        print(f"#   {a:14s} {sorted(nodes[a])}{flag}")

    order = [ctrl] + [a for a in labels if a != ctrl]
    print()
    head = f"{'arm':14s}" + "".join(f"{c:>11s}{'  d%':>8s}{'    [min,max]':>17s}"
                                    for c in COLS)
    print(head)
    for arm in order:
        row = f"{arm:14s}"
        for c in COLS:
            med = st.median([rows[r][arm][c] for r in complete])
            ds = [100 * (rows[r][arm][c] - rows[r][ctrl][c]) / rows[r][ctrl][c]
                  for r in complete]
            row += (f"{med:11.3f}{st.median(ds):+8.3f}"
                    f"  [{min(ds):+6.3f},{max(ds):+6.3f}]")
        print(row)

    print("\n# control's own round-to-round spread (the honest noise floor):")
    for c in COLS:
        v = [rows[r][ctrl][c] for r in complete]
        half = 100 * (max(v) - min(v)) / 2 / st.median(v)
        print(f"#   {c:8s} {min(v):11.3f} .. {max(v):11.3f}   +/-{half:6.3f}%")


if __name__ == "__main__":
    main()
