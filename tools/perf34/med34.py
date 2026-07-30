#!/usr/bin/env python3
"""perf study 34 — paired within-round reduction of a `panel34.sh` log.

Each label is differenced against the REFERENCE label's run from the SAME
round, then the per-round deltas are medianed. This is the form study 29's
retraction demanded: a label's round-r number is only ever compared with
another label's round-r number, so a drift that walks across the session
cancels instead of landing on whichever arm happened to occupy that slot.

    python3 tools/perf34/med34.py <panel.txt> <ref-label> [--since ROUND]

Read the NULL row as the error bar: an arm delta means nothing unless it
clears the null's spread.
"""

import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(
    r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+i/n=\s*([\d.]+)\s+cyc/n=\s*([\d.]+)"
    r"\s+br/n=\s*([\d.]+)\s+misp/n=\s*([\d.]+)\s+loads/n=\s*([\d.]+)"
)
METRICS = ["i/n", "cyc/n", "br/n", "misp/n", "loads/n"]


def main() -> None:
    path, ref = sys.argv[1], sys.argv[2]
    since = 0
    if "--since" in sys.argv:
        since = int(sys.argv[sys.argv.index("--since") + 1])

    # data[round][label] = (nodes, i/n, cyc/n, br/n, misp/n, loads/n)
    data: dict[int, dict[str, tuple]] = defaultdict(dict)
    for line in open(path):
        m = ROW.match(line)
        if not m:
            continue
        r = int(m.group(1))
        if r < since:
            continue
        data[r][m.group(2)] = (int(m.group(3)),) + tuple(float(m.group(i)) for i in range(4, 9))

    labels: list[str] = []
    for r in data.values():
        for lab in r:
            if lab not in labels:
                labels.append(lab)
    labels = [ref] + [x for x in labels if x != ref]

    print(f"reference = {ref}   rounds with a complete pair are used")
    print(f"{'label':<12} {'n':>3} " + " ".join(f"{k:>19}" for k in METRICS) + "   nodes")
    for lab in labels:
        deltas: dict[str, list[float]] = {k: [] for k in METRICS}
        nodeset = set()
        for r, row in sorted(data.items()):
            if lab not in row or ref not in row:
                continue
            nodeset.add(row[lab][0])
            for i, k in enumerate(METRICS):
                a, b = row[lab][1 + i], row[ref][1 + i]
                if b:
                    deltas[k].append(100.0 * (a - b) / b)
        n = len(deltas["cyc/n"])
        if n == 0:
            continue
        cells = []
        for k in METRICS:
            med = statistics.median(deltas[k])
            lo, hi = min(deltas[k]), max(deltas[k])
            cells.append(f"{med:+7.3f}%[{lo:+.2f},{hi:+.2f}]".rjust(19))
        nodes = "/".join(str(x) for x in sorted(nodeset))
        print(f"{lab:<12} {n:>3} " + " ".join(cells) + f"   {nodes}")


if __name__ == "__main__":
    main()
