#!/usr/bin/env python3
"""perf study 37 — paired within-round reduction of a `panel37.sh` log.

    python3 tools/perf37/med37.py <panel.txt> <ref-label> [--since ROUND]

Same reduction as study 34's `med34.py` — each label is differenced against the
REFERENCE label's run from the SAME ROUND, then the per-round deltas are
medianed — but column-agnostic, because this study runs two different event
sets through the same harness.

Read the NULL row as the error bar. An arm delta means nothing unless it clears
the null's spread on the SAME counter; on this box that disqualifies cycles
outright (studies 33/34/35/36) and leaves instructions and branches exact.

Absolute per-node values for the reference are printed too: a TLB counter's
INTERESTING form is often the level (`2m_l2_hit` going from 0 to 9/node is the
whole result) rather than a percentage of a near-zero base.
"""

import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+(.*)$")


def load(path: str, since: int) -> dict[int, dict[str, dict[str, float]]]:
    data: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        r = int(m.group(1))
        if r < since:
            continue
        row = {"nodes": float(m.group(3))}
        for tok in m.group(4).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        data[r][m.group(2)] = row
    return data


def main() -> None:
    path, ref = sys.argv[1], sys.argv[2]
    since = int(sys.argv[sys.argv.index("--since") + 1]) if "--since" in sys.argv else 0
    data = load(path, since)

    labels: list[str] = []
    metrics: list[str] = []
    for row in data.values():
        for lab, cols in row.items():
            if lab not in labels:
                labels.append(lab)
            for k in cols:
                if k != "nodes" and k not in metrics:
                    metrics.append(k)
    labels = [ref] + [x for x in labels if x != ref]

    base = {}
    for k in metrics:
        vals = [row[ref][k] for row in data.values() if ref in row and k in row[ref]]
        if vals:
            base[k] = statistics.median(vals)
    print(f"reference = {ref}   (rounds with a complete pair only)")
    print("reference per-node medians: " + "  ".join(f"{k}={base[k]:.4f}" for k in metrics))
    print()
    w = max(20, max(len(k) for k in metrics) + 2)
    print(f"{'label':<12} {'n':>3} " + " ".join(k.rjust(w) for k in metrics) + "   nodes")
    for lab in labels:
        deltas: dict[str, list[float]] = {k: [] for k in metrics}
        nodeset = set()
        for row in sorted(data.items()):
            cols = row[1]
            if lab not in cols or ref not in cols:
                continue
            nodeset.add(int(cols[lab]["nodes"]))
            for k in metrics:
                a, b = cols[lab].get(k), cols[ref].get(k)
                if a is not None and b:
                    deltas[k].append(100.0 * (a - b) / b)
        n = max((len(v) for v in deltas.values()), default=0)
        if n == 0:
            continue
        cells = []
        for k in metrics:
            d = deltas[k]
            if not d:
                cells.append("-".rjust(w))
                continue
            cells.append(
                f"{statistics.median(d):+7.3f}%[{min(d):+.2f},{max(d):+.2f}]".rjust(w)
            )
        nodes = "/".join(str(x) for x in sorted(nodeset))
        print(f"{lab:<12} {n:>3} " + " ".join(cells) + f"   {nodes}")


if __name__ == "__main__":
    main()
