#!/usr/bin/env python3
"""perf study 40 — paired within-round reduction, with a SIGN COUNT.

    python3 tools/perf40/med40.py <panel.txt> <ref-label> [--since R] [--metric M]

Same pairing as `tools/perf37/med37.py` — every label is differenced against
the REFERENCE label's run from the SAME ROUND, then the per-round deltas are
reduced — plus the three things a sub-1% cycles question needs:

* **sign count** (`k/n`), how many rounds put the arm on one side of zero.
  Study 34 and 38 both retired `cycles:u` on its SPREAD. Spread is the wrong
  statistic when the noise is common-mode: a within-round pair cancels it, and
  what survives is a vote. A null must come back a coin flip; an arm that is
  real does not.
* a **two-sided sign test p-value** on that count (exact binomial), so the vote
  is read against chance rather than by eye.
* **median, mean and min** — the coordinator's "median AND min" rule.

Read the NULL row first. If the null's sign count is not near n/2, the panel is
measuring position-in-round and nothing below it is quotable.
"""

import math
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


def sign_p(k: int, n: int) -> float:
    """Two-sided exact binomial p for k of n votes on one side (p0 = 1/2)."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def main() -> None:
    path, ref = sys.argv[1], sys.argv[2]
    args = sys.argv
    since = int(args[args.index("--since") + 1]) if "--since" in args else 0
    only = args[args.index("--metric") + 1] if "--metric" in args else None
    data = load(path, since)

    labels: list[str] = []
    metrics: list[str] = []
    for row in data.values():
        for lab, cols in row.items():
            if lab not in labels:
                labels.append(lab)
            for k in cols:
                if k != "nodes" and k not in metrics and (only is None or k == only):
                    metrics.append(k)
    labels = [ref] + [x for x in labels if x != ref]

    base = {}
    for k in metrics:
        vals = [row[ref][k] for row in data.values() if ref in row and k in row[ref]]
        if vals:
            base[k] = statistics.median(vals)
    print(f"reference = {ref}   rounds >= {since}")
    print("reference per-node medians: " + "  ".join(f"{k}={base[k]:.4f}" for k in metrics))
    nodesets = {int(c["nodes"]) for row in data.values() for c in row.values()}
    print(f"node counts seen: {sorted(nodesets)}")
    print()

    for k in metrics:
        print(f"--- {k} ---")
        print(
            f"{'label':<8} {'n':>4} {'median':>9} {'mean':>9} {'min':>9} "
            f"{'max':>9} {'neg/n':>9} {'p':>8}  {'abs/node':>10}"
        )
        for lab in labels:
            d, absv = [], []
            for _, cols in sorted(data.items()):
                if lab not in cols or ref not in cols:
                    continue
                a, b = cols[lab].get(k), cols[ref].get(k)
                if a is not None and b:
                    d.append(100.0 * (a - b) / b)
                    absv.append(a)
            if not d:
                continue
            neg = sum(1 for x in d if x < 0)
            med = statistics.median(d)
            print(
                f"{lab:<8} {len(d):>4} {med:>+8.3f}% {statistics.fmean(d):>+8.3f}% "
                f"{min(d):>+8.3f}% {max(d):>+8.3f}% {neg:>4}/{len(d):<4} "
                f"{sign_p(neg, len(d)):>8.4f}  {statistics.median(absv):>10.3f}"
            )
        print()


if __name__ == "__main__":
    main()
