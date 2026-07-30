#!/usr/bin/env python3
"""perf study 41 — reduce a front-end census into per-node rates.

    python3 tools/perf41/reduce41.py <census.txt> [label] [--csv]

One row per (label, event): median / min / max per-node rate over reps, plus
the relative spread. `SCALED[...]` markers written by `census41.sh` mean perf
multiplexed that counter and the row is NOT quotable — they are surfaced loudly
rather than averaged away.
"""

import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(r"^r(\d+)\s+(\S+)\s+(\S+)\s+nodes=(\d+)\s+(.*)$")


def main() -> None:
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    nodesets: set[int] = set()
    scaled: set[str] = set()
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        label = m.group(2)
        if want and label != want:
            continue
        nodesets.add(int(m.group(4)))
        for tok in m.group(5).split():
            if tok.startswith("SCALED["):
                scaled.add(tok)
            elif "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    vals[(label, k)].append(float(v))
                except ValueError:
                    pass
    if scaled:
        print("!! MULTIPLEXED COUNTERS (not quotable): " + " ".join(sorted(scaled)))
    print(f"node counts seen: {sorted(nodesets)}")
    print()
    print(f"{'label':<8} {'event':<48} {'n':>3} {'median/node':>13} {'min':>13} {'max':>13} {'spread%':>8}")
    for (label, ev), xs in sorted(vals.items()):
        med = statistics.median(xs)
        spread = 100.0 * (max(xs) - min(xs)) / med if med else 0.0
        print(
            f"{label:<8} {ev:<48} {len(xs):>3} {med:>13.4f} {min(xs):>13.4f} "
            f"{max(xs):>13.4f} {spread:>7.2f}%"
        )


if __name__ == "__main__":
    main()
