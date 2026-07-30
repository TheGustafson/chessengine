#!/usr/bin/env python3
"""perf study 40 — the shape of a panel's paired noise, not just its width.

    python3 tools/perf40/spread40.py <panel.txt> <ref> <label> [metric]

Studies 33-39 all retired `cycles:u` on its RANGE. A range is the wrong summary
if the noise is heavy-tailed: a few wild rounds set it while the bulk is tight,
and a trimmed statistic then converges far faster than the range suggests. This
prints the per-round paired deltas sorted, with sd, IQR, a 20% trimmed mean and
the standard error each implies — i.e. how many rounds a given effect needs.
"""

import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+(.*)$")


def main() -> None:
    path, ref, lab = sys.argv[1], sys.argv[2], sys.argv[3]
    metric = sys.argv[4] if len(sys.argv) > 4 else "cycles:u"
    data: dict[int, dict[str, float]] = defaultdict(dict)
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        for tok in m.group(4).split():
            if tok.startswith(metric + "="):
                data[int(m.group(1))][m.group(2)] = float(tok.split("=", 1)[1])
    d = sorted(
        100.0 * (c[lab] - c[ref]) / c[ref]
        for c in data.values()
        if lab in c and ref in c
    )
    n = len(d)
    if n < 4:
        print("not enough rounds")
        return
    sd = statistics.stdev(d)
    q1, q3 = d[n // 4], d[(3 * n) // 4]
    t = n // 5
    trimmed = statistics.fmean(d[t : n - t])
    tsd = statistics.stdev(d[t : n - t])
    print(f"{lab} vs {ref}  [{metric}]  n={n}")
    print("  sorted deltas: " + " ".join(f"{x:+.2f}" for x in d))
    print(f"  mean {statistics.fmean(d):+.3f}%   median {statistics.median(d):+.3f}%")
    print(f"  sd {sd:.3f}%   IQR [{q1:+.2f},{q3:+.2f}]   se(mean) {sd / n**0.5:.3f}%")
    print(f"  20%-trimmed mean {trimmed:+.3f}%   sd {tsd:.3f}%   se {tsd / (n - 2 * t) ** 0.5:.3f}%")
    for eff in (0.1, 0.2, 0.5):
        need = (2 * tsd / eff) ** 2
        print(f"  rounds for a {eff}% effect at 2 se (trimmed): {need:.0f}")


if __name__ == "__main__":
    main()
