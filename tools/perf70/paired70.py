#!/usr/bin/env python3
"""tools/perf70/paired70.py — the study-70 ordering metric, PAIRED PER POSITION.

The naive metric (sum of nodes over a position set) FAILS ITS OWN NULL: an
inert +/-1 jitter on quiet scores -- a perturbation that can only re-break
exact ties -- moves the 50-position d13 total by +18.3%, and the 300-position
d11 total by -4.7%. Search chaos is per-position and heavy-tailed, so a SUM is
dominated by whichever two positions happened to explode.

So aggregate the per-position PAIRED RATIO instead, with statistics that a
heavy tail cannot capture:

  * geometric mean of node_arm / node_base  (the mean of log-ratios; a 10x
    blow-up on one position moves it by log(10)/N, not by 10x/N)
  * median ratio
  * a paired SIGN TEST -- how many positions got cheaper, how many dearer.
    Discrete, immune to the tail entirely, and the statistic the study's own
    method rules ask for.

Usage:
  paired70.py BASEFILE ARMFILE          # two node-per-line files
  paired70.py --collect BIN DEPTH FENS  # print one node count per line
"""

import math
import statistics
import subprocess
import sys


def collect(binary, depth, fens, net, env=None):
    """Per-position node counts, in file order."""
    import os

    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(
        [binary, "bench", "16", "1", str(depth), fens, "depth", net],
        capture_output=True,
        text=True,
        env=e,
    )
    out = []
    for line in p.stderr.splitlines():
        if line.startswith("Nodes searched:"):
            out.append(int(line.split(":")[1]))
    if not out:
        raise SystemExit(f"paired70: no per-position counts from {binary}\n{p.stderr[-2000:]}")
    return out


def compare(base, arm, label=""):
    if len(base) != len(arm):
        raise SystemExit(f"paired70: length mismatch {len(base)} vs {len(arm)}")
    ratios = [a / b for a, b in zip(arm, base) if b > 0]
    logs = [math.log(r) for r in ratios]
    gm = math.exp(statistics.fmean(logs))
    med = statistics.median(ratios)
    better = sum(1 for r in ratios if r < 1.0)
    worse = sum(1 for r in ratios if r > 1.0)
    same = len(ratios) - better - worse
    # Standard error of the log-ratio mean -> a CI on the geometric mean.
    sd = statistics.pstdev(logs) if len(logs) > 1 else 0.0
    se = sd / math.sqrt(len(logs)) if logs else 0.0
    lo, hi = math.exp(-1.96 * se), math.exp(1.96 * se)
    tot = sum(arm) / sum(base)
    return {
        "label": label,
        "n": len(ratios),
        "geomean_pct": 100.0 * (gm - 1.0),
        "ci_lo_pct": 100.0 * (gm * lo - 1.0),
        "ci_hi_pct": 100.0 * (gm * hi - 1.0),
        "median_pct": 100.0 * (med - 1.0),
        "better": better,
        "worse": worse,
        "same": same,
        "total_pct": 100.0 * (tot - 1.0),
    }


def fmt(r):
    return (
        f"{r['label']:<30} n={r['n']:<5} "
        f"geo={r['geomean_pct']:+7.3f}% [{r['ci_lo_pct']:+6.3f},{r['ci_hi_pct']:+6.3f}] "
        f"med={r['median_pct']:+7.3f}% "
        f"better/worse={r['better']}/{r['worse']} (same {r['same']}) "
        f"sum={r['total_pct']:+7.3f}%"
    )


def main(argv):
    if argv[1] == "--collect":
        _, _, binary, depth, fens, net = argv[:6]
        for n in collect(binary, depth, fens, net):
            print(n)
        return
    base = [int(x) for x in open(argv[1])]
    arm = [int(x) for x in open(argv[2])]
    print(fmt(compare(base, arm, argv[3] if len(argv) > 3 else "arm")))


if __name__ == "__main__":
    main(sys.argv)
