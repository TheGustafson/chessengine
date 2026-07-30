#!/usr/bin/env python3
"""perf study 39 — read a panel39 raw file as PAIRED WITHIN-ROUND deltas.

Study 29 §5.3 retracted a four-round-old "win" because the rotated median was
measuring position-in-round: a byte-different null moved +1.5% in the same form
the arms moved +1.1%.  The delta therefore has to be taken against the control
*inside the same round*, and the sign count over rounds reported next to it —
the coordinator's "rank separation, not just spread" rule.  A median that does
not clear its null, or a sign count near 50/50, is not a result however pretty
the number is.

    python3 tools/perf39/med39.py <raw> <control-label> [label...]
"""

import collections
import statistics
import sys

COLS = ["nodes", "ins", "cyc", "loads", "miss", "stall"]


def main() -> None:
    raw, ctl = sys.argv[1], sys.argv[2]
    want = sys.argv[3:]
    rounds: dict[int, dict[str, dict[str, float]]] = collections.defaultdict(dict)
    for line in open(raw):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 8:
            continue
        rounds[int(f[0])][f[1]] = dict(zip(COLS, map(float, f[2:])))

    labels = want or sorted({lab for v in rounds.values() for lab in v})
    keep = [v for v in sorted(rounds.items()) if ctl in v[1] and all(a in v[1] for a in labels)]
    print(f"\n  panel39  control={ctl}  complete rounds={len(keep)}")
    nodesets = sorted({int(v[a]["nodes"]) for _, v in keep for a in v})
    print(f"  node counts seen: {nodesets}")

    base = {c: statistics.median(v[ctl][c] for _, v in keep) for c in COLS[1:]}
    print("  control medians: " + "  ".join(f"{c}/n={base[c]:.2f}" for c in COLS[1:]))
    print()
    hdr = f"  {'arm':10s} {'counter':7s} {'abs/n':>10s} {'Δ abs':>10s} {'Δ %':>9s} {'min Δ':>10s} {'sign':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lab in labels:
        for c in ("ins", "loads", "miss", "stall", "cyc"):
            d = [v[lab][c] - v[ctl][c] for _, v in keep]
            pct = [100.0 * x / v[ctl][c] for x, (_, v) in zip(d, keep)]
            med = statistics.median(d)
            # "min" = the least-interference round, signed toward the median.
            mn = min(d) if med < 0 else max(d)
            neg = sum(x < 0 for x in d)
            print(
                f"  {lab:10s} {c:7s} {statistics.median(v[lab][c] for _, v in keep):10.2f} "
                f"{med:+10.2f} {statistics.median(pct):+8.3f}% {mn:+10.2f} "
                f"{neg:3d}/{len(d):<4d}"
            )
        print()


if __name__ == "__main__":
    main()
