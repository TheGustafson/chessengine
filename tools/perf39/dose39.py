#!/usr/bin/env python3
"""perf study 39 — the DOSE REGRESSION that decides the additive/subtractive gap.

Study 29 §4.1 proposed that its 0.45 stall cycles per accumulator line and
study 31's 2.36 are both correct in their own regime, because "marginal cost at
a capacity cliff exceeds average value below it".  That reconciliation makes a
sharp, falsifiable prediction: **the cost function must be CONVEX at our
operating point** — each additional line of accumulator traffic must cost more
than the one before.  A LINEAR dose response falsifies it outright, and a
CONCAVE (saturating) one falsifies it with the sign reversed.

So this fits stall/node against lines/node across doses 0, 1, 2 and 4 of the
`probe-addcopy` copy, taking every delta PAIRED INSIDE A ROUND (study 29 §5.3
retracted a result that was measuring position-in-round), and reports both the
straight-line slope and the per-dose marginal slopes that show the curvature.

    python3 tools/perf39/dose39.py <raw> [evals-per-node]
"""

import collections
import statistics
import sys

COLS = ["nodes", "ins", "cyc", "loads", "miss", "stall"]
# 160 64-byte lines moved per copy: 80 read + 80 written, 2 x ACC x 2 B = 5120 B
# each way at ACC = 1280.
LINES_PER_COPY = 160.0
DOSE_OF = {"d0": 0, "d1": 1, "d2": 2, "d4": 4}


def main() -> None:
    raw = sys.argv[1]
    evals = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4311
    per_dose = LINES_PER_COPY * evals

    rounds: dict[int, dict[str, dict[str, float]]] = collections.defaultdict(dict)
    for line in open(raw):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 8:
            continue
        rounds[int(f[0])][f[1]] = dict(zip(COLS, map(float, f[2:])))

    arms = list(DOSE_OF)
    keep = [v for _, v in sorted(rounds.items()) if all(a in v for a in arms)]
    print(f"\n  DOSE REGRESSION   complete rounds = {len(keep)}")
    print(f"  evals/node = {evals} (foldcount, tools/perf39/census39.sh)")
    print(f"  one dose   = {per_dose:.2f} 64-byte lines/node "
          f"({LINES_PER_COPY:.0f} lines x {evals} evals)\n")

    # The QUIETEST round: the one whose control stalled least. The coordinator's
    # recipe asks for median AND min, min being the least-interference sample.
    quiet = min(range(len(keep)), key=lambda i: keep[i]["d0"]["stall"])

    print(f"  {'arm':6s} {'dose':>4s} {'lines/n':>9s} "
          f"{'Δstall/n':>10s} {'Δmiss/n':>9s} {'Δins/n':>9s} {'Δcyc/n':>9s} "
          f"{'st/line':>8s} {'quiet':>8s} {'sign':>8s}")
    print("  " + "-" * 90)
    pts, qpts = [], []
    for a in arms:
        dose = DOSE_OF[a]
        lines = dose * per_dose
        ds = [v[a]["stall"] - v["d0"]["stall"] for v in keep]
        dm = [v[a]["miss"] - v["d0"]["miss"] for v in keep]
        di = [v[a]["ins"] - v["d0"]["ins"] for v in keep]
        dc = [v[a]["cyc"] - v["d0"]["cyc"] for v in keep]
        med = statistics.median(ds)
        pos = sum(x > 0 for x in ds)
        spl = med / lines if lines else float("nan")
        qsl = ds[quiet] / lines if lines else float("nan")
        pts.append((lines, med))
        qpts.append((lines, ds[quiet]))
        print(f"  {a:6s} {dose:4d} {lines:9.2f} {med:+10.2f} "
              f"{statistics.median(dm):+9.2f} {statistics.median(di):+9.2f} "
              f"{statistics.median(dc):+9.2f} {spl:8.3f} {qsl:8.3f} "
              f"{pos:3d}/{len(ds):<4d}")

    # Straight line through the origin, least squares.
    def slope(ps: list[tuple[float, float]]) -> float:
        return sum(x * y for x, y in ps) / sum(x * x for x, _ in ps)

    print(f"\n  least-squares slope through the origin:")
    print(f"    medians       : {slope(pts):.3f} stall cycles per 64-byte line")
    print(f"    quietest round: {slope(qpts):.3f} stall cycles per 64-byte line")

    print("\n  MARGINAL slope between consecutive doses "
          "(this is where a capacity cliff would show):")
    order = sorted(pts)
    for (x0, y0), (x1, y1) in zip(order, order[1:]):
        print(f"    {x0:7.1f} -> {x1:7.1f} lines/n : "
              f"{(y1 - y0) / (x1 - x0):7.3f} stall cycles/line")
    print("\n  CONVEX (rising marginals) is what study 29 §4.1 needs to be true.")


if __name__ == "__main__":
    main()
