#!/usr/bin/env python3
"""perf study 40 — regress an amplified arm back to one call.

    python3 tools/perf40/doseslope40.py <panel.txt> <base-bin> <arm-bin> [--metric M]

The dose amplifier (`--features dosesort40`, `src/dose40.rs`) makes every
`partial_insertion_sort` call replay `D` real earlier inputs through the same
sort, so a call site that runs once per call runs `1 + D` times. Both binaries
carry the SAME amplifier and differ only in the sort algorithm, so at dose `D`
the arm's per-node effect is `(1 + D)` times its effect in the shipped engine.

For each dose this prints the paired within-round delta in ABSOLUTE per-node
units, then fits `delta = slope * (1 + D)` by least squares through the origin.

**The slope IS the shipped effect** — the arm's per-node delta at `D = 0` —
measured at up to 25x its size. The dose-0 row is the same quantity measured
directly, and the two agreeing is the amplifier's fidelity check; a residual
intercept would say the amplifier is not linear in the number of sorts and the
extrapolation is not allowed.
"""

import re
import statistics
import sys
from collections import defaultdict

ROW = re.compile(r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+(.*)$")


def load(path: str) -> dict[int, dict[str, dict[str, float]]]:
    data: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        row = {"nodes": float(m.group(3))}
        for tok in m.group(4).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        data[int(m.group(1))][m.group(2)] = row
    return data


def main() -> None:
    path, base, arm = sys.argv[1], sys.argv[2], sys.argv[3]
    args = sys.argv
    metrics = (
        [args[args.index("--metric") + 1]]
        if "--metric" in args
        else ["cycles:u", "instructions:u", "branches:u", "ex_ret_brn_misp:u"]
    )
    data = load(path)

    doses = sorted(
        {
            int(lab.split(":")[1])
            for row in data.values()
            for lab in row
            if lab.startswith(base + ":")
        }
    )

    for k in metrics:
        print(f"=== {k} ===")
        print(
            f"{'dose':>5} {'1+D':>4} {'n':>4} {'base/node':>11} {'arm-base':>10} "
            f"{'rel':>8} {'neg/n':>9} {'per sort exec':>14}"
        )
        pts = []
        for d in doses:
            bl, al = f"{base}:{d}", f"{arm}:{d}"
            deltas, bases = [], []
            for _, cols in sorted(data.items()):
                if bl in cols and al in cols and k in cols[bl] and k in cols[al]:
                    deltas.append(cols[al][k] - cols[bl][k])
                    bases.append(cols[bl][k])
            if not deltas:
                continue
            med = statistics.median(deltas)
            bmed = statistics.median(bases)
            neg = sum(1 for x in deltas if x < 0)
            pts.append((1 + d, med))
            print(
                f"{d:>5} {1 + d:>4} {len(deltas):>4} {bmed:>11.2f} {med:>+10.3f} "
                f"{100 * med / bmed:>+7.3f}% {neg:>4}/{len(deltas):<4} "
                f"{med / (1 + d):>+14.3f}"
            )
        if len(pts) >= 2:
            num = sum(x * y for x, y in pts)
            den = sum(x * x for x, _ in pts)
            slope = num / den
            print(f"  least-squares slope through origin: {slope:+.3f} per node")
            print(f"  residuals: " + "  ".join(f"D={x - 1}:{y - slope * x:+.2f}" for x, y in pts))
        print()


if __name__ == "__main__":
    main()
