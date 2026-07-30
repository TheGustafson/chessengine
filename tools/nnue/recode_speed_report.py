#!/usr/bin/env python3
"""Report for recode_speed_ab.sh runs: frame validity, then rank separation.

Reads <outdir>/{A,B}.rN.{perf,nodes}; prints node identity, instruction
parity, per-round paired cycle deltas, rank separation (rounds where B <
A), medians and minima. Median and min per the commissioning frame (min =
least interference)."""

import sys
from pathlib import Path


def read_perf(p: Path):
    cyc = ins = None
    for line in p.read_text().splitlines():
        parts = line.split(",")
        if len(parts) > 3 and parts[2] == "cycles:u":
            cyc = int(parts[0])
        if len(parts) > 3 and parts[2] == "instructions:u":
            ins = int(parts[0])
    assert cyc and ins, p
    return cyc, ins


def main():
    out = Path(sys.argv[1])
    rounds = sorted(
        int(p.stem.split(".r")[1]) for p in out.glob("A.r*.perf")
    )
    rows = []
    nodes = set()
    for r in rounds:
        ca, ia = read_perf(out / f"A.r{r}.perf")
        cb, ib = read_perf(out / f"B.r{r}.perf")
        na = (out / f"A.r{r}.nodes").read_text().strip()
        nb = (out / f"B.r{r}.nodes").read_text().strip()
        nodes.update((na, nb))
        rows.append((r, ca, cb, ia, ib))

    print(f"rounds: {len(rows)}   node totals seen: {sorted(nodes)}")
    if len(nodes) != 1:
        print("FRAME INVALID: node totals differ — arms searched different trees")
        sys.exit(1)
    print("node identity: OK (single total across all runs and both arms)")

    ia_med = sorted(x[3] for x in rows)[len(rows) // 2]
    ib_med = sorted(x[4] for x in rows)[len(rows) // 2]
    dins = 100.0 * (ib_med - ia_med) / ia_med
    print(f"instructions median: A {ia_med:,}  B {ib_med:,}  Δ {dins:+.4f}%")

    print(f"{'r':>3} {'cycles A':>15} {'cycles B':>15} {'Δ%(B vs A)':>10}")
    wins = 0
    deltas = []
    for r, ca, cb, ia, ib in rows:
        d = 100.0 * (cb - ca) / ca
        deltas.append(d)
        wins += cb < ca
        print(f"{r:>3} {ca:>15,} {cb:>15,} {d:>+10.2f}")

    deltas.sort()
    ca_all = sorted(x[1] for x in rows)
    cb_all = sorted(x[2] for x in rows)
    n = len(rows)
    # Full pairwise comparison (the commissioning frame's rank-separation
    # statistic): of all n*n (A_i, B_j) pairs, how many favour B?
    pair_wins = sum(1 for a in ca_all for b in cb_all if b < a)
    print(f"\npaired rounds where B faster: {wins}/{n}")
    print(f"pairwise (all {n}x{n}): B faster in {pair_wins}/{n * n}")
    print(f"median paired Δ: {deltas[n // 2]:+.2f}%   "
          f"best/worst: {deltas[0]:+.2f}% / {deltas[-1]:+.2f}%")
    print(f"min cycles:  A {ca_all[0]:,}   B {cb_all[0]:,}   "
          f"Δ {100.0 * (cb_all[0] - ca_all[0]) / ca_all[0]:+.2f}%")
    print(f"median cyc:  A {ca_all[n // 2]:,}   B {cb_all[n // 2]:,}   "
          f"Δ {100.0 * (cb_all[n // 2] - ca_all[n // 2]) / ca_all[n // 2]:+.2f}%")


if __name__ == "__main__":
    main()
