#!/usr/bin/env python3
"""perf study 39 — replicate study 29's `dstpin` statistic exactly, then show
what it is made of.

Study 29 §3 (the CORRECTION table) reports its subtractive probe as a change in
**stall cycles per instruction**, because the probe deletes ~6.5% of the
engine's instructions and it attributed that deletion to compiler CSE:

    base   stall/instr 0.4681
    dstpin stall/instr 0.4464   (-4.63%)   =>  -114.6 stall cycles/node
                                           =>   0.45 stall cycles per line

This recomputes that statistic on today's panel, so the comparison to study 31
is like for like and not a box or engine-version difference, and prints the RAW
per-node delta beside it — the raw and normalised numbers differ by ~2.5x, and
that difference is a modelling choice, not a measurement.

    python3 tools/perf39/replicate29.py <raw> <control> <arm> [lines-removed]
"""

import collections
import statistics
import sys

COLS = ["nodes", "ins", "cyc", "loads", "miss", "stall"]


def main() -> None:
    raw, ctl, arm = sys.argv[1:4]
    lines = float(sys.argv[4]) if len(sys.argv) > 4 else 253.7

    rounds: dict[int, dict[str, dict[str, float]]] = collections.defaultdict(dict)
    for line in open(raw):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 8:
            continue
        rounds[int(f[0])][f[1]] = dict(zip(COLS, map(float, f[2:])))
    keep = [v for _, v in sorted(rounds.items()) if ctl in v and arm in v]

    def med(lab: str, c: str) -> float:
        return statistics.median(v[lab][c] for v in keep)

    b_ins, b_st = med(ctl, "ins"), med(ctl, "stall")
    a_ins, a_st = med(arm, "ins"), med(arm, "stall")
    b_spi, a_spi = b_st / b_ins, a_st / a_ins
    d_spi = 100.0 * (a_spi - b_spi) / b_spi
    normalised = d_spi / 100.0 * b_st

    print(f"\n  replicate29  control={ctl}  arm={arm}  rounds={len(keep)}  "
          f"lines removed={lines}")
    print(f"    {ctl:8s}  i/n={b_ins:9.2f}  stall/n={b_st:9.2f}  "
          f"stall/instr={b_spi:.4f}")
    print(f"    {arm:8s}  i/n={a_ins:9.2f}  stall/n={a_st:9.2f}  "
          f"stall/instr={a_spi:.4f}")
    print(f"    Δ instructions          : {a_ins - b_ins:+9.2f}/node "
          f"({100.0 * (a_ins - b_ins) / b_ins:+.2f}%)")
    print(f"    Δ stall, RAW            : {a_st - b_st:+9.2f}/node "
          f"=> {abs(a_st - b_st) / lines:.3f} stall cycles per line")
    print(f"    Δ stall/instr           : {d_spi:+9.3f}%   "
          f"(study 29 measured −4.63% on this statistic)")
    print(f"    Δ stall, NORMALISED     : {normalised:+9.2f}/node "
          f"=> {abs(normalised) / lines:.3f} stall cycles per line")
    print(f"                              (study 29 published −114.6 => 0.45)\n")


if __name__ == "__main__":
    main()
