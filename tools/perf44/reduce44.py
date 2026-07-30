#!/usr/bin/env python3
"""perf study 44 — reduce a census44.sh dump to per-node medians.

    python3 tools/perf44/reduce44.py docs/research/perf-artifacts-44/census.txt [label]

Prints median, relative spread (IQR/median) and n for every event, then runs
the two closure checks this study is graded on:

  1. the ex_no_retire.* cycle partition (must tile `cycles:u`)
  2. the dispatch-slot accounting (must tile 6 x `cycles:u`)

Refuses any row perf marked SCALED (multiplexed): Zen 4 has 6 PMCs and every
group is <= 6 events, so a scaled row means something went wrong.
"""

import statistics
import sys
from collections import defaultdict


def load(path):
    vals = defaultdict(list)
    nrows = 0
    for line in open(path):
        if "SCALED" in line:
            raise SystemExit(f"refusing multiplexed row: {line.strip()}")
        toks = line.split()
        if len(toks) < 4:
            continue
        nrows += 1
        for t in toks[3:]:
            if "=" not in t:
                continue
            k, v = t.split("=", 1)
            if k == "nodes":
                continue
            vals[k].append(float(v))
    return vals, nrows


def spread(xs):
    if len(xs) < 4:
        return float("nan")
    med = statistics.median(xs)
    if med == 0:                       # `loop_buffer` is exactly 0 (study 41)
        return 0.0
    q = statistics.quantiles(xs, n=4)
    return (q[2] - q[0]) / med * 100


def main():
    path = sys.argv[1]
    vals, nrows = load(path)
    med = {k: statistics.median(v) for k, v in vals.items()}

    print(f"# {path}  ({nrows} runs)\n")
    print(f"{'event':<62}{'per node':>13}{'IQR%':>9}{'n':>5}")
    print("-" * 89)
    for k in sorted(vals, key=lambda k: -med[k]):
        print(f"{k:<62}{med[k]:>13.3f}{spread(vals[k]):>9.2f}{len(vals[k]):>5}")

    cyc = med["cycles:u"]
    print(f"\n## closure 1 — the ex_no_retire cycle partition (denominator {cyc:.2f} c/node)\n")
    parts = [
        ("load_not_complete  (memory)", med["ex_no_retire.load_not_complete:u"]),
        ("not_complete - load (execution)",
         med["ex_no_retire.not_complete:u"] - med["ex_no_retire.load_not_complete:u"]),
        ("other", med["ex_no_retire.other:u"]),
        ("empty  (front end, study 41)", med["ex_no_retire.empty:u"]),
        ("thread_not_selected  (SMT)", med["ex_no_retire.thread_not_selected:u"]),
    ]
    tot = sum(v for _, v in parts)
    for name, v in parts:
        print(f"  {name:<38}{v:>10.2f}{v / cyc * 100:>9.2f} %")
    print(f"  {'--- all no-retire cycles':<38}{tot:>10.2f}{tot / cyc * 100:>9.2f} %")
    print(f"  {'RETIRING cycles (cycles - above)':<38}{cyc - tot:>10.2f}{(cyc - tot) / cyc * 100:>9.2f} %")
    if "ex_no_retire.all:u" in med:
        a = med["ex_no_retire.all:u"]
        print(f"\n  ex_no_retire.all measured = {a:.2f}; parts sum = {tot:.2f}; "
              f"residual {(tot - a) / a * 100:+.3f} %")

    print("\n## closure 2 — dispatch slots (6/cycle)\n")
    disp = med["de_src_op_disp.op_cache:u"] + med["de_src_op_disp.decoder:u"]
    slots = [
        ("ops dispatched", disp),
        ("back-end bound", med["de_no_dispatch_per_slot.backend_stalls:u"]),
        ("front-end bound", med["de_no_dispatch_per_slot.no_ops_from_frontend:u"]),
        ("SMT contention", med["de_no_dispatch_per_slot.smt_contention:u"]),
    ]
    stot = sum(v for _, v in slots)
    for name, v in slots:
        print(f"  {name:<38}{v:>10.2f}{v / stot * 100:>9.2f} %")
    print(f"  {'--- total slots':<38}{stot:>10.2f}")
    print(f"  implied cycles = {stot / 6:.2f}  vs measured {cyc:.2f}  "
          f"({(stot / 6 - cyc) / cyc * 100:+.3f} %)")


if __name__ == "__main__":
    main()
