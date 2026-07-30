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
    """Return (per-node medians, per-run RATIO-to-cycles medians, run count).

    ## Why the second return value exists (coordinator note, 2026-07-30)

    `census44.sh` walks its 12 event groups in a FIXED order, so every group
    always runs in the same slot of a rep. This frame has a measured
    within-round drift (study 44 §4.2: +3.9% from a round's first slot to its
    last), which means a per-node LEVEL from group `g_ret` and a per-node LEVEL
    from group `g_slot` are not directly comparable -- each carries its own
    slot's bias, and dividing one by a GLOBAL median of `cycles:u` pooled over
    all groups imports a bias that belongs to a different slot entirely.

    The fix is to never compare across groups in absolute terms. Every group
    carries `cycles:u`, so each run yields its own self-contained ratio
    `event / cycles` measured in the SAME run, at the SAME slot, under the SAME
    instantaneous load. Those ratios are what the closure checks below use;
    any common multiplicative drift cancels exactly in the division.
    """
    vals = defaultdict(list)
    ratios = defaultdict(list)
    nrows = 0
    for line in open(path):
        if "SCALED" in line:
            raise SystemExit(f"refusing multiplexed row: {line.strip()}")
        toks = line.split()
        if len(toks) < 4:
            continue
        nrows += 1
        run = {}
        for t in toks[3:]:
            if "=" not in t:
                continue
            k, v = t.split("=", 1)
            if k == "nodes":
                continue
            run[k] = float(v)
            vals[k].append(float(v))
        cyc = run.get("cycles:u")
        if cyc:
            for k, v in run.items():
                ratios[k].append(v / cyc)
    return vals, ratios, nrows


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
    vals, ratios, nrows = load(path)
    med = {k: statistics.median(v) for k, v in vals.items()}
    # Fraction-of-cycles for every event, computed WITHIN each run and then
    # aggregated -- never level-over-pooled-level. See load()'s docstring.
    frac = {k: statistics.median(v) for k, v in ratios.items()}
    cyc_ref = med["cycles:u"]

    def as_cycles(ev):
        """That event's share of cycles, expressed back in cycles/node."""
        return frac[ev] * cyc_ref

    print(f"# {path}  ({nrows} runs)\n")
    print(f"{'event':<62}{'per node':>13}{'IQR%':>9}{'n':>5}")
    print("-" * 89)
    for k in sorted(vals, key=lambda k: -med[k]):
        print(f"{k:<62}{med[k]:>13.3f}{spread(vals[k]):>9.2f}{len(vals[k]):>5}")

    cyc = cyc_ref
    print(f"\n## closure 1 — the ex_no_retire cycle partition (denominator {cyc:.2f} c/node)")
    print("   shares computed WITHIN each run, then aggregated (see load() docstring)\n")
    parts = [
        ("load_not_complete  (memory)", as_cycles("ex_no_retire.load_not_complete:u")),
        ("not_complete - load (execution)",
         as_cycles("ex_no_retire.not_complete:u")
         - as_cycles("ex_no_retire.load_not_complete:u")),
        ("other", as_cycles("ex_no_retire.other:u")),
        ("empty  (front end, study 41)", as_cycles("ex_no_retire.empty:u")),
        ("thread_not_selected  (SMT)", as_cycles("ex_no_retire.thread_not_selected:u")),
    ]
    tot = sum(v for _, v in parts)
    for name, v in parts:
        print(f"  {name:<38}{v:>10.2f}{v / cyc * 100:>9.2f} %")
    print(f"  {'--- all no-retire cycles':<38}{tot:>10.2f}{tot / cyc * 100:>9.2f} %")
    print(f"  {'RETIRING cycles (cycles - above)':<38}{cyc - tot:>10.2f}{(cyc - tot) / cyc * 100:>9.2f} %")
    if "ex_no_retire.all:u" in frac:
        a = as_cycles("ex_no_retire.all:u")
        print(f"\n  ex_no_retire.all measured = {a:.2f}; parts sum = {tot:.2f}; "
              f"residual {(tot - a) / a * 100:+.3f} %")

    print("\n## closure 2 — dispatch slots (6/cycle)")
    print("   same within-run normalisation\n")
    disp = as_cycles("de_src_op_disp.op_cache:u") + as_cycles("de_src_op_disp.decoder:u")
    slots = [
        ("ops dispatched", disp),
        ("back-end bound", as_cycles("de_no_dispatch_per_slot.backend_stalls:u")),
        ("front-end bound", as_cycles("de_no_dispatch_per_slot.no_ops_from_frontend:u")),
        ("SMT contention", as_cycles("de_no_dispatch_per_slot.smt_contention:u")),
    ]
    stot = sum(v for _, v in slots)
    for name, v in slots:
        print(f"  {name:<38}{v:>10.2f}{v / stot * 100:>9.2f} %")
    print(f"  {'--- total slots':<38}{stot:>10.2f}")
    print(f"  implied cycles = {stot / 6:.2f}  vs measured {cyc:.2f}  "
          f"({(stot / 6 - cyc) / cyc * 100:+.3f} %)")


if __name__ == "__main__":
    main()
