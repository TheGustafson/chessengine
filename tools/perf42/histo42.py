#!/usr/bin/env python3
"""perf study 42 -- per-bin address histogram inside one span.

    python3 tools/perf42/histo42.py <ibs-script> <lo-hex> <hi-hex> [--grain-kb K]

`regions42.py` merges adjacent occupied bins, which is right for finding the
mappings but wrong for finding the TABLE inside one of them: glibc puts the
transposition table and the NNUE weights in the same 200 MB+ heap mapping, so
the span analysis returns one blob.

The transposition table has a signature no other structure here has: its
addresses are UNIFORM. `mul_hi64` spreads a Zobrist key over the whole table,
so every bin inside it gets the same traffic to within counting noise, and the
plateau ends exactly at the allocation's edges. The NNUE weight tables are the
opposite -- accesses follow the position's active features, and the histogram
is spiky. Print the histogram and the table names itself.
"""

from __future__ import annotations

import argparse
import collections
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from regions42 import parse  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("lo")
    ap.add_argument("hi")
    ap.add_argument("--grain-kb", type=float, default=1024.0)
    ap.add_argument("--nodes", type=float, default=0.0)
    ap.add_argument("--period", type=float, default=50021.0)
    args = ap.parse_args()

    lo, hi = int(args.lo, 16), int(args.hi, 16)
    grain = int(args.grain_kb * 1024)
    bins: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for addr, op, lvl, _tlb, _ip in parse(args.script):
        if lo <= addr < hi:
            b = (addr - lo) // grain
            bins[b][lvl] += 1
            bins[b][op] += 1
            bins[b]["n"] += 1

    scale = args.period / args.nodes if args.nodes else 0.0
    nbins = (hi - lo + grain - 1) // grain
    print(f"{'off(MB)':>9} {'n':>7} {'L1':>7} {'L2':>7} {'L3':>7} {'DRAM':>7} {'ld':>7} {'st':>7}  bar")
    peak = max((bins[b]["n"] for b in bins), default=1)
    for b in range(nbins):
        c = bins[b]
        if not c["n"]:
            continue
        bar = "#" * int(40 * c["n"] / max(peak, 1))
        print(
            f"{b * grain / 1048576.0:9.2f} {c['n']:7d} {c['L1']:7d} {c['L2']:7d} "
            f"{c['L3']:7d} {c['DRAM']:7d} {c['LOAD']:7d} {c['STORE']:7d}  {bar}"
        )
    if scale:
        tot = sum(bins[b]["n"] for b in bins)
        l3 = sum(bins[b]["L3"] for b in bins)
        dram = sum(bins[b]["DRAM"] for b in bins)
        print(
            f"\nspan total: n={tot} ({tot * scale:.3f}/node)  "
            f"L3={l3} ({l3 * scale:.4f}/node)  DRAM={dram} ({dram * scale:.4f}/node)"
        )


if __name__ == "__main__":
    main()
