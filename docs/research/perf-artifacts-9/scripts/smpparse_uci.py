#!/usr/bin/env python3
"""Parse smpuci9.py output into an SMP scaling table (per position + pooled)."""

import statistics
import sys
from collections import defaultdict


def main(path):
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in open(path):
        if not line.startswith("round=") or "FAILED" in line:
            continue
        f = dict(kv.split("=", 1) for kv in line.split())
        rows[f["arm"]][f["pos"]][int(f["threads"])].append(
            (int(f["nodes"]), int(f["nps"]), int(f["ms"]))
        )

    for arm, per_pos in rows.items():
        for pos, per_t in per_pos.items():
            ts = sorted(per_t)
            med = lambda t, i: statistics.median(r[i] for r in per_t[t])
            b_nps, b_ms, b_nodes = med(ts[0], 1), med(ts[0], 2), med(ts[0], 0)
            print(f"\n=== {arm} / {pos} ===")
            print(
                f"{'T':>3} {'nodes(med)':>12} {'NPS(med)':>12} {'ms(med)':>8} "
                f"{'NPS speedup':>12} {'ttd speedup':>12} {'node infl':>10}"
            )
            for t in ts:
                print(
                    f"{t:>3} {med(t, 0):>12,.0f} {med(t, 1):>12,.0f} {med(t, 2):>8,.0f} "
                    f"{med(t, 1) / b_nps:>12.3f} {b_ms / med(t, 2):>12.3f} "
                    f"{med(t, 0) / b_nodes:>10.3f}"
                )


if __name__ == "__main__":
    main(sys.argv[1])
