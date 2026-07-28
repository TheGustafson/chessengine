#!/usr/bin/env python3
"""Parse smp9.sh output into an SMP scaling table.

Two statistics per (arm, threads), both relative to that arm's OWN 1-thread
figure so the two engines are compared on scaling, not on absolute speed:

  * NPS speedup       = median(nps at T) / median(nps at 1)   -- raw throughput
  * time-to-depth     = median(ms at 1)  / median(ms at T)    -- the EFFECTIVE
                        speedup; a shared TT makes threads search overlapping
                        trees, so nodes grow with T and NPS scaling flatters
                        the result.
"""

import statistics
import sys
from collections import defaultdict


def main(path: str) -> None:
    rows = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        if not line.startswith("round="):
            continue
        f = dict(kv.split("=", 1) for kv in line.split())
        rows[f["arm"]][int(f["threads"])].append(
            (int(f["nodes"]), int(f["nps"]), int(f["ms"]))
        )

    for arm, per_t in rows.items():
        ts = sorted(per_t)
        base_nps = statistics.median(r[1] for r in per_t[ts[0]])
        base_ms = statistics.median(r[2] for r in per_t[ts[0]])
        print(f"\n=== {arm} ===")
        print(
            f"{'threads':>7} {'nodes(med)':>12} {'NPS(med)':>11} {'ms(med)':>8} "
            f"{'NPS speedup':>12} {'eff (ttd)':>10} {'node infl':>10}"
        )
        base_nodes = statistics.median(r[0] for r in per_t[ts[0]])
        for t in ts:
            nodes = statistics.median(r[0] for r in per_t[t])
            nps = statistics.median(r[1] for r in per_t[t])
            ms = statistics.median(r[2] for r in per_t[t])
            print(
                f"{t:>7} {nodes:>12,.0f} {nps:>11,.0f} {ms:>8,.0f} "
                f"{nps / base_nps:>12.3f} {base_ms / ms:>10.3f} "
                f"{nodes / base_nodes:>10.3f}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
