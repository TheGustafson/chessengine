#!/usr/bin/env python3
"""perf study 33 -- turn a sampled profile into PER-NODE event counts.

    python3 tools/perf33/pernode33.py <perf.data> <bench-stdout> <period> [more...]

A per-symbol SHARE is not comparable across builds, because the denominator
moves: `search::<NonPv>` is 18.4% of study 28's PGO load-stall profile and
8.94% of the same profile on the non-PGO probe build, and almost all of that
difference is the non-PGO build stalling more everywhere else.  The quantity
that IS comparable -- and the one study 28 actually quoted -- is stall cycles
PER NODE at the symbol.

With `perf record -c <period>` each sample stands for exactly `period` events,
so per-node cost at a symbol = samples x period / nodes.  No separate `perf
stat` run is needed and no second execution can drift away from the first.

Pass several (perf.data, bench, period) triples to get them side by side.
"""

from __future__ import annotations

import re
import subprocess
import sys


def nodes_of(bench_path: str) -> int:
    txt = open(bench_path).read()
    m = re.findall(r"Nodes searched\s*:\s*([0-9]+)", txt)
    if not m:
        raise SystemExit(f"no node count in {bench_path}")
    return int(m[-1])


def shares(data: str) -> tuple[dict[str, int], int]:
    """symbol -> sample count, and the total.

    Counted from `perf script`, not `perf report`: the report header does not
    carry `nr_samples` on this perf build, and reading a percentage back out of
    a formatted table loses precision on exactly the small symbols this study
    cares about.
    """
    out = subprocess.run(
        ["perf", "script", "-i", data, "-F", "ip,sym", "--no-demangle"],
        capture_output=True, text=True).stdout
    d: dict[str, int] = {}
    n = 0
    for line in out.splitlines():
        f = line.split()
        if len(f) < 2:
            continue
        n += 1
        d[f[1]] = d.get(f[1], 0) + 1
    return d, n


def main() -> int:
    args = sys.argv[1:]
    cols = []
    while args:
        data, bench, period, *args = args
        sh, nsamp = shares(data)
        nodes = nodes_of(bench)
        cols.append((data.split("/")[-1].replace(".data", ""), sh,
                     int(period), nodes, nsamp))

    keys, seen = [], set()
    for _, sh, _, _, _ in cols:
        for k in sh:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    print(f"  {'symbol':52s}" + "".join(f"{c[0]:>16s}" for c in cols))
    print(f"  {'':52s}" + "".join(f"{'ev/node':>16s}" for c in cols))
    tot = [0.0] * len(cols)
    rows = []
    for k in keys:
        vals = []
        for _, sh, period, nodes, nsamp in cols:
            vals.append(sh.get(k, 0) * period / nodes)
        rows.append((max(vals), k, vals))
    for i, (_, sh, period, nodes, nsamp) in enumerate(cols):
        tot[i] = nsamp * period / nodes
    names = [k for _, k, _ in rows]
    dm = dict(zip(names, subprocess.run(
        ["c++filt", "--no-strip-underscore"], input="\n".join(names),
        capture_output=True, text=True).stdout.splitlines() or names))
    for _, k, vals in sorted(rows, reverse=True)[:20]:
        pretty = dm.get(k, k).replace("chessengine::", "")
        print(f"  {pretty[:52]:52s}" + "".join(f"{v:16.1f}" for v in vals))
    print(f"  {'TOTAL (all samples)':52s}" + "".join(f"{t:16.1f}" for t in tot))
    for c in cols:
        print(f"    {c[0]}: {c[4]:,} samples x period {c[2]} / {c[3]:,} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
