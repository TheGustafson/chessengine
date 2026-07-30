#!/usr/bin/env python3
"""perf study 33 -- stall cycles PER LOAD and PER MISS, by symbol.

    python3 tools/perf33/mlp33.py <stall.data> <ibs.script> <ibs.mmap> <binary>

THE QUESTION THIS ANSWERS.  `search::<NonPv>` carries 8.80% of the engine's
load-stall cycles but only 1.08% of its non-L1 loads, and 90.5% of the loads it
does issue are L1 hits.  Those two facts are only compatible if the stall is
not about cache LEVEL at all -- if the node's loads are exposed at whatever
latency they have because there is nothing to overlap them with.

The statistic that says so is a ratio of shares, deliberately:

    stall index  = (share of engine stall cycles) / (share of retired load ops)
    miss index   = (share of engine stall cycles) / (share of non-L1 loads)

Shares, not absolutes, because the two profiles have different units and
because `L1-dcache-loads` DOUBLE-COUNTS a 512-bit load on Zen 4 (2.0015 events
per `vmovdqu64`, measured in study 29).  Anything that mixed a perf-stat load
total into a comparison between a scalar symbol and an AVX-512 kernel would be
wrong by ~2x in exactly the direction that flatters this study's conclusion.
IBS counts retired OPS, so both shares here are on the same footing.

A stall index near 1.0 means a symbol waits its fair share.  Well above 1.0
with a LOW miss share means serial, dependent loads: latency that nothing else
is covering.  Well below 1.0 with a HIGH miss share means the opposite -- a
streaming kernel with many misses in flight at once.
"""

from __future__ import annotations

import bisect
import collections
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from elfmap33 import ip_to_elf  # noqa: E402

HEAD = re.compile(r"^\s*([0-9a-f]+)\s+[0-9a-f]+\s+\|")
FIELD = re.compile(r"\|(OP|LVL)\s+([^|]*?)\s*(?=\|)")
TAIL = re.compile(r"\|BLK\s+\S+\s+([0-9a-f]+)\s*$")
MMAP = re.compile(r"\[0x([0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0)[^]]*\]: r-xp")


def symbols(binary: str):
    out = subprocess.run(["nm", "-S", binary], capture_output=True, text=True).stdout
    syms = []
    for line in out.splitlines():
        f = line.split()
        if len(f) == 4 and f[2].lower() in "tw":
            syms.append((int(f[0], 16), int(f[1], 16), f[3]))
    syms.sort()
    return [s[0] for s in syms], [(s[0] + s[1], s[2]) for s in syms]


def main() -> int:
    stall_data, ibs_script, ibs_mmap, binary = sys.argv[1:5]

    stall = subprocess.run(
        ["perf", "script", "-i", stall_data, "-F", "ip,sym", "--no-demangle"],
        capture_output=True, text=True).stdout
    stall_n: collections.Counter[str] = collections.Counter()
    for line in stall.splitlines():
        f = line.split()
        if len(f) >= 2:
            stall_n[f[1]] += 1
    stall_tot = sum(stall_n.values())

    to_elf, vbase, foff, delta = ip_to_elf(ibs_mmap, binary)
    starts, ends = symbols(binary)
    loads: collections.Counter[str] = collections.Counter()
    miss: collections.Counter[str] = collections.Counter()
    for line in open(ibs_script):
        h, t = HEAD.match(line), TAIL.search(line)
        if not h or not t:
            continue
        f = dict(FIELD.findall(line))
        if f.get("OP") != "LOAD" or f.get("LVL") in (None, "N/A"):
            continue
        fa = to_elf(int(t.group(1), 16))
        i = bisect.bisect_right(starts, fa) - 1
        if i < 0 or fa >= ends[i][0]:
            continue
        loads[ends[i][1]] += 1
        if f["LVL"] != "L1 hit":
            miss[ends[i][1]] += 1
    load_tot, miss_tot = sum(loads.values()), sum(miss.values())

    names = [n for n, _ in stall_n.most_common(16)]
    dm = dict(zip(names, subprocess.run(
        ["c++filt", "--no-strip-underscore"], input="\n".join(names),
        capture_output=True, text=True).stdout.splitlines() or names))

    print(f"  stall samples {stall_tot:,}   IBS load samples {load_tot:,}   "
          f"non-L1 {miss_tot:,}")
    print(f"\n  {'%stall':>7s} {'%loads':>7s} {'%miss':>7s} {'stall/load':>11s} "
          f"{'stall/miss':>11s}  symbol")
    for n in names:
        s = 100.0 * stall_n[n] / stall_tot
        l = 100.0 * loads.get(n, 0) / max(load_tot, 1)
        ms = 100.0 * miss.get(n, 0) / max(miss_tot, 1)
        si = s / l if l else float("nan")
        mi = s / ms if ms else float("nan")
        pretty = dm.get(n, n)
        pretty = re.sub(r"\[[0-9a-f]+\]", "", pretty).replace("chessengine::", "")
        print(f"  {s:6.2f}% {l:6.2f}% {ms:6.2f}% {si:11.2f} {mi:11.2f}  {pretty[:58]}")
    print("\n  stall/load and stall/miss are INDICES: a symbol's share of stall")
    print("  cycles divided by its share of load ops / of non-L1 loads. 1.00 =")
    print("  the engine average. High stall/load + low miss share = serial,")
    print("  dependent loads. Low stall/load + high miss share = overlapped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
