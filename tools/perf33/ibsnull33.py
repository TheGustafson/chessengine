#!/usr/bin/env python3
"""perf study 33 -- THE NULL FOR THE IBS ATTRIBUTION.

    python3 tools/perf33/ibsnull33.py <ibs.script> <ibs.mmap> <ibs.data> <binary>

`ibsmiss33.py` and `mlp33.py` both resolve an IBS sample's IP to a symbol with
a bisect over `nm -S` ranges, because `perf script -F sym` returns [unknown]
for IBS records on this perf build.  A home-made symbol resolver is exactly the
kind of instrument that fails silently, so it gets checked against the one
resolver that is known to work: `perf report`, on the same file.

The check: for every symbol, my share of ALL IBS samples vs perf report's.  If
they agree the bisect is sound and any disagreement in the LOAD-only view is a
property of the data, not of my parser.  If they do not agree, nothing built on
the bisect may be quoted.
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


def main() -> int:
    script, mmapf, data, binary = sys.argv[1:5]
    out = subprocess.run(["nm", "-S", binary], capture_output=True, text=True).stdout
    syms = sorted((int(f[0], 16), int(f[1], 16), f[3])
                  for f in (l.split() for l in out.splitlines())
                  if len(f) == 4 and f[2].lower() in "tw")
    starts = [s[0] for s in syms]
    ends = [(s[0] + s[1], s[2]) for s in syms]

    to_elf, vbase, foff, delta = ip_to_elf(mmapf, binary)

    mine: collections.Counter[str] = collections.Counter()
    mine_load: collections.Counter[str] = collections.Counter()
    tot = unmapped = 0
    for line in open(script):
        h, t = HEAD.match(line), TAIL.search(line)
        if not h or not t:
            continue
        tot += 1
        fa = to_elf(int(t.group(1), 16))
        i = bisect.bisect_right(starts, fa) - 1
        if i < 0 or fa >= ends[i][0]:
            unmapped += 1
            continue
        mine[ends[i][1]] += 1
        if dict(FIELD.findall(line)).get("OP") == "LOAD":
            mine_load[ends[i][1]] += 1

    rep = subprocess.run(
        ["perf", "report", "-i", data, "--stdio", "-q", "--no-children",
         "--percent-limit", "0.3", "--no-demangle"],
        capture_output=True, text=True).stdout
    theirs = {}
    for line in rep.splitlines():
        mm = re.match(r"\s*([0-9.]+)%\s+\S+\s+\S+\s+\[\.\]\s+(\S+)", line)
        if mm:
            theirs[mm.group(2)] = float(mm.group(1))

    print(f"  {tot:,} parsed samples, {unmapped:,} outside every nm range "
          f"({100.0 * unmapped / max(tot, 1):.1f}% -- libc/ld, expected)")
    print(f"\n  {'perf':>7s} {'mine':>7s} {'delta':>7s} {'%LOAD':>7s}  symbol")
    worst = 0.0
    for name, p in sorted(theirs.items(), key=lambda kv: -kv[1])[:14]:
        mypct = 100.0 * mine.get(name, 0) / max(tot, 1)
        loadpct = 100.0 * mine_load.get(name, 0) / max(mine.get(name, 1), 1)
        worst = max(worst, abs(mypct - p))
        print(f"  {p:6.2f}% {mypct:6.2f}% {mypct - p:+6.2f}% {loadpct:6.1f}%  {name[-52:]}")
    print(f"\n  worst absolute disagreement: {worst:.2f} percentage points")
    print("  The %LOAD column is the fraction of each symbol's tagged ops that")
    print("  IBS reports as loads -- it is what makes the load-share table")
    print("  differ from the sample-share table, and it is DATA, not a parser bug,")
    print("  if and only if the perf/mine columns agree above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
