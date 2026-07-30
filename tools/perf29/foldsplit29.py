#!/usr/bin/env python3
"""perf study 29 -- split a kernel's samples into ACCUMULATOR STREAM vs WEIGHT ROW.

Reads `perf annotate --stdio -l --percent-type global-period` output for one
symbol and buckets every memory-touching instruction by the provenance of its
base register, which the caller supplies as a register->stream map (read off
the prologue -- see the study doc, which prints the prologue for each kernel).

Usage:
  perf annotate ... | python3 foldsplit29.py acc=rsi,rdi weight=rdx,rcx,rax,r8
"""

import re
import sys
from collections import Counter

# perf annotate --stdio lines look like:
#     0.42 :   76697:  vpaddw (%rsi,%r9,1),%zmm6,%zmm6
LINE = re.compile(r"^\s*([\d.]+)\s*:\s*([0-9a-f]+):\s+([a-z0-9.]+)\s*(.*)$")
MEM = re.compile(r"(-?0x[0-9a-f]+)?\(%(\w+)(?:,%(\w+)(?:,(\d))?)?\)")
SKIP = ("lea", "nop", "data16", "cs", "xchg", "jmp", "j")


def split_ops(ops):
    out, depth, cur = [], 0, ""
    for ch in ops:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def main():
    groups = {}
    for a in sys.argv[1:]:
        name, regs = a.split("=", 1)
        for r in regs.split(","):
            groups[r] = name
    tot = Counter()
    unclassified = Counter()
    grand = 0.0
    for line in sys.stdin:
        m = LINE.match(line.rstrip("\n"))
        if not m:
            continue
        pct, _addr, mnem, ops = m.group(1), m.group(2), m.group(3), m.group(4)
        pct = float(pct)
        grand += pct
        if any(mnem.startswith(p) for p in SKIP):
            tot["_nonmem"] += pct
            continue
        ops = ops.split("#")[0].strip()
        mm = MEM.search(ops) if ops else None
        if not mm:
            tot["_nonmem"] += pct
            continue
        parts = split_ops(ops)
        memlast = MEM.search(parts[-1]) is not None
        memearlier = any(MEM.search(p) for p in parts[:-1])
        base = mm.group(2)
        if memlast and not memearlier and mnem.startswith(("mov", "vmov")):
            tot[f"store:{groups.get(base, base)}"] += pct
            continue
        g = groups.get(base)
        if g is None:
            unclassified[f"{mnem} %{base}"] += pct
            tot[f"load:?{base}"] += pct
        else:
            tot[f"load:{g}"] += pct
    print(f"# total sampled weight in symbol: {grand:.3f}")
    for k, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {v:10.3f}  ({100 * v / max(grand, 1e-9):5.2f}%)  {k}")
    if unclassified:
        print("# unclassified base registers")
        for k, v in sorted(unclassified.items(), key=lambda x: -x[1]):
            print(f"  {v:10.3f}  {k}")


if __name__ == "__main__":
    main()
