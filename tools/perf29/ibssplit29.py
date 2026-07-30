#!/usr/bin/env python3
"""perf study 29 -- split IBS load samples by SYMBOL x DATA-ADDRESS REGION x LEVEL.

IBS tags a retired op with its data address and the level that served it. That
is enough to answer the question a symbol share cannot: inside `at_full` or
`i16_deltas_into`, is the traffic the ACCUMULATOR STACK or a WEIGHT TABLE, and
which of the two actually misses.

The two regions are told apart by the mapping they live in: `/proc/pid/maps`
is gone by the time we parse, so the script clusters addresses into contiguous
regions (gap > 64 MB starts a new region) and prints the clusters with their
spans, which makes the ~100 MB weight tables and the ~1 MB accumulator stacks
self-evident. Pass `--regions a:b,...` to pin the classification.

Usage:
  python3 ibssplit29.py <script-file> <text-base-hex> <file-off-hex> <disasm> [--regions ...]
"""

import re
import sys
from collections import Counter, defaultdict

# perf script's column order is FIXED: data address, data_src, ip.
SAMPLE = re.compile(r"^\s*([0-9a-f]+)\s+[0-9a-f]+\s+(\|.*\|[^|]*)\s+([0-9a-f]+)\s*$")
LVL = re.compile(r"\|LVL ([^|]*?)\s*\|")
OP = re.compile(r"\|OP ([^|]*?)\s*\|")
SYM = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN = re.compile(r"^\s+([0-9a-f]+):\s+([a-z0-9.]+)\s*(.*)$")


def load_disasm(path):
    """addr -> (symbol, mnemonic, operands)"""
    out = {}
    cur = None
    with open(path) as f:
        for line in f:
            s = SYM.match(line.rstrip("\n"))
            if s:
                cur = s.group(2)
                continue
            i = INSN.match(line.rstrip("\n"))
            if i and cur:
                out[int(i.group(1), 16)] = (cur, i.group(2), i.group(3).split("#")[0].strip())
    return out


def main():
    script, base_hex, off_hex, disasm_path = sys.argv[1:5]
    base = int(base_hex, 16)
    foff = int(off_hex, 16)
    regions = []
    for a in sys.argv[5:]:
        if a.startswith("--regions="):
            for spec in a.split("=", 1)[1].split(","):
                name, lo, hi = spec.split(":")
                regions.append((name, int(lo, 16), int(hi, 16)))
    dis = load_disasm(disasm_path)

    per_sym = Counter()
    per_sym_region = defaultdict(Counter)
    per_sym_lvl = defaultdict(Counter)
    per_sym_region_lvl = defaultdict(Counter)
    addrs = []
    nloads = 0
    with open(script) as f:
        for line in f:
            m = SAMPLE.match(line.rstrip("\n"))
            if not m:
                continue
            addr = int(m.group(1), 16)
            rest = m.group(2)
            ip = int(m.group(3), 16)
            op = OP.search(rest)
            if not op or op.group(1) != "LOAD":
                continue
            lvl = LVL.search(rest)
            lvl = lvl.group(1) if lvl else "?"
            v = ip - base + foff
            ent = dis.get(v)
            if ent is None:
                continue
            sym = ent[0]
            nloads += 1
            per_sym[sym] += 1
            per_sym_lvl[sym][lvl] += 1
            addrs.append(addr)
            reg = "other"
            for name, lo, hi in regions:
                if lo <= addr < hi:
                    reg = name
                    break
            per_sym_region[sym][reg] += 1
            per_sym_region_lvl[(sym, reg)][lvl] += 1

    print(f"# resolved LOAD samples inside the binary: {nloads}")
    if not regions:
        addrs.sort()
        print("# address clusters (gap > 64 MB starts a new one)")
        lo = prev = addrs[0]
        n = 0
        for a in addrs:
            if a - prev > (64 << 20):
                print(f"  0x{lo:012x} .. 0x{prev:012x}  span={(prev-lo)/1e6:10.3f} MB  n={n}")
                lo = a
                n = 0
            prev = a
            n += 1
        print(f"  0x{lo:012x} .. 0x{prev:012x}  span={(prev-lo)/1e6:10.3f} MB  n={n}")
        return

    for sym, tot in per_sym.most_common(14):
        print(f"\n=== {sym}   loads={tot} ({100*tot/nloads:.2f}% of resolved)")
        for reg, n in per_sym_region[sym].most_common():
            lv = per_sym_region_lvl[(sym, reg)]
            hit = lv.get("L1 hit", 0)
            print(f"    {reg:12s} n={n:7d} ({100*n/tot:5.1f}%)  L1-hit {100*hit/max(n,1):5.1f}%  "
                  + " ".join(f"{k}={v}" for k, v in lv.most_common(5)))


if __name__ == "__main__":
    main()
