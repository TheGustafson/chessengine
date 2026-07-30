#!/usr/bin/env python3
"""perf study 29 -- STATIC load census of a disassembly, per symbol.

The exact-count chain is: instrumented call/row counts (exact integers) x
loads-per-call (static shape). This script establishes the second factor with
no estimation in it: it disassembles the binary and, for every symbol matching
a substring, counts memory-SOURCE operands, separating vector loads (the
kernels' streams) from scalar ones and from stores.

`lea` computes an address and does NOT touch memory; padding mnemonics
(`nop*`, `data16`) carry a fake operand for encoding length only. Both are
excluded.

Usage: objdump -d --no-show-raw-insn bin | python3 loadscan29.py <substr>...
"""

import re
import sys
from collections import Counter, defaultdict

SYM = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN = re.compile(r"^\s+[0-9a-f]+:\s+([a-z0-9.]+)\s*(.*)$")
MEM = re.compile(r"(-?0x[0-9a-f]+)?\(%(\w+)(?:,%(\w+)(?:,(\d))?)?\)")
SKIP = ("lea", "nop", "data16", "cs", "xchg")


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
    wants = sys.argv[1:]
    cur = None
    loads = defaultdict(Counter)
    stores = defaultdict(Counter)
    insns = Counter()
    for line in sys.stdin:
        line = line.rstrip("\n")
        s = SYM.match(line)
        if s:
            cur = s.group(2)
            continue
        if cur is None or not any(w in cur for w in wants):
            continue
        i = INSN.match(line)
        if not i:
            continue
        mnem, ops = i.group(1), i.group(2).split("#")[0].strip()
        if any(mnem.startswith(p) for p in SKIP):
            continue
        insns[cur] += 1
        if not ops or not MEM.search(ops):
            continue
        parts = split_ops(ops)
        # AT&T: the destination is the last operand. A pure store has the
        # memory operand there and nowhere else.
        memlast = MEM.search(parts[-1]) is not None
        memearlier = any(MEM.search(p) for p in parts[:-1])
        if memlast and not memearlier and mnem.startswith(("mov", "vmov")):
            stores[cur][mnem] += 1
        else:
            loads[cur][mnem] += 1

    for sym in sorted(insns, key=lambda s: -insns[s]):
        ld, st = loads[sym], stores[sym]
        vec_ld = sum(v for k, v in ld.items() if k.startswith("v"))
        print(f"\n=== {sym}")
        print(f"    insns={insns[sym]}  loads={sum(ld.values())} "
              f"(vector {vec_ld})  stores={sum(st.values())}")
        for k, v in sorted(ld.items(), key=lambda x: -x[1]):
            print(f"      load  {v:5d}  {k}")
        for k, v in sorted(st.items(), key=lambda x: -x[1]):
            print(f"      store {v:5d}  {k}")


if __name__ == "__main__":
    main()
