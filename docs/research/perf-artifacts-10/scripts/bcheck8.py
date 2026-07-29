#!/usr/bin/env python3
"""perf study 8 — find surviving BOUNDS CHECKS in a hot symbol, weighted by
the dynamic instructions they cost.

    python3 bcheck8.py <annot.txt> [<annot.txt> ...]

A bounds check LLVM cannot hoist is a `cmp`/`test` + a conditional jump whose
target is one of the panic landing pads.  Inside a short SIMD kernel loop that
is a large fraction of the loop body: removing ONE such pair from `l1_dot`'s
11-instruction group loop was -1.75% engine instructions and -5.25% engine
branches (perf study 8 §4.1).  This lists every conditional jump to a panic
target with the dynamic weight of the jump plus its comparison.
"""

import re
import sys

ROW = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
PANIC = re.compile(r"panic|unwrap_failed|slice_index|abort")
JCC = re.compile(r"^j(?!mp)[a-z]+$")

for path in sys.argv[1:]:
    rows = []
    for line in open(path):
        m = ROW.match(line)
        if m:
            rows.append((int(m.group(2), 16), float(m.group(1)), m.group(3), m.group(4).strip()))
    if not rows:
        continue
    # A jump inside the symbol to an address that is itself a panic call, or a
    # direct call-site name, both count.
    panic_addrs = set()
    for addr, _pct, op, rest in rows:
        if op.startswith("call") and PANIC.search(rest):
            panic_addrs.add(addr)
    # Blocks that begin a panic sequence: any address within 40 instructions
    # before a panic call with no intervening return.
    idx = {a: i for i, (a, _, _, _) in enumerate(rows)}
    pre_panic = set()
    for a in panic_addrs:
        i = idx[a]
        for j in range(max(0, i - 40), i + 1):
            pre_panic.add(rows[j][0])

    total = sum(r[1] for r in rows)
    hits = []
    for i, (addr, pct, op, rest) in enumerate(rows):
        if not JCC.match(op):
            continue
        m = re.match(r"([0-9a-f]+)", rest)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt in pre_panic or tgt in panic_addrs:
            cost = pct
            if i > 0 and rows[i - 1][2] in ("cmp", "test", "cmpq", "cmpl", "cmpb", "cmpw"):
                cost += rows[i - 1][1]
            hits.append((cost, addr, op, rest, rows[i - 1][2] if i else ""))

    name = path.split("annot-")[-1].replace(".txt", "")
    tot = sum(h[0] for h in hits)
    print(f"{name:24s} {len(hits):3d} bounds checks, {tot:6.2f}% of the symbol's instructions")
    for cost, addr, op, rest, prev in sorted(hits, reverse=True)[:6]:
        if cost >= 0.05:
            print(f"       {cost:6.2f}%  {addr:x}  {prev} + {op} {rest[:40]}")
