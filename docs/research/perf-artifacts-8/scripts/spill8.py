#!/usr/bin/env python3
"""perf study 8 — the spill signature of a symbol.

    python3 spill8.py <annot.txt>

impl-1 established that a monolithic hot function's cost is register pressure,
not code size, and that the diagnostic is the fraction of DYNAMIC instructions
that are stack-relative `mov`s (`%rsp`/`%rbp` base with no index).  `next_move`
went 13.5% -> 4.9% when its stage-inits were outlined, for -5.2% engine
instructions.  This prints the same statistic, split load/store, plus the
static count, so a candidate can be ranked before it is touched.
"""

import re
import sys

ROW = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-f]+):\s+(\S+)\s+(.*)$")
# A stack slot access: %rsp base, optionally with a displacement and/or an
# index (PGO indexes spill arrays).  %rbp is deliberately NOT accepted: the
# release profile omits frame pointers, so %rbp is an ordinary GPR and
# counting `0x8(%rbp)` as a spill reported make_move_with at a nonsense 64%.
STACK = re.compile(r"\(%rsp[,)]")

tot = sto = ld = 0.0
n_static = n_stack = 0
for line in open(sys.argv[1]):
    m = ROW.match(line)
    if not m:
        continue
    pct, _addr, op, rest = float(m.group(1)), m.group(2), m.group(3), m.group(4)
    tot += pct
    n_static += 1
    if not op.startswith(("mov", "vmov")):
        continue
    if not STACK.search(rest):
        continue
    n_stack += 1
    # "mov <src>,<dst>" — a stack destination is a spill (store).
    dst = rest.split(",")[-1]
    if STACK.search(dst):
        sto += pct
    else:
        ld += pct

print(f"static instructions      : {n_static}")
print(f"static stack-rel movs    : {n_stack} ({n_stack / max(n_static, 1) * 100:.1f}%)")
print(f"dynamic stack-rel movs   : {(sto + ld) / max(tot, 1e-9) * 100:.2f}%  of the symbol")
print(f"   of which stores(spill): {sto / max(tot, 1e-9) * 100:.2f}%")
print(f"   of which loads (fill) : {ld / max(tot, 1e-9) * 100:.2f}%")
