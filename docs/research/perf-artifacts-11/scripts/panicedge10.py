#!/usr/bin/env python3
"""perf study 10 — find surviving bounds/panic edges ANYWHERE in the binary,
including the ones study 8's `bcheck8.py` cannot see.

    python3 panicedge10.py <binary> [top-n]

Why a second tool.  `bcheck8.py` classifies a conditional jump as a bounds
check by reading the jump's TARGET SYMBOL and matching /panic|slice_index/.
In this codebase that misses most of them: LLVM sinks the panic landing pad
into the *cold tail of the enclosing function*, so the disassembly reads

    ja  745e0 <...ThreatContext>::at_full+0x2530>

and the target symbol is the hot function itself.  Perf study 10 §2.1 found
five such edges in one 40-instruction loop that way — by hand.

Method: disassemble once, mark every basic block that CALLS a panic entry
point (`core::panicking::*`, `slice_*_fail`, `unwrap_failed`), walk backwards
from each call to the nearest preceding label/branch target to get the block
start, then report every conditional jump in the binary whose target is one
of those block starts, grouped by enclosing symbol.  Static counts only —
this is a CANDIDATE LIST, exactly like `bcheck8.py`.  Rank by
`static edges in the loop x trip count`, or price with `probe10.sh`.
"""

import collections
import re
import subprocess
import sys

binary = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25

out = subprocess.run(
    ["objdump", "-d", "--no-show-raw-insn", "-C", binary],
    capture_output=True, text=True, check=True,
).stdout

SYM = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN = re.compile(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$")
JCC = re.compile(r"^j(?!mp\b)[a-z]+$")
PANIC = re.compile(r"panic|unwrap_failed|slice_index|slice_end|slice_start|_fail|abort")

# addr -> (symbol, mnemonic, operand); ordered list of addrs
insns, order, sym_of = {}, [], {}
cur = None
for line in out.splitlines():
    m = SYM.match(line)
    if m:
        cur = m.group(2)
        continue
    m = INSN.match(line)
    if m and cur:
        a = int(m.group(1), 16)
        insns[a] = (m.group(2), m.group(3))
        sym_of[a] = cur
        order.append(a)

# Every address that is the target of SOME jump: the block-start candidates.
targets = set()
for a in order:
    mnem, ops = insns[a]
    if mnem.startswith("j"):
        t = re.match(r"^([0-9a-f]+)", ops.strip())
        if t:
            targets.add(int(t.group(1), 16))

# A block is "panicky" if, walking forward from a block start until the next
# block start, we hit a call to a panic entry point.
panic_blocks = set()
for i, a in enumerate(order):
    if a not in targets:
        continue
    for b in order[i:i + 24]:
        mnem, ops = insns[b]
        if mnem == "call" and PANIC.search(ops):
            panic_blocks.add(a)
            break
        if b != a and b in targets:
            break

edges = collections.Counter()
for a in order:
    mnem, ops = insns[a]
    if not JCC.match(mnem):
        continue
    t = re.match(r"^([0-9a-f]+)", ops.strip())
    if t and int(t.group(1), 16) in panic_blocks:
        edges[sym_of[a]] += 1

print(f"{len(panic_blocks)} panic landing pads; "
      f"{sum(edges.values())} conditional edges into them\n")
print(f"{'edges':>6}  symbol")
for sym, n in edges.most_common(top_n):
    print(f"{n:>6}  {sym}")
