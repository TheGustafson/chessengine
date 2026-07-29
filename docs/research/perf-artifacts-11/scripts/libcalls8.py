#!/usr/bin/env python3
"""perf study 8 — attribute libc mem* call sites to the enclosing symbol.

    python3 libcalls8.py <objdump-d-C output> [symbol-substring]

perf cannot symbolise glibc's ifunc-resolved `__mem*_avx512_*` bodies, so a
multi-percent memcpy block shows up as bare libc addresses with no caller.
This walks the disassembly instead, so every `call <mem*@plt>` is reported
with the function it sits in and, where the size is an immediate, the size.
"""

import re
import sys

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else ""

SYM = re.compile(r"^[0-9a-f]+ <(.+)>:$")
INS = re.compile(r"^\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(\S+)\s*(.*)$")
CALL = re.compile(r"mem(cpy|move|set)")

cur = "?"
recent = []  # rolling window of recent instructions, for the size immediate
hits = []
for line in open(path):
    m = SYM.match(line)
    if m:
        cur = m.group(1)
        recent = []
        continue
    m = INS.match(line)
    if not m:
        continue
    addr, op, rest = m.group(1), m.group(2), m.group(3)
    if op.startswith("call") and CALL.search(rest):
        fn = CALL.search(rest).group(0)
        size = None
        for _, rop, rrest in reversed(recent[-12:]):
            mm = re.match(r"\$0x([0-9a-f]+),%(?:edx|rdx)\b", rrest)
            if rop.startswith("mov") and mm:
                size = int(mm.group(1), 16)
                break
        hits.append((cur, addr, fn, size))
    recent.append((addr, op, rest))

agg = {}
for sym, addr, fn, size in hits:
    if want and want not in sym:
        continue
    agg.setdefault((sym, fn, size), []).append(addr)

for (sym, fn, size), addrs in sorted(agg.items(), key=lambda kv: -(kv[0][2] or 0)):
    s = f"{size}B" if size is not None else "dyn"
    print(f"{s:>8}  {fn:8s} x{len(addrs):<3d}  {sym[:150]}")
