#!/usr/bin/env python3
"""perf study 9 -- attribute libc mem* calls to their enclosing symbol.

    python3 libcalls9.py <objdump-dC output> [size-filter]

perf cannot symbolise glibc's ifunc-resolved `__mem*_avx512_*` bodies, so a
multi-percent memcpy block shows up as bare libc addresses with no caller.
This walks the disassembly instead. Study 8's version keyed on the call
target's *name*; this one keys on the PLT slot comment (`# ... <memcpy@...>`),
which is how the indirect `call *0x...(%rip)` form actually appears.

Also recovers the size argument when it is an immediate into %edx/%rdx.
"""

import re
import sys
from collections import defaultdict

path = sys.argv[1]
SYM = re.compile(r"^[0-9a-f]+ <(.+)>:$")
CALL = re.compile(r"^\s+[0-9a-f]+:\s+call\s+.*<(mem(?:cpy|move|set))@")
EDX = re.compile(r"^\s+[0-9a-f]+:\s+mov\s+\$0x([0-9a-f]+),%(?:edx|rdx)$")

cur = "?"
last_size = None
hits = defaultdict(lambda: defaultdict(int))
for line in open(path):
    m = SYM.match(line)
    if m:
        cur, last_size = m.group(1), None
        continue
    m = EDX.match(line)
    if m:
        last_size = int(m.group(1), 16)
        continue
    m = CALL.match(line)
    if m:
        hits[cur][(m.group(1), last_size)] += 1
        last_size = None

rows = []
for sym, per in hits.items():
    for (fn, size), n in per.items():
        rows.append((size or 0, n, fn, size or 0, sym))
rows.sort(reverse=True)
print(f"{'size':>8} {'n':>3}  {'fn':<7} symbol")
for size, n, fn, raw, sym in rows[:40]:
    s = str(raw) if raw else "?"
    print(f"{s:>8} {n:>3}  {fn:<7} {sym[:100]}")
