#!/usr/bin/env python3
"""perf study 38 — census of libc `memcpy`/`memset` CALL SITES in a binary.

    python3 tools/perf38/memcpy38.py <objdump-output> [symbol-substring]

Why this exists: perf study 14 removed a 2560-byte `memcpy` call from the
threat channel's empty-delta branch by chunking the copy by hand, and left a
source comment saying LLVM now emits 40 inline 64-byte moves. **It does not.**
Today's binary calls `memcpy@GLIBC` from that exact site. LLVM's
loop-idiom-recognize pass turns the chunked loop straight back into a memcpy,
so the comment is a claim about a compiler that is no longer the one we ship.

A hand-written source comment cannot hold a codegen fact. This script can:
it walks a disassembly, attributes every `call *<...>memcpy@GLIBC` /
`memset@GLIBC` to its enclosing symbol, and reports the immediately preceding
`mov $imm,%edx` (the size argument in the SysV ABI) when there is one, so a
constant-size call is visible as a constant.

Output is one row per call site, grouped by symbol, plus a total.
"""

import re
import sys
from collections import defaultdict

SYM = re.compile(r"^[0-9a-f]+ <(.+)>:$")
INSN = re.compile(r"^\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(.*)$")
# `mov $0xa00,%edx` / `mov $0xa00,%rdx` — the third SysV integer argument.
SIZE = re.compile(r"^mov\s+\$0x([0-9a-f]+),%[er]dx")
TARGET = re.compile(r"(memcpy|memset|memmove|bcmp|memcmp)@")


def main() -> None:
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    sym = "?"
    # Ring of recent instructions so the size argument can be found even when
    # LLVM schedules other setup between the `mov` and the `call`.
    recent: list[str] = []
    rows = defaultdict(list)
    for line in open(path):
        m = SYM.match(line.strip())
        if m:
            sym = m.group(1)
            recent = []
            continue
        m = INSN.match(line.rstrip())
        if not m:
            continue
        addr, text = m.group(1), m.group(2).strip()
        # The call is INDIRECT through the GOT (`call *0x...(%rip)`), so the
        # callee's name appears only in objdump's `#` comment. Match the whole
        # line, and strip the comment only for the size scan below.
        if text.startswith("call") and (t := TARGET.search(text)):
            size = "?"
            for prev in reversed(recent):
                s = SIZE.match(prev)
                if s:
                    size = str(int(s.group(1), 16))
                    break
                if prev.startswith(("call", "j")):
                    break
            rows[sym].append((addr, t.group(1), size))
        recent.append(text.split("#")[0].strip())
        recent = recent[-24:]

    total = 0
    for sym in sorted(rows, key=lambda s: -len(rows[s])):
        if want and want not in sym:
            continue
        for addr, fn, size in rows[sym]:
            print(f"{addr}  {fn:<8} size={size:<8} {sym}")
            total += 1
    print(f"\n{total} call sites" + (f" matching {want!r}" if want else ""))


if __name__ == "__main__":
    main()
