#!/usr/bin/env python3
"""perf study 40 — pull one symbol out of an `objdump -d` dump and describe it.

    python3 tools/perf40/symscan40.py <dump.asm> <symbol-substring> [--dump]

Prints the instruction count, whether the symbol contains a `rep movs`, the
longest run of consecutive `vmov*` (which is what an INLINED 2560-byte
accumulator copy looks like — study 38 §3), and the opcode histogram. The point
is to read a codegen property off the binary instead of asserting it in prose:
the same source text compiles to a `memcpy` CALL in one context and 40 inline
zmm moves in another, and nothing in the tree was watching which.
"""

import re
import sys
from collections import Counter

INSN = re.compile(r"^\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(.*)$")
SYM = re.compile(r"^([0-9a-f]+) <(.+)>:")


def sym_block(path: str, name: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    on = False
    for line in open(path):
        m = SYM.match(line)
        if m:
            on = name in m.group(2)
            continue
        if on:
            if not line.strip():
                on = False
                continue
            mm = INSN.match(line)
            if mm:
                out.append((mm.group(1), mm.group(2).strip()))
    return out


def main() -> None:
    path, name = sys.argv[1], sys.argv[2]
    blk = sym_block(path, name)
    if not blk:
        print(f"{name}: not found in {path}")
        return
    ops = [t.split()[0] if t.split() else "?" for _, t in blk]
    best = cur = 0
    bestat = curat = None
    for a, t in blk:
        if t.startswith("vmov"):
            if cur == 0:
                curat = a
            cur += 1
            if cur > best:
                best, bestat = cur, curat
        else:
            cur = 0
    print(f"{path}  {name}")
    print(f"  instructions      : {len(blk)}")
    print(f"  rep movs/stos     : {sum(1 for _, t in blk if t.startswith('rep'))}")
    print(f"  calls             : {sum(1 for o in ops if o.startswith('call'))}")
    print(f"  longest vmov run  : {best} at {bestat}")
    print(f"  top opcodes       : {Counter(ops).most_common(10)}")
    if "--dump" in sys.argv:
        for a, t in blk:
            print(f"    {a}: {t}")


if __name__ == "__main__":
    main()
