#!/usr/bin/env python3
"""perf study 34 — bucket a `perf annotate --stdio` dump into loop regions.

`init_quiets` inlines `generate_quiets`, the offense reverse-fan, the per-quiet
scoring loop AND `partial_insertion_sort`, so the 6.59% symbol share hides four
different mechanisms. Backward jumps are loop back-edges: grouping the annotated
cycle shares by the innermost enclosing back-edge gives a decomposition that
needs no extra build (study 23 had to mark the parts `#[inline(never)]`, which
cost +23.5 i/n and changed what it was measuring).

    python3 tools/perf34/regions34.py <annot.txt> [min_pct]
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(r"^\s*([\d.]+)\s*:\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
JMP = re.compile(r"^j")
TARGET = re.compile(r"^([0-9a-f]+)")


def main() -> None:
    path = sys.argv[1]
    min_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    rows = []
    with open(path) as fh:
        for line in fh:
            m = LINE.match(line)
            if m:
                rows.append((float(m.group(1)), int(m.group(2), 16), m.group(3), m.group(4)))
    if not rows:
        print("no annotated rows")
        return

    # Back-edges: a jump whose target is at or before the jump itself.
    edges = []  # (target, jump_addr)
    for pct, addr, op, args in rows:
        if not JMP.match(op):
            continue
        t = TARGET.match(args.strip())
        if not t:
            continue
        tgt = int(t.group(1), 16)
        if tgt <= addr:
            edges.append((tgt, addr))

    # Innermost enclosing loop for each instruction = smallest containing edge.
    loop_pct: dict[tuple[int, int], float] = defaultdict(float)
    loop_n: dict[tuple[int, int], int] = defaultdict(int)
    outside = 0.0
    for pct, addr, op, args in rows:
        best = None
        for tgt, j in edges:
            if tgt <= addr <= j and (best is None or (j - tgt) < (best[1] - best[0])):
                best = (tgt, j)
        if best is None:
            outside += pct
        else:
            loop_pct[best] += pct
            loop_n[best] += 1

    total = sum(p for p, _, _, _ in rows)
    print(f"annotated rows={len(rows)} total_pct={total:.2f} straight_line={outside:.2f}%")
    print(f"{'loop':>28}  {'span':>6} {'instr':>5} {'share':>7}")
    for k in sorted(loop_pct, key=lambda k: -loop_pct[k]):
        if loop_pct[k] < min_pct:
            continue
        tgt, j = k
        print(f"{tgt:#x}..{j:#x}".rjust(28) + f"  {j - tgt:6d} {loop_n[k]:5d} {loop_pct[k]:6.2f}%")


if __name__ == "__main__":
    main()
