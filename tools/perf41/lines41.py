#!/usr/bin/env python3
"""perf study 41 — executed vs allocated bytes, per symbol, in 64-byte lines.

    python3 tools/perf41/lines41.py <binary> <perf-script-ip-sym-output>

Allocated size comes from the symbol table; executed size comes from the
distinct 64-byte cache lines that carried at least one sample. The difference
is cold code sitting inside a hot function — the only kind of footprint an
outlining arm can remove without changing what the engine computes.

`EXEC` is a lower bound (sampling misses a line that runs a few times in the
whole frame), so `COLD` is an upper bound on the prize. That is the useful
direction: if COLD is small, do not attempt the refactor.
"""

import re
import subprocess
import sys
from collections import defaultdict

LINE = 64


def symtab(path: str) -> list[tuple[int, int, str]]:
    out = subprocess.run(
        ["nm", "-C", "--print-size", "--numeric-sort", path],
        capture_output=True, text=True, check=True,
    ).stdout
    syms = []
    for ln in out.splitlines():
        p = ln.split(None, 3)
        if len(p) == 4 and p[2] in ("t", "T"):
            try:
                syms.append((int(p[0], 16), int(p[1], 16), p[3].strip()))
            except ValueError:
                pass
    return syms


def main() -> None:
    binary, script = sys.argv[1], sys.argv[2]
    syms = symtab(binary)

    lines: dict[str, set[int]] = defaultdict(set)
    hits: dict[str, int] = defaultdict(int)
    total = 0
    pat = re.compile(r"^\s*([0-9a-f]+)\s+(.*)$")
    for ln in open(script):
        m = pat.match(ln.rstrip("\n"))
        if not m:
            continue
        ip = int(m.group(1), 16)
        name = m.group(2).strip()
        total += 1
        lines[name].add(ip // LINE)
        hits[name] += 1

    size = {n: s for _, s, n in syms}
    rows = []
    for name, ls in lines.items():
        alloc = size.get(name, 0)
        execd = len(ls) * LINE
        rows.append((hits[name], name, alloc, execd))
    rows.sort(reverse=True)

    print(f"samples: {total}   line size: {LINE} B")
    print(f"{'%cyc':>6} {'ALLOC':>8} {'EXEC':>8} {'COLD':>8} {'exec%':>6}  symbol")
    sa = se = 0
    for h, name, alloc, execd in rows[:26]:
        if alloc == 0:
            continue
        execd = min(execd, alloc)
        sa += alloc
        se += execd
        print(f"{100.0 * h / total:>5.1f}% {alloc:>8} {execd:>8} {alloc - execd:>8} "
              f"{100.0 * execd / alloc:>5.1f}%  {name[:66]}")
    print(f"{'':>6} {sa:>8} {se:>8} {sa - se:>8}   top-26 total "
          f"(= {se / 32768:.2f} x 32 KB L1i executed, {sa / 32768:.2f} x allocated)")


if __name__ == "__main__":
    main()
