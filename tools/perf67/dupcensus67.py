#!/usr/bin/env python3
"""perf study 67 — THE DUPLICATION CENSUS.

  python3 tools/perf67/dupcensus67.py <dbg-binary> <symbol-substring> [...]

For each named symbol, resolve every instruction through `addr2line -i` and
roll the byte budget up by the outermost inlined callee; then sum the SAME
callee across symbols. `make_move_with` is one function in the source and PGO
emits it inside `do_move`, which it emits inside four search monomorphisations
-- so the machine holds several copies, each of which is fetched into a 32 KB
L1i on its own. `nm` cannot see this and neither can any instruction-count
share; a duplicated hot callee is invisible to every instrument this project
used before study 67.
"""
import collections
import re
import subprocess
import sys

dbg = sys.argv[1]
wants = sys.argv[2:]

nm = subprocess.run(["nm", "--print-size", "-C", dbg], capture_output=True, text=True).stdout
syms = []
for ln in nm.splitlines():
    p = ln.split(" ", 3)
    if len(p) < 4 or not p[1].strip():
        continue
    try:
        addr, size = int(p[0], 16), int(p[1], 16)
    except ValueError:
        continue
    name = p[3]
    if any(w in name for w in wants):
        syms.append((name, addr, size))

grand = collections.Counter()
copies = collections.defaultdict(set)
for name, addr, size in syms:
    od = subprocess.run(["objdump", "-d", "--start-address", hex(addr),
                         "--stop-address", hex(addr + size), dbg],
                        capture_output=True, text=True).stdout
    insns = []
    for ln in od.splitlines():
        m = re.match(r"^\s+([0-9a-f]+):\t([0-9a-f ]+)\t", ln)
        if m:
            insns.append((int(m.group(1), 16), len(m.group(2).split())))
    a2l = subprocess.run(["addr2line", "-e", dbg, "-f", "-C", "-i", "-a"],
                         input="\n".join(hex(a) for a, _ in insns),
                         capture_output=True, text=True).stdout
    stacks, cur, pend = {}, None, []
    for ln in a2l.splitlines():
        if ln.startswith("0x"):
            if cur is not None:
                stacks[cur] = pend
            cur, pend = int(ln, 16), []
        else:
            pend.append(ln)
    if cur is not None:
        stacks[cur] = pend
    local = collections.Counter()
    for a, sz in insns:
        fns = stacks.get(a, [])[0::2]
        fn = "(own code)" if len(fns) <= 1 else fns[-2].split("::{closure")[0]
        local[fn] += sz
    print(f"\n### {name[:100]}  ({size} B)")
    for fn, b in local.most_common(12):
        print(f"  {b:>7} {100*b/size:>5.1f}%  {fn[:88]}")
    for fn, b in local.items():
        grand[fn] += b
        copies[fn].add(name)

print("\n=== SAME CALLEE, SUMMED OVER THE NAMED SYMBOLS")
print(f"{'total B':>8}{'copies':>8}  callee")
for fn, b in grand.most_common(30):
    print(f"{b:>8}{len(copies[fn]):>8}  {fn[:88]}")
