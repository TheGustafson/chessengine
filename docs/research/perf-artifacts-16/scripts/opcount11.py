#!/usr/bin/env python3
"""perf study 11 — total i/n of one or more opcodes across the WHOLE profile.

    python3 opcount11.py <perf.data> <engine-i-per-node> <opcode> [opcode ...]

Study 10 measured `vpmovsxbw` inside ONE symbol per engine and compared the
two.  That is valid only if neither engine executes the opcode anywhere else —
an assumption study 10 listed as an open threat to validity (its §6.5) and
never discharged.  This walks EVERY symbol above 0.05% of the profile and sums
each opcode's sample weight, so the cross-engine comparison rests on a
whole-engine count rather than on a symbol-attribution assumption.
"""

import re
import subprocess
import sys
from collections import defaultdict

data = sys.argv[1]
per_node = float(sys.argv[2])
ops = sys.argv[3:]
if not ops:
    sys.exit("usage: opcount11.py <perf.data> <i-per-node> <opcode>...")

rep = subprocess.run(
    ["perf", "report", "-i", data, "--stdio", "--no-children",
     "--percent-limit", "0.05", "-s", "sym"],
    capture_output=True, text=True).stdout

SYM = re.compile(r"^\s*([0-9.]+)%\s+\[\.\]\s+(.*?)\s*$")
ROW = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*:\s*[0-9a-f]+:\s+(\S+)")

syms = []
for line in rep.splitlines():
    m = SYM.match(line)
    if m:
        syms.append((float(m.group(1)), m.group(2)))

totals = defaultdict(float)
rows = []
for share, sym in syms:
    ann = subprocess.run(
        ["perf", "annotate", "-i", data, "--stdio", "-s", sym,
         "--percent-limit", "0"],
        capture_output=True, text=True).stdout
    loc = defaultdict(float)
    tot = 0.0
    for line in ann.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        pct = float(m.group(1))
        tot += pct
        op = m.group(2)
        for o in ops:
            if op == o:
                loc[o] += pct
    if tot <= 0 or not loc:
        continue
    per = {o: share / 100.0 * per_node * loc[o] / tot for o in loc}
    for o, v in per.items():
        totals[o] += v
    rows.append((sum(per.values()), share, sym, per))

rows.sort(reverse=True)
w = max((len(o) for o in ops), default=8) + 2
print(f"{'share':>7} {'sum i/n':>8} " + "".join(f"{o:>{w}}" for o in ops) + "  symbol")
for _, share, sym, per in rows:
    cells = "".join(f"{per.get(o, 0.0):>{w}.1f}" for o in ops)
    print(f"{share:>6.2f}% {sum(per.values()):>8.1f} {cells}  {sym[:80]}")

print()
for o in ops:
    print(f"TOTAL {o:<14} = {totals[o]:8.1f} i/n   "
          f"({totals[o] / per_node * 100:5.2f}% of engine)")
