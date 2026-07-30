#!/usr/bin/env python3
"""perf study 40 — add a DERIVED counter column to a panel log.

    python3 tools/perf40/derive40.py <panel.txt> <name>=<A>-<B> [more...] > out.txt

Why this exists: on Zen 4, `ex_no_retire.thread_not_selected` counts the cycles
in which this thread retired nothing **because the other SMT thread on the same
core was selected instead**. That is the single largest source of run-to-run
noise in a panel pinned with `taskset` to one logical CPU — the sibling's load
is not under our control and it moves with whatever else the box is doing.

Subtracting it gives `cycclean = cycles - thread_not_selected`: the same
measurement with the machine's own accounting of SMT interference removed. It is
not a different quantity being measured, it is the same one with a *measured*
confound taken out, so the null decides whether it is an improvement — see
`spread40.py` on the byte-identical null for both columns.
"""

import re
import sys

ROW = re.compile(r"^(r\d+\s+\S+\s+nodes=\d+\s+)(.*)$")


def main() -> None:
    path = sys.argv[1]
    specs = []
    for s in sys.argv[2:]:
        name, expr = s.split("=", 1)
        a, b = expr.split("-", 1)
        specs.append((name, a, b))
    for line in open(path):
        m = ROW.match(line.rstrip())
        if not m:
            continue
        head, rest = m.group(1), m.group(2)
        cols = {}
        for tok in rest.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    cols[k] = float(v)
                except ValueError:
                    pass
        extra = ""
        for name, a, b in specs:
            if a in cols and b in cols:
                extra += f"{name}={cols[a] - cols[b]:.6f} "
        print(head + rest.strip() + " " + extra)


if __name__ == "__main__":
    main()
