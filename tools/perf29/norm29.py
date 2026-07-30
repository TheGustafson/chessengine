#!/usr/bin/env python3
"""perf study 29 -- normalise a panel PER INSTRUCTION.

`probe-accpin` deletes work: pinning the accumulator source to one buffer lets
LLVM CSE loads away, so the probe runs 6.8% fewer instructions than the
control. Its raw cycle delta is therefore confounded -- some of it is better
locality and some is simply less to do.

IPC and stall-cycles-per-instruction are not confounded. They ask "did the
memory system serve this code better", independently of how much code there
was. That is the statistic that separates the two probes cleanly, and it is
the one the study quotes for its headline.

Usage: python3 norm29.py <panel-file> <control-arm>
"""

import re
import statistics as st
import sys
from collections import defaultdict

LINE = re.compile(
    r"^r(\d+)\s+(\S+)\s+nodes=(\d+)\s+i/n=\s*([\d.]+)\s+cyc/n=\s*([\d.]+)\s+"
    r"loads/n=\s*([\d.]+)\s+miss/n=\s*([\d.]+)\s+rate=([\d.]+)%\s+stall/n=\s*([\d.]+)"
)


def main():
    path, ctrl = sys.argv[1], sys.argv[2]
    v = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        m = LINE.match(line.strip())
        if not m:
            continue
        arm = m.group(2)
        i, c, ld, ms, stall = (float(m.group(k)) for k in (4, 5, 6, 7, 9))
        v[arm]["ipc"].append(i / c)
        v[arm]["stall_per_i"].append(stall / i)
        v[arm]["miss_per_load"].append(ms / ld)
    keys = ["ipc", "stall_per_i", "miss_per_load"]
    base = {k: st.median(v[ctrl][k]) for k in keys}
    print(f"# {path}  control={ctrl}  n={len(v[ctrl]['ipc'])}")
    print(f"{'arm':10s}{'IPC':>9s}{'xctrl':>9s}"
          f"{'stall/instr':>13s}{'d%':>9s}{'miss/load':>11s}{'d%':>9s}")
    order = [ctrl] + [a for a in v if a != ctrl]
    for a in order:
        ipc, sp, mp = (st.median(v[a][k]) for k in keys)
        print(f"{a:10s}{ipc:9.4f}{ipc / base['ipc']:9.4f}"
              f"{sp:13.4f}{100 * (sp - base['stall_per_i']) / base['stall_per_i']:+9.2f}"
              f"{mp:11.4f}{100 * (mp - base['miss_per_load']) / base['miss_per_load']:+9.2f}")


if __name__ == "__main__":
    main()
