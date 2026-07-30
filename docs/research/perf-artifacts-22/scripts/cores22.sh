#!/usr/bin/env bash
# perf study 22 — per-core busy%, to pick a measurement core that is not being
# fought over.  A sibling perf study and a GPU trainer share this box; the first
# core this study pinned to (3) had another arm's bench on it and the harness
# process ran at 2.2% CPU.
set -euo pipefail
grep -E '^cpu[0-9]+ ' /proc/stat > /tmp/perf22-cores-a.txt
sleep "${1:-4}"
grep -E '^cpu[0-9]+ ' /proc/stat > /tmp/perf22-cores-b.txt
python3 - <<'PY'
def rd(f):
    d = {}
    for line in open(f):
        p = line.split()
        d[p[0]] = [int(x) for x in p[1:]]
    return d
a, b = rd('/tmp/perf22-cores-a.txt'), rd('/tmp/perf22-cores-b.txt')
for k in sorted(a, key=lambda x: int(x[3:])):
    d = [y - x for x, y in zip(a[k], b[k])]
    tot, idle = sum(d), d[3] + d[4]
    print(f'{k:6s} busy={100 * (tot - idle) / max(tot, 1):5.1f}%')
PY
