#!/usr/bin/env python3
"""perf study 49 — the master per-symbol table.

Three views of the same symbols, each from a different instrument:
  loads/node        sampled `ls_dispatch.ld_dispatch`  x the measured total
  stall c/node      sampled `ex_no_retire.load_not_complete` x its total
  miss-lat c/node   IBS summed DC-miss latency share x `ls_alloc_mab_count`

The last column is the one that decides the study.  `ls_alloc_mab_count`
increments once per cycle per outstanding miss, so it IS the engine's total
"cycles loads spend missing" -- 22354/node here.  A symbol's own share of that
is the absolute ceiling on the retirement stall its own loads can cause: every
`load_not_complete` cycle charged to a symbol must lie inside a miss interval of
one of THAT symbol's loads.  So `stall / miss-lat` cannot exceed 100 %, and the
engine-wide value is 7.9 % because misses overlap ~6-way.
"""
import re
import subprocess
import sys

TOTALS = dict(loads=2275.86, stall=1764.05, mablat=22354.28)


def report(data, limit=0.15):
    out = subprocess.run(["perf", "report", "-i", data, "--stdio",
                          "-F", "overhead,symbol", "--percent-limit", str(limit)],
                         capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        m = re.match(r"^\s+([0-9.]+)%\s+\[\.\]\s+(.*)$", line)
        if m:
            d[short(m.group(2).strip())] = float(m.group(1))
    return d


def short(s):
    s = s.replace("chessengine::", "").replace("eval::nnue::", "")
    return s.replace("<", "").replace(">", "")


def ibs(path):
    import collections
    lat = collections.Counter()
    for line in open(path, errors="replace"):
        if "|" not in line:
            continue
        head, _, tail = line.rpartition("|")
        tok = tail.split()
        if len(tok) < 4:
            continue
        try:
            w = int(tok[2])
        except ValueError:
            continue
        if w > 0:
            lat[short(" ".join(tok[4:]).split("+0x")[0])] += w
    startup = ("init_magics", "load_v2_bytes", "quantise_halfka")
    tot = sum(v for k, v in lat.items() if not any(s in k for s in startup))
    return {k: 100 * v / tot for k, v in lat.items()}


def main():
    ld, lnc, il = report(sys.argv[1]), report(sys.argv[2]), ibs(sys.argv[3])
    keys = sorted(set(ld) | set(lnc), key=lambda k: -lnc.get(k, 0))
    print(f"{'symbol':52s} {'loads/n':>8s} {'stall c/n':>10s} "
          f"{'misslat c/n':>12s} {'stall/lat':>10s} {'OLD c/load':>11s}")
    for k in keys[:18]:
        L = ld.get(k, 0) / 100 * TOTALS["loads"]
        S = lnc.get(k, 0) / 100 * TOTALS["stall"]
        M = il.get(k, 0) / 100 * TOTALS["mablat"]
        conv = f"{100*S/M:9.1f}%" if M > 0 else "        --"
        cpl = f"{S/L:11.2f}" if L > 0 else "         --"
        print(f"{k[:52]:52s} {L:8.1f} {S:10.1f} {M:12.1f} {conv} {cpl}")
    L = TOTALS["loads"]; S = TOTALS["stall"]; M = TOTALS["mablat"]
    print(f"{'-- ENGINE':52s} {L:8.1f} {S:10.1f} {M:12.1f} {100*S/M:9.1f}% {S/L:11.2f}")


main()
