#!/usr/bin/env python3
"""Perf study 50 — attribute IBS load samples to NAMED MEMORY REGIONS.

Study 49's `ibs49.py` censuses IBS by *symbol*: which code missed. This asks the
other question, which nothing in the repo has asked: **which ARRAY did it miss
in.** IBS records the data address (`PERF_SAMPLE_ADDR`, `perf record -d`) and
the measured DC-miss latency (`PERF_SAMPLE_WEIGHT`, `-W`), and the engine's
`foldcount` build registers every hot allocation's `[lo, hi)` with
`foldcount::region`, so the join is exact — no model, no attribution heuristic.

The region dump MUST come from the SAME PROCESS as the samples: these are heap
addresses and ASLR re-rolls them every run.

    perf script -i x.data -F ip,sym,symoff,addr,weight,data_src \
      | python3 ibsaddr50.py FOLDCOUNT.txt [SYMBOL-SUBSTRING ...]
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"^\s*(?P<addr>[0-9a-f]+)\s+(?P<src>[0-9a-f]+)\s+"
    r"\|OP (?P<op>[^|]*)\|LVL (?P<lvl>[^|]*)\|SNP [^|]*\|TLB (?P<tlb>[^|]*)"
    r"\|LCK [^|]*\|BLK\s+\S+\s+(?P<w>\d+)\s+(?P<ip>[0-9a-f]+)\s+(?P<sym>.*)$"
)


def load_regions(path):
    regions = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("FOLDREGION "):
                continue
            parts = line.split()
            name = parts[1]
            lo = int(parts[2].split("=")[1], 16)
            hi = int(parts[3].split("=")[1], 16)
            regions.append((lo, hi, name))
    regions.sort()
    return regions


def classify(regions, addr):
    # Linear scan: <=16 regions, and clarity beats a bisect here.
    for lo, hi, name in regions:
        if lo <= addr < hi:
            return name
    if 0x7FF000000000 <= addr:
        return "(stack/mmap)"
    return "(other)"


def main():
    fold = sys.argv[1]
    wanted = [w for w in sys.argv[2:]] or ["forward_into"]
    regions = load_regions(fold)

    # per (symbol-bucket, region) -> [loads, misses, latency]
    tab = defaultdict(lambda: [0, 0, 0])
    lvls = defaultdict(lambda: defaultdict(int))
    sym_tot = defaultdict(lambda: [0, 0, 0])
    engine = [0, 0, 0]

    for line in sys.stdin:
        m = LINE.match(line)
        if not m or "LOAD" not in m.group("op"):
            continue
        lvl = m.group("lvl").strip()
        addr = int(m.group("addr"), 16)
        w = int(m.group("w"))
        sym = m.group("sym")
        miss = 0 if lvl == "L1 hit" else 1
        engine[0] += 1
        engine[1] += miss
        engine[2] += w
        for tag in wanted:
            if tag in sym:
                reg = classify(regions, addr)
                cell = tab[(tag, reg)]
                cell[0] += 1
                cell[1] += miss
                cell[2] += w
                sym_tot[tag][0] += 1
                sym_tot[tag][1] += miss
                sym_tot[tag][2] += w
                lvls[tag][lvl] += 1

    print(f"ENGINE loads={engine[0]}  misses={engine[1]} "
          f"({100.0 * engine[1] / max(1, engine[0]):.2f}%)  latency={engine[2]}")
    for tag in wanted:
        t = sym_tot[tag]
        if not t[0]:
            print(f"\n=== {tag}: no samples")
            continue
        print(f"\n=== {tag}: loads={t[0]} ({100.0 * t[0] / max(1, engine[0]):.2f}% of engine)  "
              f"misses={t[1]} ({100.0 * t[1] / t[0]:.2f}% miss rate)  "
              f"latency={t[2]} ({100.0 * t[2] / max(1, engine[2]):.2f}% of engine)")
        print(f"{'region':16s} {'loads':>8s} {'%loads':>7s} {'misses':>8s} "
              f"{'missrate':>9s} {'%sym-miss':>10s} {'latency':>10s} {'%sym-lat':>9s}")
        rows = [(k[1], v) for k, v in tab.items() if k[0] == tag]
        rows.sort(key=lambda r: -r[1][2])
        for reg, v in rows:
            print(f"{reg:16s} {v[0]:8d} {100.0 * v[0] / t[0]:6.2f}% {v[1]:8d} "
                  f"{100.0 * v[1] / max(1, v[0]):8.2f}% {100.0 * v[1] / max(1, t[1]):9.2f}% "
                  f"{v[2]:10d} {100.0 * v[2] / max(1, t[2]):8.2f}%")
        print("  levels: " + "  ".join(
            f"{k}={v}" for k, v in sorted(lvls[tag].items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
