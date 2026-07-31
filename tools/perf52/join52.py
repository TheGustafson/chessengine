#!/usr/bin/env python3
"""perf study 49 — join, per instruction offset inside ONE symbol, the stall a
sampled non-precise counter attributes there against the load-miss latency IBS
measures there.

This is the test that separates the two live explanations for a symbol with a
high stall-per-load ratio:

  * REAL, low memory-level parallelism -- its own few misses are serial, so each
    blocks retirement for its whole latency.  Prediction: the stall lands ON the
    instructions IBS says are missing.
  * MISATTRIBUTION -- the stall belongs to a callee or to an overlapping load
    elsewhere, and the rip is captured in whatever independent work the machine
    ran ahead into.  Prediction: NO relationship between the two per offset.

Usage:
   join49.py <ibs-script-file> <symbol-substring> < <perf annotate --stdio out>
"""
import re
import sys
import collections

ANN = re.compile(r"^\s*([0-9.]+)\s*:\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
OFF = re.compile(r"\+(0x[0-9a-f]+)")


def read_ibs(path, sym_sub):
    lat = collections.Counter()
    miss = collections.Counter()
    ops = collections.Counter()
    for line in open(path, errors="replace"):
        if sym_sub not in line or "|" not in line:
            continue
        head, _, tail = line.rpartition("|")
        tok = tail.split()
        if len(tok) < 4:
            continue
        try:
            w = int(tok[2])
        except ValueError:
            continue
        m = OFF.search(tail)
        off = int(m.group(1), 16) if m else 0
        ops[off] += 1
        if w > 0:
            miss[off] += 1
            lat[off] += w
    return ops, miss, lat


def main() -> int:
    ibs_path, sym_sub = sys.argv[1], sys.argv[2]
    ops, miss, lat = read_ibs(ibs_path, sym_sub)
    rows = []
    base = None
    for line in sys.stdin:
        m = ANN.match(line)
        if not m:
            continue
        pct, addr = float(m.group(1)), int(m.group(2), 16)
        if base is None:
            base = addr
        rows.append((addr, pct, m.group(3), m.group(4)))
    if not rows:
        print("no annotated instructions", file=sys.stderr)
        return 1
    base = min(r[0] for r in rows)
    tot_pct = sum(r[1] for r in rows)
    tot_lat = sum(lat.values())

    # bucket both signals into 16-byte windows: IBS rips and the counter's rips
    # need not name the identical byte, and a window is the honest resolution.
    W = 16
    b_pct = collections.Counter()
    b_lat = collections.Counter()
    for addr, pct, _, _ in rows:
        b_pct[(addr - base) // W] += pct
    for off, l in lat.items():
        b_lat[off // W] += l

    keys = set(b_pct) | set(b_lat)
    xs = [100 * b_lat[k] / tot_lat if tot_lat else 0 for k in keys]
    ys = [100 * b_pct[k] / tot_pct if tot_pct else 0 for k in keys]
    n = len(keys)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    r = sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")

    print(f"## {sym_sub}: {n} 16-byte windows, "
          f"IBS miss-latency total {tot_lat}, counter total {tot_pct:.1f}%")
    print(f"## Pearson r between IBS miss latency and attributed stall = {r:+.3f}")
    print(f"{'window':>8s} {'IBS lat%':>9s} {'stall%':>8s}   verdict")
    top_lat = sorted(keys, key=lambda k: -b_lat[k])[:8]
    top_pct = sorted(keys, key=lambda k: -b_pct[k])[:8]
    print("-- windows carrying the most TRUE miss latency --")
    for k in top_lat:
        print(f"  +{k*W:#06x} {100*b_lat[k]/tot_lat:9.2f} "
              f"{100*b_pct[k]/tot_pct:8.2f}")
    print("-- windows carrying the most ATTRIBUTED stall --")
    for k in top_pct:
        print(f"  +{k*W:#06x} {100*b_lat[k]/tot_lat if tot_lat else 0:9.2f} "
              f"{100*b_pct[k]/tot_pct:8.2f}")
    # concentration: how much attributed stall sits in windows with ZERO
    # measured miss latency, and vice versa
    z1 = sum(b_pct[k] for k in keys if b_lat[k] == 0) / tot_pct * 100
    z2 = sum(b_lat[k] for k in keys if b_pct[k] == 0) / tot_lat * 100 if tot_lat else 0
    print(f"## attributed stall sitting in windows with ZERO measured miss "
          f"latency: {z1:.1f}%")
    print(f"## measured miss latency in windows with ZERO attributed stall: "
          f"{z2:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
