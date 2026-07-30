#!/usr/bin/env python3
"""perf study 42 -- which DATA REGION does each cache miss come from.

    python3 tools/perf42/regions42.py <ibs-script> [--nodes N] [--period P]
                                      [--grain MB] [--min-samples K]

IBS tags a retired op with its own IP, its DATA ADDRESS and the level that
served it. Study 33 used the IP half. This study needs the ADDRESS half,
because the question -- what does the transposition table cost -- is a question
about one contiguous 16 MB allocation, not about a function.

Regions are built bottom-up: samples are binned to `--grain` MB blocks, blocks
with any traffic are merged when adjacent, and each resulting span is reported
with its level histogram. A span's SIZE is what identifies it: run the same
bench at two Hash sizes and the TT is the span that followed `Hash`. Nothing
else in the process does.

`--nodes` and `--period` convert sample counts to events per node: each sample
stands for exactly `period` ops of the sampled class, so
    per-node = samples * period / nodes.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

HEAD = re.compile(r"^\s*([0-9a-f]+)\s+[0-9a-f]+\s+\|")
FIELD = re.compile(r"\|(OP|LVL|TLB)\s+([^|]*?)\s*(?=\|)")
TAIL = re.compile(r"\|BLK\s+\S+\s+([0-9a-f]+)\s*$")

# Anything that is not an L1/L2/L3 hit is DRAM: perf spells the far level
# differently across versions ("RAM hit", "Local RAM"), and pattern-matching
# the spelling is how a study silently loses its deepest misses.
LEVELS = ("L1", "L2", "L3", "DRAM")


def level_of(lvl: str) -> str:
    if "L1" in lvl:
        return "L1"
    if "L2" in lvl:
        return "L2"
    if "L3" in lvl:
        return "L3"
    return "DRAM"


def parse(path: str):
    for line in open(path):
        h = HEAD.match(line)
        if not h:
            continue
        addr = int(h.group(1), 16)
        if addr == 0:
            continue
        f = dict(FIELD.findall(line))
        op, lvl = f.get("OP", ""), f.get("LVL", "")
        if lvl in ("", "N/A") or op not in ("LOAD", "STORE"):
            continue
        t = TAIL.search(line)
        yield addr, op, level_of(lvl), f.get("TLB", ""), int(t.group(1), 16) if t else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--nodes", type=float, default=0.0)
    ap.add_argument("--period", type=float, default=50021.0)
    ap.add_argument("--grain", type=float, default=1.0, help="bin size in MB")
    ap.add_argument("--min-samples", type=int, default=20)
    args = ap.parse_args()

    grain = int(args.grain * 1024 * 1024)
    bins: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    tlb: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for addr, op, lvl, tl, _ip in parse(args.script):
        b = addr // grain
        bins[b][lvl] += 1
        bins[b][op] += 1
        bins[b]["n"] += 1
        tlb[b][tl] += 1
        total[lvl] += 1
        total["n"] += 1

    # Merge adjacent occupied bins into spans.
    spans = []
    for b in sorted(bins):
        if spans and b == spans[-1][1] + 1:
            spans[-1][1] = b
        else:
            spans.append([b, b])

    scale = args.period / args.nodes if args.nodes else 0.0
    print(f"samples={total['n']}  period={args.period:g}  nodes={args.nodes:g}")
    print(
        "levels: "
        + "  ".join(f"{k}={total[k]} ({100.0 * total[k] / max(total['n'], 1):.2f}%)" for k in LEVELS)
    )
    if scale:
        print("        " + "  ".join(f"{k}/node={total[k] * scale:.3f}" for k in LEVELS))
    print()
    hdr = f"{'span (hex)':>34} {'MB':>8} {'samples':>9} {'%':>6}"
    for k in LEVELS:
        hdr += f" {k:>8}"
    hdr += f" {'ld':>8} {'st':>8} {'tlbmiss':>8}"
    if scale:
        hdr += f" {'L3+DRAM/node':>13}"
    print(hdr)
    for lo, hi in spans:
        c = collections.Counter()
        t = collections.Counter()
        for b in range(lo, hi + 1):
            c.update(bins[b])
            t.update(tlb[b])
        if c["n"] < args.min_samples:
            continue
        mb = (hi - lo + 1) * grain / 1048576.0
        row = (
            f"{lo * grain:#018x}-{(hi + 1) * grain:#012x} {mb:8.0f} "
            f"{c['n']:9d} {100.0 * c['n'] / max(total['n'], 1):5.2f}%"
        )
        for k in LEVELS:
            row += f" {c[k]:8d}"
        miss = sum(v for k, v in t.items() if "hit" not in k.lower() and k not in ("", "N/A"))
        row += f" {c['LOAD']:8d} {c['STORE']:8d} {miss:8d}"
        if scale:
            row += f" {(c['L3'] + c['DRAM']) * scale:13.4f}"
        print(row)


if __name__ == "__main__":
    main()
