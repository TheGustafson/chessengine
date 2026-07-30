#!/usr/bin/env python3
"""perf study 33 -- rank the engine's loads by WHAT THEY COST, not by how many.

    python3 tools/perf33/ibsmiss33.py <ibs-script> <ibs-mmap> <binary> [sym-regex]

`ibsline33.py` showed that 90.5% of the loads inside `search::<NonPv>` are L1
hits.  That kills the naive reading of study 28's +303.4 cycles/node but it
does not name the mechanism, because a load COUNT is the wrong ranking: one
DRAM fill outweighs fifty L1 hits.  This tool ranks by the non-L1 loads
themselves, and reports three views:

  1. per SYMBOL, engine-wide -- where the misses are at all;
  2. per SOURCE LINE inside a chosen symbol, ranked by misses;
  3. per DATA REGION (address cluster), so a line's misses can be attributed to
     a named table rather than guessed at.

Levels are taken verbatim from perf's IBS decode; the exact spelling ("RAM hit"
vs "Local RAM") differs between perf versions, so anything that is not an L1,
L2 or L3 hit is counted as DRAM rather than pattern-matched by name.
"""

from __future__ import annotations

import bisect
import collections
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from elfmap33 import ip_to_elf  # noqa: E402

HEAD = re.compile(r"^\s*([0-9a-f]+)\s+[0-9a-f]+\s+\|")
FIELD = re.compile(r"\|(OP|LVL|TLB)\s+([^|]*?)\s*(?=\|)")
TAIL = re.compile(r"\|BLK\s+\S+\s+([0-9a-f]+)\s*$")
MMAP = re.compile(r"\[0x([0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0)[^]]*\]: r-xp")


def symbols(binary: str) -> tuple[list[int], list[tuple[int, str]]]:
    out = subprocess.run(["nm", "-S", binary], capture_output=True, text=True).stdout
    syms = []
    for line in out.splitlines():
        f = line.split()
        if len(f) == 4 and f[2].lower() in "tw":
            syms.append((int(f[0], 16), int(f[1], 16), f[3]))
    syms.sort()
    starts = [s[0] for s in syms]
    return starts, [(s[0] + s[1], s[2]) for s in syms]


def short(n: str) -> str:
    n = re.sub(r"^_R.*?11chessengine", "", n)
    return n[-70:]


def resolve(binary: str, addrs: list[int]) -> dict[int, list[tuple[str, str]]]:
    if not addrs:
        return {}
    out = subprocess.run(
        ["addr2line", "-i", "-f", "-p", "-e", binary] + [f"0x{a:x}" for a in addrs],
        capture_output=True, text=True).stdout.splitlines()
    stacks, cur = [], []
    for ln in out:
        if ln.startswith(" (inlined by) "):
            body = ln[len(" (inlined by) "):]
        else:
            if cur:
                stacks.append(cur)
            cur = []
            body = ln
        fn, _, loc = body.partition(" at ")
        cur.append((fn.strip(), loc.strip().split("/")[-1]))
    if cur:
        stacks.append(cur)
    return dict(zip(addrs, stacks))


class Tally:
    __slots__ = ("l1", "l2", "l3", "dram")

    def __init__(self) -> None:
        self.l1 = self.l2 = self.l3 = self.dram = 0

    def add(self, lvl: str) -> None:
        if lvl == "L1 hit":
            self.l1 += 1
        elif lvl == "L2 hit":
            self.l2 += 1
        elif lvl == "L3 hit":
            self.l3 += 1
        else:
            self.dram += 1

    @property
    def total(self) -> int:
        return self.l1 + self.l2 + self.l3 + self.dram

    @property
    def miss(self) -> int:
        return self.l2 + self.l3 + self.dram

    @property
    def deep(self) -> int:
        """L3 + DRAM -- the fills that leave the core complex's L2."""
        return self.l3 + self.dram


def table(title: str, d: dict[str, Tally], key, n: int, denom: int) -> None:
    print(f"\n  {title}")
    print(f"  {'miss':>7s} {'%engine':>8s} {'loads':>8s} {'%miss':>6s} "
          f"{'L2':>7s} {'L3':>6s} {'DRAM':>5s}  key")
    for k, t in sorted(d.items(), key=lambda kv: -key(kv[1]))[:n]:
        print(f"  {t.miss:7,d} {100.0 * t.miss / max(denom, 1):7.2f}% {t.total:8,d} "
              f"{100.0 * t.miss / max(t.total, 1):5.1f}% {t.l2:7,d} {t.l3:6,d} "
              f"{t.dram:5,d}  {k[:62]}")


def main() -> int:
    script_path, mmap_path, binary = sys.argv[1:4]
    pat = sys.argv[4] if len(sys.argv) > 4 else r"6Worker6search.*5NonPv"

    to_elf, vbase, foff, delta = ip_to_elf(mmap_path, binary)
    starts, ends = symbols(binary)

    per_sym: dict[str, Tally] = collections.defaultdict(Tally)
    tgt_addr_lvl: collections.Counter[tuple[int, str]] = collections.Counter()
    tgt_region: dict[int, Tally] = collections.defaultdict(Tally)
    all_loads = 0

    for line in open(script_path):
        h, t = HEAD.match(line), TAIL.search(line)
        if not h or not t:
            continue
        f = dict(FIELD.findall(line))
        if f.get("OP") != "LOAD":
            continue
        lvl = f.get("LVL", "?")
        if lvl == "N/A":
            continue
        fa = to_elf(int(t.group(1), 16))
        i = bisect.bisect_right(starts, fa) - 1
        if i < 0 or fa >= ends[i][0]:
            continue
        name = ends[i][1]
        all_loads += 1
        per_sym[name].add(lvl)
        if re.search(pat, name):
            tgt_addr_lvl[(fa, lvl)] += 1
            tgt_region[int(h.group(1), 16) >> 20].add(lvl)  # 1 MB buckets

    eng_miss = sum(t.miss for t in per_sym.values())
    print(f"  engine loads sampled {all_loads:,}; non-L1 {eng_miss:,} "
          f"({100.0 * eng_miss / max(all_loads, 1):.1f}%)")
    table("PER SYMBOL, ranked by NON-L1 LOADS (engine-wide)",
          per_sym, lambda t: t.miss, 14, eng_miss)

    stacks = resolve(binary, sorted({a for a, _ in tgt_addr_lvl}))
    by_line: dict[str, Tally] = collections.defaultdict(Tally)
    by_phase: dict[str, Tally] = collections.defaultdict(Tally)
    for (fa, lvl), n in tgt_addr_lvl.items():
        st = stacks.get(fa) or [("?", "?")]
        for _ in range(n):
            by_line[st[0][1]].add(lvl)
            by_phase[st[-2][0] if len(st) >= 2 else "search (own code)"].add(lvl)

    tgt_miss = sum(t.miss for t in by_line.values())
    tgt_tot = sum(t.total for t in by_line.values())
    print(f"\n  TARGET SYMBOL: {tgt_tot:,} loads, {tgt_miss:,} non-L1 "
          f"({100.0 * tgt_miss / max(tgt_tot, 1):.1f}%) = "
          f"{100.0 * tgt_miss / max(eng_miss, 1):.2f}% of the engine's misses")
    table("  ...BY SOURCE LINE, ranked by MISSES", by_line, lambda t: t.miss, 20, eng_miss)
    table("  ...BY PHASE, ranked by MISSES", by_phase, lambda t: t.miss, 14, eng_miss)

    print("\n  ...BY DATA REGION (1 MB buckets of the DATA address, top 16)")
    print(f"  {'miss':>7s} {'loads':>8s} {'%miss':>6s}  data address bucket")
    for b, t in sorted(tgt_region.items(), key=lambda kv: -kv[1].miss)[:16]:
        print(f"  {t.miss:7,d} {t.total:8,d} {100.0 * t.miss / max(t.total, 1):5.1f}%  "
              f"0x{b << 20:012x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
