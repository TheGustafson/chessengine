#!/usr/bin/env python3
"""perf study 33 -- WHICH LOAD IN THE SEARCH NODE MISSES, AND HOW FAR IT GOES.

    python3 tools/perf33/ibsline33.py <ibs-script> <ibs-mmap> <binary> [sym-regex]

Study 28 measured +303.4 load-stall cycles/node at `search::<NonPv>` and study
31 killed its only named mechanism, leaving a 19.9% symbol with no explanation.
A symbol share cannot explain itself and neither can a line share: what is
needed is, per load, WHERE IT CAME FROM.

IBS supplies exactly that -- a retired op's own IP, its data address, and the
level that served it.  This tool joins that to DWARF: IP -> inline stack ->
source line, then cross-tabulates source line x data source, restricted to
samples inside one symbol.

The output columns are the mechanism.  A line whose loads are 95% L1 hits costs
nothing however many samples it has; a line with a few thousand DRAM fills is
the node's real memory cost even if it is one source line.
"""

from __future__ import annotations

import collections
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from elfmap33 import ip_to_elf  # noqa: E402

# perf script's column order for IBS is FIXED regardless of the -F order:
#   <data addr> <raw data_src> |OP ..|LVL ..|SNP ..|TLB ..|LCK ..|BLK  ..  <ip>
# Note there is NO closing '|' after the BLK field, so the whole record is
# parsed in three pieces rather than by one monolithic regex -- the monolithic
# one silently matched nothing, which reads exactly like "no samples".
HEAD = re.compile(r"^\s*([0-9a-f]+)\s+[0-9a-f]+\s+\|")
FIELD = re.compile(r"\|(OP|LVL|TLB)\s+([^|]*?)\s*(?=\|)")
TAIL = re.compile(r"\|BLK\s+\S+\s+([0-9a-f]+)\s*$")
MMAP = re.compile(r"\[0x([0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0)[^]]*\]: r-xp")

LEVELS = ["L1 hit", "L2 hit", "L3 hit", "Local RAM", "Rmt RAM", "N/A"]


def sym_range(binary: str, pat: str) -> tuple[str, int, int]:
    out = subprocess.run(["nm", "-S", binary], capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split()
        if len(f) == 4 and re.search(pat, f[3]):
            return f[3], int(f[0], 16), int(f[0], 16) + int(f[1], 16)
    raise SystemExit(f"no sized symbol matching /{pat}/ in {binary}")


def resolve(binary: str, addrs: list[int]) -> dict[int, list[tuple[str, str]]]:
    """addr -> inline stack, innermost first, as (function, file:line)."""
    if not addrs:
        return {}
    out = subprocess.run(
        ["addr2line", "-i", "-f", "-p", "-e", binary] + [f"0x{a:x}" for a in addrs],
        capture_output=True, text=True).stdout.splitlines()
    stacks: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
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


def main() -> int:
    script_path, mmap_path, binary = sys.argv[1:4]
    pat = sys.argv[4] if len(sys.argv) > 4 else r"6Worker6search.*5NonPv"

    to_elf, vbase, pgoff, delta = ip_to_elf(mmap_path, binary)
    sym, lo, hi = sym_range(binary, pat)
    print(f"  symbol {sym[:60]}\n  ELF range 0x{lo:x}..0x{hi:x} ({hi - lo} bytes)")
    print(f"  text mapped at 0x{vbase:x} from file offset 0x{pgoff:x}; "
          f"p_vaddr - p_offset = 0x{delta:x}")

    # (file addr, level) counts for LOADS in the target symbol, plus global tallies.
    per_addr_lvl: collections.Counter[tuple[int, str]] = collections.Counter()
    data_addrs: collections.Counter[int] = collections.Counter()
    tot = insym = loads_insym = 0
    global_lvl: collections.Counter[str] = collections.Counter()
    for line in open(script_path):
        h, t = HEAD.match(line), TAIL.search(line)
        if not h or not t:
            continue
        f = dict(FIELD.findall(line))
        tot += 1
        op, lvl, ip = f.get("OP", "?"), f.get("LVL", "?"), int(t.group(1), 16)
        fa = to_elf(ip)
        if not (lo <= fa < hi):
            continue
        insym += 1
        if op != "LOAD":
            continue
        loads_insym += 1
        global_lvl[lvl] += 1
        per_addr_lvl[(fa, lvl)] += 1
        data_addrs[int(h.group(1), 16) >> 22] += 1  # 4 MB buckets

    print(f"\n  IBS samples {tot:,}; in symbol {insym:,} ({100.0*insym/max(tot,1):.2f}%); "
          f"LOADS in symbol {loads_insym:,}")
    print("  loads in symbol by data source: " + "  ".join(
        f"{k}={v:,} ({100.0*v/max(loads_insym,1):.1f}%)"
        for k, v in global_lvl.most_common()))

    stacks = resolve(binary, sorted({a for a, _ in per_addr_lvl}))

    # Aggregate to source line, and to the outermost inline frame below search.
    by_line: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    by_phase: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for (fa, lvl), n in per_addr_lvl.items():
        st = stacks.get(fa) or [("?", "?")]
        by_line[st[0][1]][lvl] += n
        by_phase[st[-2][0] if len(st) >= 2 else "search (own code)"][lvl] += n

    def table(title: str, d: dict[str, collections.Counter[str]], n: int = 22) -> None:
        print(f"\n  {title}")
        hdr = f"  {'loads':>8s} {'%miss':>6s} {'L2':>7s} {'L3':>7s} {'DRAM':>7s}  key"
        print(hdr)
        rows = sorted(d.items(), key=lambda kv: -(sum(kv[1].values())))
        for key, c in rows[:n]:
            t = sum(c.values())
            l1 = c.get("L1 hit", 0)
            l2, l3 = c.get("L2 hit", 0), c.get("L3 hit", 0)
            dram = c.get("Local RAM", 0) + c.get("Rmt RAM", 0)
            print(f"  {t:8,d} {100.0*(t-l1)/t:5.1f}% {l2:7,d} {l3:7,d} {dram:7,d}  {key[:60]}")

    table("BY SOURCE LINE -- loads inside the symbol, and where they were served",
          by_line)
    table("BY PHASE (outermost inline frame below search)", by_phase)

    print("\n  DATA-ADDRESS CLUSTERS (4 MB buckets, top 14) -- cross-check on the above")
    for b, n in data_addrs.most_common(14):
        print(f"  {n:8,d}  0x{b << 22:012x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
