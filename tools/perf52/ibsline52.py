#!/usr/bin/env python3
"""perf study 52 -- PRECISE per-line / per-inline-frame census inside ONE symbol.

Study 33 decomposed `search::<NonPv>` with a sampled *cycles*/*stall* counter,
and its own §4.1 warns that such a counter "names where the ROB head parks, not
where the cost is" -- two of the three hottest lines it found were parking
spots.  This tool answers the same question with AMD IBS, which tags a micro-op
at ISSUE and records that op's OWN rip, so there is no skid by construction.

It also classifies each sampled op against the disassembly, which is the thing
no previous study did: whether the op is a load, and whether that load is
`%rsp`-relative (a spill reload) or a real data-structure access.  A static
count of the symbol says 50.5% of its load instructions are `%rsp`-relative;
only a dynamic census can say whether that is where the executed ops go.

    python3 ibsline52.py <ibs-script.txt> <binary> [symbol-regex]

<ibs-script.txt> is `perf script -F ip,sym,symoff,weight,data_src`.
"""

from __future__ import annotations

import collections
import re
import subprocess
import sys

# `perf script -F ip,sym,symoff,weight,data_src` puts the fields we want after
# the final '|': BLK <x> <weight> <ip> <symbol>+0x<off>
TAIL = re.compile(r"BLK\s+\S+\s+(\d+)\s+([0-9a-f]+)\s+(\S+)\+(0x[0-9a-f]+)")
MEMOP = re.compile(r"(^|[\s,])(-?(0x)?[0-9a-f]*)?\(%[a-z0-9]+(,%[a-z0-9]+(,\d)?)?\)")
NOMEM = {"lea", "nop", "nopw", "nopl", "endbr64"}
CMPS = {"cmp", "cmpb", "cmpl", "cmpw", "cmpq", "test", "testb", "testl",
        "testw", "testq", "push"}


def sym_range(binary: str, pat: str) -> tuple[str, int, int]:
    out = subprocess.run(["nm", "--print-size", binary],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split()
        if len(f) == 4 and re.search(pat, f[3]):
            return f[3], int(f[0], 16), int(f[1], 16)
    raise SystemExit(f"no sized symbol matching /{pat}/")


def disasm(binary: str, lo: int, hi: int) -> dict[int, tuple[str, str]]:
    """offset -> (mnemonic, operands), disassembled once."""
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn",
         f"--start-address=0x{lo:x}", f"--stop-address=0x{hi:x}", binary],
        capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+(\S+)\s*(.*)", line)
        if m:
            d[int(m.group(1), 16) - lo] = (m.group(2), m.group(3))
    return d


def classify(mnem: str, ops: str) -> str:
    ops = ops.split("#")[0]
    if not MEMOP.search(ops) or mnem in NOMEM:
        return "reg-only"
    dst = ops.rsplit(",", 1)[-1].strip() if "," in ops else ""
    src = ops.rsplit(",", 1)[0] if "," in ops else ops
    if MEMOP.search(dst) and not MEMOP.search(src) and mnem not in CMPS:
        return "store-rsp" if "(%rsp" in dst else "store-mem"
    return "load-rsp" if "(%rsp" in src else "load-mem"


def short(name: str) -> str:
    name = re.sub(r"::h[0-9a-f]{16}$", "", name)
    name = re.sub(r"\[[0-9a-f]{16}\]", "", name)
    return name.replace("chessengine::", "")


def main() -> int:
    script, binary = sys.argv[1], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else r"6Worker6search.*5NonPv"
    # `nm` reports the v0-MANGLED name (needed for objdump/addr2line ranges);
    # `perf script` demangles.  The two never compare equal, so the perf side
    # is matched on its own demangled pattern.  Getting this wrong silently
    # yields zero samples -- which is how it was caught.
    disp = sys.argv[4] if len(sys.argv) > 4 \
        else ">::search::<chessengine::search::NonPv>"
    sym, lo, size = sym_range(binary, pat)
    code = disasm(binary, lo, lo + size)

    ops_at: collections.Counter[int] = collections.Counter()
    loads_at: collections.Counter[int] = collections.Counter()
    lat_at: collections.Counter[int] = collections.Counter()
    tot_ops = tot_in = 0
    for line in open(script):
        m = TAIL.search(line)
        if not m:
            continue
        tot_ops += 1
        if not m.group(3).endswith(disp) and m.group(3) != disp:
            continue
        off = int(m.group(4), 16)
        if off >= size:
            continue
        tot_in += 1
        ops_at[off] += 1
        if "OP LOAD" in line:
            loads_at[off] += 1
            lat_at[off] += int(m.group(1))

    print(f"\n  symbol {short(sym)}")
    print(f"  ops sampled engine-wide {tot_ops:,}   in symbol {tot_in:,} "
          f"({100.0 * tot_in / max(tot_ops, 1):.2f}%)")
    print(f"  distinct offsets hit {len(ops_at):,} of {len(code):,} instructions")

    # --- the operand-class census, weighted by SAMPLED ops (dynamic) ---------
    dyn = collections.Counter()
    stat = collections.Counter()
    for off, (mn, opnds) in code.items():
        stat[classify(mn, opnds)] += 1
    for off, n in ops_at.items():
        if off in code:
            dyn[classify(*code[off])] += n
    print("\n  OPERAND CLASS -- static instruction count vs DYNAMIC sampled ops")
    print(f"    {'class':12s} {'static':>8s} {'static%':>8s} {'dyn ops':>9s} {'dyn%':>7s}")
    st = sum(stat.values()) or 1
    dt = sum(dyn.values()) or 1
    for k in ("load-rsp", "load-mem", "store-rsp", "store-mem", "reg-only"):
        print(f"    {k:12s} {stat[k]:8d} {100*stat[k]/st:7.1f}% "
              f"{dyn[k]:9d} {100*dyn[k]/dt:6.1f}%")

    # --- resolve every hit offset to its DWARF inline stack ------------------
    uniq = sorted(ops_at)
    p = subprocess.run(
        ["addr2line", "-i", "-f", "-p", "-e", binary]
        + [f"0x{lo + o:x}" for o in uniq],
        capture_output=True, text=True).stdout.splitlines()
    # `-p` prints one line per frame; continuation frames start with
    # " (inlined by) ".  Frames run innermost-first.
    stacks: list[list[str]] = []
    cur: list[str] = []
    for line in p:
        if line.startswith(" (inlined by)"):
            cur.append(line[len(" (inlined by) "):])
        else:
            if cur:
                stacks.append(cur)
            cur = [line]
    if cur:
        stacks.append(cur)
    if len(stacks) != len(uniq):
        print(f"\n  !! addr2line returned {len(stacks)} stacks for {len(uniq)} "
              f"addresses -- inline view suppressed", file=sys.stderr)
        return 0

    inner_ops = collections.Counter()
    inner_ld = collections.Counter()
    line_ops = collections.Counter()
    phase_ops = collections.Counter()
    phase_ld = collections.Counter()
    phase_lat = collections.Counter()
    for off, st_ in zip(uniq, stacks):
        n, nl = ops_at[off], loads_at[off]
        fn = st_[0].split(" at ")[0].strip()
        inner_ops[short(fn)] += n
        inner_ld[short(fn)] += nl
        loc = st_[0].split(" at ")[-1].strip()
        line_ops[loc.split("/")[-1]] += n
        # PHASE = outermost frame BELOW the symbol itself.  addr2line prints
        # frames innermost-first, so the symbol is last and the phase is the
        # one before it (or the symbol itself, for search's own code).
        names = [f.split(" at ")[0].strip() for f in st_]
        phase = short(names[-2]) if len(names) >= 2 else short(names[-1])
        phase_ops[phase] += n
        phase_ld[phase] += nl
        phase_lat[phase] += lat_at[off]

    print(f"\n  INNERMOST inline frame, by SAMPLED OPS (precise)")
    print(f"    {'ops%':>7s} {'ld%':>7s}  function")
    for fn, n in inner_ops.most_common(22):
        print(f"    {100*n/tot_in:6.2f}% {100*inner_ld[fn]/max(sum(loads_at.values()),1):6.2f}%  {fn}")

    print(f"\n  PHASE (outermost frame below the symbol), by SAMPLED OPS")
    print(f"    {'ops%':>7s} {'ld%':>7s} {'lat%':>7s}  phase")
    tl = sum(phase_lat.values()) or 1
    tld = sum(phase_ld.values()) or 1
    for fn, n in phase_ops.most_common(18):
        print(f"    {100*n/tot_in:6.2f}% {100*phase_ld[fn]/tld:6.2f}% "
              f"{100*phase_lat[fn]/tl:6.2f}%  {fn}")

    print(f"\n  TOP SOURCE LINES by SAMPLED OPS")
    for loc, n in line_ops.most_common(18):
        print(f"    {100*n/tot_in:6.2f}%  {loc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
