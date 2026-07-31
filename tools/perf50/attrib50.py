#!/usr/bin/env python3
"""perf study 50 — what KIND of instruction does a sampled stall counter land on?

`ex_no_retire.load_not_complete` counts cycles in which nothing retired and the
oldest un-retired op is a load that has not returned.  If the sample rip were
faithful, essentially every sample would land on an instruction that actually
performs a load.  This script joins `perf annotate --stdio` percentages against
a classification of each instruction, so the share landing on non-load
instructions is a direct, model-free measure of misattribution.

Usage:  perf annotate -i D --stdio --symbol=S | python3 attrib49.py [label]
"""
import re
import sys
import collections

LINE = re.compile(r"^\s*([0-9.]+)\s*:\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
# a memory SOURCE operand: something like 0x10(%rsp), (%rax), 0x0(%rax,%rbx,8)
MEMOP = re.compile(r"(^|[\s,])(-?(0x)?[0-9a-f]*)?\(%[a-z0-9]+(,%[a-z0-9]+(,\d)?)?\)")
NOMEM = {"lea", "nop", "nopw", "nopl", "endbr64"}


def classify(mnem: str, ops: str):
    ops = ops.split("//")[0].split("#")[0]
    has_mem = bool(MEMOP.search(ops)) and mnem not in NOMEM
    if not has_mem:
        return "no-memory"
    dst = ops.rsplit(",", 1)[-1].strip() if "," in ops else ""
    dst_is_mem = bool(MEMOP.search(dst))
    src = ops.rsplit(",", 1)[0] if "," in ops else ops
    src_is_mem = bool(MEMOP.search(src))
    if dst_is_mem and not src_is_mem:
        # push/store forms: mov %reg,(mem)
        return "store-only"
    return "load"


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "symbol"
    rows = []
    for line in sys.stdin:
        m = LINE.match(line)
        if not m:
            continue
        pct, addr, mnem, ops = float(m.group(1)), int(m.group(2), 16), m.group(3), m.group(4)
        rows.append((addr, pct, mnem, ops))
    if not rows:
        print("no annotated instructions parsed", file=sys.stderr)
        return 1
    rows.sort()
    by_class = collections.Counter()
    calls = [i for i, r in enumerate(rows) if r[2].startswith("call")]
    after_call = set()
    for i in calls:
        for j in range(i + 1, min(i + 4, len(rows))):
            after_call.add(rows[j][0])
    tot = sum(r[1] for r in rows)
    after_call_pct = 0.0
    for addr, pct, mnem, ops in rows:
        by_class[classify(mnem, ops)] += pct
        if addr in after_call:
            after_call_pct += pct
    print(f"## {label}: {len(rows)} instructions, {len(calls)} call sites, "
          f"{tot:.1f}% of symbol samples parsed")
    for k in ("load", "store-only", "no-memory"):
        print(f"   {k:12s} {by_class[k]:6.2f}%  ({100*by_class[k]/tot:5.1f}% of parsed)")
    print(f"   within 3 insns after a call: {after_call_pct:.2f}%")
    print("   top 8 instructions:")
    for addr, pct, mnem, ops in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"     {pct:6.2f}%  {addr:x}: {mnem:8s} {ops.split('//')[0][:46]:46s}"
              f" [{classify(mnem, ops)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
