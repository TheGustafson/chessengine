#!/usr/bin/env python3
"""Aggregate a raw callgrind file into EXACT per-function self-Ir and call counts.

`callgrind_annotate` splits one function across every source file its inlined
code came from, which makes a Rust function with heavy `core::` inlining appear
as six rows.  This merges by function name, and — the part annotate cannot give
you at all — reports the exact number of CALLS to each function, parsed from the
`calls=` records.  Both numbers are deterministic: callgrind counts, it does not
sample.

Usage:
    cgsum17.py <callgrind.out> <nodes> [--top N] [--grep REGEX] [--callers FN]
"""

import re
import sys
from collections import defaultdict


def parse(path, event=0):
    """Return (self_by_fn, calls_by_callee, calls_by_edge, inclusive_by_edge).

    `event` selects a column when the file carries several (`--branch-sim=yes`
    emits `Ir Bc Bcm Bi Bim`); the `events:` header line names them.
    """
    names = {}          # (kind, id) -> name
    self_ir = defaultdict(int)
    calls_by_callee = defaultdict(int)
    calls_by_edge = defaultdict(int)
    incl_by_edge = defaultdict(int)

    cur_fn = None
    cur_cfn = None
    pending_calls = 0

    name_re = re.compile(r"^(fl|fi|fe|fn|cob|cfi|cfn|ob)=(?:\((\d+)\))?\s*(.*)$")

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line[0] == "#":
                continue
            m = name_re.match(line)
            if m:
                kind, num, nm = m.group(1), m.group(2), m.group(3)
                key = "fn" if kind in ("fn", "cfn") else "fl"
                if num is not None:
                    if nm:
                        names[(key, num)] = nm
                    else:
                        nm = names.get((key, num), f"?{num}")
                if kind == "fn":
                    cur_fn = nm
                    cur_cfn = None
                elif kind == "cfn":
                    cur_cfn = nm
                continue
            if line.startswith("calls="):
                pending_calls = int(line.split("=", 1)[1].split()[0])
                continue
            if line[0].isdigit() or line[0] in "+-*":
                parts = line.split()
                # Callgrind OMITS TRAILING ZERO EVENT COLUMNS. Treating a short
                # line as unparseable and `continue`-ing was a real bug: it left
                # `pending_calls` unconsumed, so that `calls=` record was lost
                # AND the next genuine self-cost line was charged to the call
                # edge. Exact for `Ir` (which is never zero on a cost line) and
                # wrong for every other event -- measured against the files'
                # own `totals:` line at -2.7% for SF's Bc, -49% for its Bi, and
                # 100x smaller on ARCHON, i.e. biased in the direction of the
                # conclusion. D33 reviewer C, and `--audit` now checks it.
                if len(parts) < 2:
                    continue
                try:
                    cost = int(parts[1 + event]) if 1 + event < len(parts) else 0
                except ValueError:
                    cost = 0
                if pending_calls:
                    calls_by_callee[cur_cfn] += pending_calls
                    calls_by_edge[(cur_fn, cur_cfn)] += pending_calls
                    incl_by_edge[(cur_fn, cur_cfn)] += cost
                    pending_calls = 0
                else:
                    self_ir[cur_fn] += cost
                continue
    return self_ir, calls_by_callee, calls_by_edge, incl_by_edge


def totals(path):
    """The file's own `totals:` line -- the ground truth every parse must match.

    Study 17's headline rule is that a bucketing which does not sum to the
    engine's total cannot be audited. The same applies one level down: a PARSER
    that does not sum to the file's own total cannot be audited either, and this
    one was wrong for every event except `Ir` until D33 reviewer C summed it.
    """
    for line in open(path):
        if line.startswith("totals:"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    return None


def demangle(n):
    n = re.sub(r"\[.*?\]$", "", n).strip()
    return n


def main():
    path = sys.argv[1]
    nodes = float(sys.argv[2])
    top = 40
    pat = None
    callers_of = None
    incl_of = None
    event = 0
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--top":
            top = int(args[i + 1]); i += 2
        elif args[i] == "--grep":
            pat = re.compile(args[i + 1]); i += 2
        elif args[i] == "--callers":
            callers_of = args[i + 1]; i += 2
        elif args[i] == "--incl":
            incl_of = args[i + 1]; i += 2
        elif args[i] == "--event":
            event = int(args[i + 1]); i += 2
        else:
            i += 1

    self_ir, calls, edges, incl = parse(path, event)

    # ALWAYS audit against the file's own totals line -- one assertion that
    # would have caught the `--event` bug on its first run (D33 reviewer C).
    tt = totals(path)
    if tt is not None and event < len(tt):
        got = sum(self_ir.values())
        if got != tt[event]:
            print(f"# PARSE AUDIT FAILED: self-cost sum {got:,} != totals[{event}] "
                  f"{tt[event]:,}  ({100.0 * (got - tt[event]) / tt[event]:+.3f}%)",
                  file=sys.stderr)
            sys.exit(2)

    if incl_of:
        # Exact inclusive cost of a call tree: sum every incoming edge.  This is
        # the only inlining-immune way to compare two engines whose compilers
        # made different inlining decisions -- self costs move between symbols,
        # the inclusive cost of the entry point does not.
        cre = re.compile(incl_of)
        agg = defaultdict(lambda: [0, 0])
        for (c, e), n in edges.items():
            if e and cre.search(e):
                agg[e][0] += n
                agg[e][1] += incl[(c, e)]
        for e, (n, ii) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            print(f"calls/node={n/nodes:8.4f}  INCL_i/n={ii/nodes:9.1f}  "
                  f"incl_i/call={ii/n if n else 0:8.1f}  {demangle(e)}")
        return

    if callers_of:
        cre = re.compile(callers_of)
        rows = [(c, e, n, incl[(c, e)]) for (c, e), n in edges.items()
                if e and cre.search(e)]
        rows.sort(key=lambda r: -r[3])
        print(f"{'calls/node':>11} {'incl i/n':>10}  caller -> callee")
        for c, e, n, ii in rows[:top]:
            print(f"{n/nodes:11.4f} {ii/nodes:10.1f}  {demangle(c)}  ->  {demangle(e)}")
        return

    rows = [(fn, ir) for fn, ir in self_ir.items() if fn]
    if pat:
        rows = [r for r in rows if pat.search(r[0])]
    rows.sort(key=lambda r: -r[1])
    tot = sum(ir for _, ir in self_ir.items())
    print(f"# nodes={nodes:,.0f}  total_Ir={tot:,}  total_i/n={tot/nodes:.1f}")
    print(f"{'self/node':>10} {'calls/node':>11} {'per_call':>9}  function")
    for fn, ir in rows[:top]:
        c = calls.get(fn, 0)
        ipc = ir / c if c else float("nan")
        print(f"{ir/nodes:10.2f} {c/nodes:11.4f} {ipc:9.1f}  {demangle(fn)}")


if __name__ == "__main__":
    main()
