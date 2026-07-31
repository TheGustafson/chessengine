#!/usr/bin/env python3
"""perf study 59 — FULL inline stack census inside one symbol, by sampled IBS ops.

`tools/perf52/ibsline52.py` prints the innermost frame and the outermost frame.
Inside a symbol that PGO has flattened completely (`at_full` is 99.84% "its own
phase"), both ends are uninformative: the innermost says `copy_nonoverlapping`
and the outermost says the symbol. What is missing is the frame in BETWEEN —
which caller asked for that copy.

    python3 tools/perf59/stack59.py <ibs-script.txt> <binary> <nm-regex> <perf-suffix> [depth]

Prints, for each innermost frame, its top calling stacks with sampled-op shares.
"""
import collections
import re
import subprocess
import sys

TAIL = re.compile(r"BLK\s+\S+\s+(\d+)\s+([0-9a-f]+)\s+(\S+)\+(0x[0-9a-f]+)")


def sym_range(binary, pat):
    out = subprocess.run(["nm", "--print-size", binary],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split()
        if len(f) == 4 and re.search(pat, f[3]):
            return f[3], int(f[0], 16), int(f[1], 16)
    raise SystemExit(f"no sized symbol matching /{pat}/")


def main():
    script, binary, pat, disp = sys.argv[1:5]
    depth = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    _sym, lo, size = sym_range(binary, pat)

    ops = collections.Counter()
    total = 0
    for line in open(script):
        m = TAIL.search(line)
        if not m or not m.group(3).endswith(disp):
            continue
        off = int(m.group(4), 16)
        if off < size:
            ops[off] += 1
            total += 1

    uniq = sorted(ops)
    out = subprocess.run(
        ["addr2line", "-i", "-f", "-p", "-e", binary] + [f"0x{lo + o:x}" for o in uniq],
        capture_output=True, text=True).stdout.splitlines()
    stacks, cur = [], []
    for ln in out:
        if ln.startswith(" (inlined by) "):
            cur.append(ln[len(" (inlined by) "):])
        else:
            if cur:
                stacks.append(cur)
            cur = [ln]
    if cur:
        stacks.append(cur)

    def clean(fr):
        fr = re.sub(r"::h[0-9a-f]{16}", "", fr)
        fr = fr.replace("chessengine::", "").replace("core::", "")
        fn = fr.split(" at ")[0]
        loc = fr.split(" at ")[-1].split("/")[-1] if " at " in fr else ""
        fn = re.sub(r"<.*?>", "", fn)
        return f"{fn.strip()} @{loc}"

    agg = collections.Counter()
    for off, st in zip(uniq, stacks):
        key = " <- ".join(clean(f) for f in st[:depth])
        agg[key] += ops[off]

    print(f"symbol {disp}   sampled ops {total:,}")
    print(f"{'ops%':>7}  innermost <- caller chain (depth {depth})")
    for k, v in agg.most_common(30):
        print(f"{100.0 * v / max(total, 1):>7.2f}  {k}")


main()
