#!/usr/bin/env python3
"""perf study 67 — attribute a symbol's 64-byte code lines to SOURCE, and rank
them by sampled ops.

  python3 tools/perf67/srcmap67.py <ibsops-dump> <symbol-substring> \
      <dbg-binary> <sym-vaddr-in-dbg> <size> [topN]

The dump is taken on the plain PGO binary; the addresses are resolved in the
line-tables PGO binary, whose instruction stream this study proves identical
(same symbol size, same 9280 mnemonics). `addr2line -i` expands PGO's inline
frames, which is what turns "+0x9000 is cold" into "that is the history burst".
"""
import collections
import re
import subprocess
import sys


def _hit(sym, want):
    """EXACT symbol match unless `want` ends in '*'.

    A substring test here is a bug with teeth: 'search::<...::NonPv>' is a
    substring of 'qsearch::<...::NonPv>', so a substring match folded 18,451
    qsearch samples into the search symbol's histogram and inflated every
    within-symbol share this study quotes. Caught and corrected mid-study.
    """
    return sym.startswith(want[:-1]) if want.endswith("*") else sym == want

dump, want, dbg = sys.argv[1], sys.argv[2], sys.argv[3]
vaddr, size = int(sys.argv[4], 0), int(sys.argv[5], 0)
top = int(sys.argv[6]) if len(sys.argv) > 6 else 45

line_ct = collections.Counter()
total = 0
pat = re.compile(r"^\s*([0-9a-f]+)\s+(.*)\+0x([0-9a-f]+)\s*$")
for ln in open(dump):
    m = pat.match(ln.rstrip())
    if not m:
        continue
    total += 1
    if _hit(m.group(2), want):
        line_ct[int(m.group(3), 16) // 64] += 1
n = sum(line_ct.values())
nlines = (size + 63) // 64

addrs = "\n".join(hex(vaddr + i * 64) for i in range(nlines))
out = subprocess.run(["addr2line", "-e", dbg, "-f", "-C", "-i", "-a"],
                     input=addrs, capture_output=True, text=True).stdout
frames = {}
cur = None
pend = []
for ln in out.splitlines():
    if ln.startswith("0x"):
        if cur is not None:
            frames[cur] = pend
        cur = int(ln, 16)
        pend = []
    else:
        pend.append(ln)
if cur is not None:
    frames[cur] = pend

def label(i):
    f = frames.get(vaddr + i * 64, [])
    # frames alternate fn, file:line; innermost first. take innermost fn + its line
    if len(f) >= 2:
        fn = f[0].split("::{closure")[0]
        loc = f[1].split("/")[-1]
        outer = ""
        if len(f) >= 4:
            outer = " <- " + f[-2].split("/")[-1]
        return f"{loc:<28} {fn[:70]}{outer}"
    return "?"

print(f"{n} samples in symbol ({100*n/total:.2f}% of engine ops), {nlines} lines\n")
print("=== HOTTEST 64-BYTE CODE LINES")
print(f"{'%sym':>6} {'off':>7}  source")
for i, c in line_ct.most_common(top):
    print(f"{100*c/n:>5.2f}% +0x{i*64:05x}  {label(i)}")

print("\n=== SOURCE-LINE ROLLUP (ops by innermost file:line)")
roll = collections.Counter()
bytes_by = collections.Counter()
for i in range(nlines):
    f = frames.get(vaddr + i * 64, [])
    key = f[1].split("/")[-1] if len(f) >= 2 else "?"
    roll[key] += line_ct[i]
    bytes_by[key] += 64
print(f"{'%sym':>6} {'bytes':>7}  file:line")
for k, c in roll.most_common(top):
    print(f"{100*c/n:>5.2f}% {bytes_by[k]:>7}  {k}")

print("\n=== NEVER-SAMPLED RUNS >= 2 LINES, with source")
runs, start = [], None
for i in range(nlines + 1):
    z = i < nlines and line_ct[i] == 0
    if z and start is None:
        start = i
    elif not z and start is not None:
        runs.append((start, i - start))
        start = None
for s, l in sorted(runs, key=lambda r: -r[1]):
    if l < 2:
        continue
    print(f"  +0x{s*64:05x} {l*64:5d} B  {label(s)}")
