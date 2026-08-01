#!/usr/bin/env python3
"""perf study 67 — the CODE COLDNESS MAP of one symbol.

  python3 tools/perf67/coldmap67.py <ibsops-dump> <symbol-substring> <size-bytes>

IBS with `cnt_ctl=0` samples DISPATCHED OPS with no skid, so a 64-byte line
that collects zero samples over a whole-frame run is a line the frame never
(or almost never) executes -- yet it still occupies L1i, because the fetch
granularity is the line. Prints per-line counts in ADDRESS order plus the
contiguous zero runs, which are the outlining candidates.
"""
import collections
import re
import sys


def _hit(sym, want):
    """EXACT symbol match unless `want` ends in '*'.

    A substring test here is a bug with teeth: 'search::<...::NonPv>' is a
    substring of 'qsearch::<...::NonPv>', so a substring match folded 18,451
    qsearch samples into the search symbol's histogram and inflated every
    within-symbol share this study quotes. Caught and corrected mid-study.
    """
    return sym.startswith(want[:-1]) if want.endswith("*") else sym == want

dump, want, size = sys.argv[1], sys.argv[2], int(sys.argv[3], 0)
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
cold = [i for i in range(nlines) if line_ct[i] == 0]
warm = [i for i in range(nlines) if 0 < line_ct[i] <= max(1, n // (nlines * 40))]
print(f"symbol ~{size} B = {nlines} lines; {n} samples ({100*n/total:.2f}% of engine ops)")
print(f"NEVER-SAMPLED lines: {len(cold)} = {100*len(cold)/nlines:.1f}%  ({64*len(cold)} B)")
print(f"<=1/40-of-average lines: {len(warm)} = {100*len(warm)/nlines:.1f}%  ({64*len(warm)} B)")

runs, start = [], None
for i in range(nlines + 1):
    z = i < nlines and line_ct[i] == 0
    if z and start is None:
        start = i
    elif not z and start is not None:
        runs.append((start, i - start))
        start = None
runs.sort(key=lambda r: -r[1])
print(f"\ncontiguous never-sampled runs (top 25), offset .. length:")
for s, l in runs[:25]:
    print(f"  +0x{s*64:05x} .. +0x{(s+l)*64:05x}   {l:3d} lines = {l*64:5d} B")
tot_run = sum(l for _, l in runs if l >= 2)
print(f"\nruns of >=2 lines total {tot_run} lines = {tot_run*64} B "
      f"({100*tot_run/nlines:.1f}% of the symbol)")

# hot mass concentration
vals = sorted(line_ct.values(), reverse=True)
acc = 0
for k, v in enumerate(vals, 1):
    acc += v
    if acc >= 0.90 * n:
        print(f"\n90% of this symbol's ops live in {k} lines = {k*64} B "
              f"({100*k/nlines:.1f}% of it)")
        break
