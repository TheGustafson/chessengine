#!/usr/bin/env python3
"""perf study 67 — aggregate an ibsops67 dump.

  python3 tools/perf67/ibsagg67.py <dump> [symbol-substring] [topN]

With no symbol: per-symbol op shares. With a symbol: the hottest OFFSETS
inside it (no skid -- IBS records the sampled op's own rip).
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

dump = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else None
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40

sym_ct = collections.Counter()
off_ct = collections.Counter()
total = 0
pat = re.compile(r"^\s*([0-9a-f]+)\s+(.*)\+0x([0-9a-f]+)\s*$")
for line in open(dump):
    m = pat.match(line.rstrip())
    if not m:
        continue
    total += 1
    sym = m.group(2)
    sym_ct[sym] += 1
    if want and want in sym:
        off_ct[int(m.group(3), 16)] += 1

print(f"{total} samples")
if not want:
    print(f"\n{'share':>8}  {'samples':>8}  symbol")
    for s, c in sym_ct.most_common(top):
        print(f"{100*c/total:>7.2f}%  {c:>8}  {s}")
else:
    n = sum(off_ct.values())
    print(f"\n{want}: {n} samples = {100*n/total:.2f}% of engine ops")
    print(f"{'%sym':>7} {'%eng':>7}  offset")
    for o, c in off_ct.most_common(top):
        print(f"{100*c/n:>6.2f}% {100*c/total:>6.2f}%  +0x{o:x}")
