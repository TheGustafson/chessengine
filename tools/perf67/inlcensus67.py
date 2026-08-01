#!/usr/bin/env python3
"""perf study 67 — WHAT IS INLINED INTO A SYMBOL, in bytes and in sampled ops.

  python3 tools/perf67/inlcensus67.py <dbg-binary> <sym-vaddr> <size> \
      <ibsops-dump> <symbol-substring> <base-vaddr-in-dump-binary>

PGO fuses dozens of callees into `search::<NonPv>`; `nm` reports one 45 KB
symbol and every previous profile in this project reported it as one blob.
Resolving EVERY instruction address through `addr2line -i` and rolling up by
the OUTERMOST INLINED CALLEE (the frame directly under the symbol itself)
gives the byte budget of the function -- which is what an instruction-cache
argument needs and an `instructions:u` share cannot provide.
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

dbg, vaddr, size = sys.argv[1], int(sys.argv[2], 0), int(sys.argv[3], 0)
dump, want, base = sys.argv[4], sys.argv[5], int(sys.argv[6], 0)

# 1. instruction addresses + sizes from objdump
od = subprocess.run(["objdump", "-d", "--start-address", hex(vaddr),
                     "--stop-address", hex(vaddr + size), dbg],
                    capture_output=True, text=True).stdout
insns = []
for ln in od.splitlines():
    m = re.match(r"^\s+([0-9a-f]+):\t([0-9a-f ]+)\t", ln)
    if m:
        insns.append((int(m.group(1), 16), len(m.group(2).split())))
print(f"{len(insns)} instructions, {sum(s for _, s in insns)} bytes", file=sys.stderr)

# 2. inline stacks
a2l = subprocess.run(["addr2line", "-e", dbg, "-f", "-C", "-i", "-a"],
                     input="\n".join(hex(a) for a, _ in insns),
                     capture_output=True, text=True).stdout
stacks, cur, pend = {}, None, []
for ln in a2l.splitlines():
    if ln.startswith("0x"):
        if cur is not None:
            stacks[cur] = pend
        cur, pend = int(ln, 16), []
    else:
        pend.append(ln)
if cur is not None:
    stacks[cur] = pend

# 3. samples per address
samp = collections.Counter()
tot = 0
pat = re.compile(r"^\s*([0-9a-f]+)\s+(.*)\+0x([0-9a-f]+)\s*$")
for ln in open(dump):
    m = pat.match(ln.rstrip())
    if not m:
        continue
    tot += 1
    if _hit(m.group(2), want):
        samp[vaddr + int(m.group(3), 16)] += 1
nsym = sum(samp.values())

def outer_callee(st):
    """st = [fn0, loc0, fn1, loc1, ...] innermost first; last pair is the symbol."""
    fns = st[0::2]
    locs = st[1::2]
    if len(fns) <= 1:
        return "(search's own code)", locs[0].split("/")[-1] if locs else "?"
    return fns[-2].split("::{closure")[0], locs[-2].split("/")[-1]

by = collections.Counter()
byops = collections.Counter()
for a, sz in insns:
    fn, _ = outer_callee(stacks.get(a, []))
    by[fn] += sz
    byops[fn] += samp[a]

print(f"\nsymbol {size} B, {nsym} sampled ops ({100*nsym/tot:.2f}% of engine)")
print(f"{'bytes':>7} {'%B':>6} {'%ops':>6}  outermost inlined callee")
for fn, b in by.most_common(40):
    print(f"{b:>7} {100*b/size:>5.1f}% {100*byops[fn]/nsym:>5.1f}%  {fn[:96]}")
