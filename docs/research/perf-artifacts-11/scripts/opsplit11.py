#!/usr/bin/env python3
"""perf study 11 — opcode-class histogram of a `perf annotate --stdio` dump.

    python3 opsplit11.py <annot.txt> <symbol-share-percent> <engine-i-per-node>

Study 10's `sfsplit10.sh` established the method: each stage of an NNUE eval
has a SIGNATURE opcode, so an opcode histogram splits a symbol that the
compiler has inlined into one blob.  This version reports absolute i/n for
every opcode instead of grepping one at a time, so both engines can be laid
side by side.

`vpmovsxbw` is the i8->i16 widening.  In BOTH engines the FullThreats
transformer is the only i8 weight table, so it executes if and only if a
threat accumulator row is being folded — and since perf study 11 verified
that ARCHON and SF eca43a97 carry the SAME FullThreats feature set at the
SAME 60720 dimensions, the count is directly comparable as
`rows_folded x lanes/32`.
"""

import re
import sys
from collections import defaultdict

path = sys.argv[1]
sym_share = float(sys.argv[2])  # symbol's % of the whole engine
per_node = float(sys.argv[3])  # engine's total instructions/node

ROW = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-f]+):\s+(\S+)(.*)$")
hist = defaultdict(float)
n = 0
for line in open(path):
    m = ROW.match(line)
    if not m:
        continue
    n += 1
    hist[m.group(3)] += float(m.group(1))

total = sum(hist.values())
if total <= 0:
    print("no samples in annotation")
    sys.exit(1)

# The annotate percentages are LOCAL to the symbol; scale to engine i/n.
scale = sym_share / 100.0 * per_node / total

SIMD = ("v", "p")  # vector-ish opcode prefixes


def cls(op):
    if op.startswith("vpmovsxbw"):
        return "THREAT-FOLD (i8->i16 widen)"
    if op.startswith(("vpdpbusd", "vpmaddubsw", "vpmulhrsw")):
        return "MATMUL/head"
    if op.startswith(("vpaddw", "vpsubw", "vpaddd", "vpsubd")):
        return "ACC add/sub"
    if op.startswith(("vmovdqu", "vmovdqa", "vmovup", "vmovap")):
        return "vector load/store"
    if op.startswith(("vpmax", "vpmin", "vpackus", "vpsra", "vpsrl", "vpsll")):
        return "clamp/shift"
    if op.startswith("v"):
        return "other SIMD"
    return "scalar"


print(f"{n} annotated instructions; symbol = {sym_share:.2f}% of engine "
      f"= {sym_share / 100.0 * per_node:.1f} i/n")
print()
print(f"{'opcode':<20}{'local %':>9}{'i/n':>9}")
for op, pct in sorted(hist.items(), key=lambda kv: -kv[1])[:22]:
    print(f"{op:<20}{pct * 100 / total:>8.2f}%{pct * scale:>9.1f}")

print()
byclass = defaultdict(float)
for op, pct in hist.items():
    byclass[cls(op)] += pct
print(f"{'class':<32}{'local %':>9}{'i/n':>9}")
for c, pct in sorted(byclass.items(), key=lambda kv: -kv[1]):
    print(f"{c:<32}{pct * 100 / total:>8.2f}%{pct * scale:>9.1f}")

simd = sum(v for k, v in byclass.items() if k != "scalar")
print()
print(f"SIMD share {simd * 100 / total:5.1f}%   scalar share "
      f"{byclass['scalar'] * 100 / total:5.1f}%")
