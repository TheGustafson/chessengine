#!/usr/bin/env python3
"""perf study 47 -- STATIC opcode census of a symbol, split by operand width.

    python3 tools/perf47/census47.py <binary> <symbol-substring> [...]

objdump the whole .text once, slice out each requested symbol by its `<name>:`
header, and histogram the mnemonics.  Width is read off the REGISTER operands
(zmm/ymm/xmm) rather than guessed from the mnemonic, because the same mnemonic
appears at three widths and only the 512-bit ones cost a full issue slot on
this box (study 46 s1.2).
"""
import re, subprocess, sys, collections

binary = sys.argv[1]
wanted = sys.argv[2:]
dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", binary],
                     capture_output=True, text=True).stdout.splitlines()

# split into symbols
cur, syms = None, collections.defaultdict(list)
hdr = re.compile(r"^[0-9a-f]+ <(.*)>:$")
for line in dis:
    m = hdr.match(line)
    if m:
        cur = m.group(1)
        continue
    if cur is not None:
        syms[cur].append(line)

def width(ops):
    if "zmm" in ops: return 512
    if "ymm" in ops: return 256
    if "xmm" in ops: return 128
    return 0

for w in wanted:
    hits = [s for s in syms if w in s]
    if not hits:
        print(f"!! no symbol matching {w!r}"); continue
    for s in sorted(hits, key=lambda s: -len(syms[s]))[:1]:
        hist = collections.Counter()
        total = 0
        for line in syms[s]:
            m = re.match(r"^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$", line)
            if not m: continue
            mn, ops = m.group(1), m.group(2)
            ops = ops.split("#")[0]
            total += 1
            hist[(width(ops), mn)] += 1
        n512 = sum(v for (wd, _), v in hist.items() if wd == 512)
        print(f"\n=== {s[:90]}   ({total} insns, {n512} are 512-bit)")
        for (wd, mn), v in sorted(hist.items(), key=lambda kv: (-kv[0][0], -kv[1])):
            tag = {512: "zmm", 256: "ymm", 128: "xmm", 0: "gpr"}[wd]
            print(f"  {tag:>3}  {mn:<16} {v:5d}")
