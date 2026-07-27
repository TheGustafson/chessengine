#!/usr/bin/env python3
"""study-7 asm mix: sym+symoff histogram x objdump(-C) ->
opcode / register-width / load-store / region mix, dynamic + static.
usage: opc7.py <perf.data> <binary> <sym-regex> [brief]"""
import subprocess, sys, re, collections, bisect, os, pickle

data, binary, pat = sys.argv[1], sys.argv[2], sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else 'all'
hist = collections.Counter(); tot = 0; sel = 0
p = subprocess.run(['perf', 'script', '-i', data, '-F', 'ip,sym,symoff'],
                   capture_output=True, text=True)
for line in p.stdout.splitlines():
    line = line.strip()
    if not line or line == '[unknown]':
        continue
    m = re.match(r'^\s*[0-9a-f]+\s+(.*)\+0x([0-9a-f]+)$', line)
    if not m:
        continue
    sym, off = m.group(1), int(m.group(2), 16)
    tot += 1
    if re.search(pat, sym):
        hist[off] += c if (c := 1) else 1
        sel += 1
print(f"# samples total {tot}  selected {sel} ({100.0*sel/max(tot,1):.2f}%)")

cache = binary.replace('/', '_') + '.objdump.pkl'
cpath = os.path.join(os.path.dirname(os.path.abspath(data)), cache)
if os.path.exists(cpath):
    funcs = pickle.load(open(cpath, 'rb'))
else:
    dp = subprocess.run(['objdump', '-d', '-C', '--no-show-raw-insn', binary],
                        capture_output=True, text=True)
    funcs = {}; cur = None; base = 0
    for line in dp.stdout.splitlines():
        m = re.match(r'^([0-9a-f]+) <(.*)>:$', line)
        if m:
            cur = m.group(2); base = int(m.group(1), 16); funcs[cur] = []
            continue
        m = re.match(r'^\s+([0-9a-f]+):\t(\S+)\s*(.*)$', line)
        if m and cur is not None:
            funcs[cur].append((int(m.group(1), 16) - base, m.group(2), m.group(3)))
    pickle.dump(funcs, open(cpath, 'wb'))

cands = [k for k in funcs if re.search(pat, k)]
if not cands:
    print("!! no objdump function matched"); sys.exit(1)
cands.sort(key=lambda k: -len(funcs[k]))
fn = cands[0]
print(f"# objdump fn: {fn[:110]}  ({len(funcs[fn])} static instrs, {len(cands)} candidates)")


def memclass(mn, op):
    """load / store / both / none — AT&T syntax heuristic (src...,dst)."""
    if mn.startswith(('j', 'call', 'ret', 'nop', 'lea')):
        return 'none'
    parts = [x for x in re.split(r',(?![^(]*\))', op) if x]
    if not parts:
        return 'none'
    dst_mem = '(' in parts[-1] or re.match(r'^-?0x[0-9a-f]+$', parts[-1].strip() or 'x')
    src_mem = any('(' in x for x in parts[:-1])
    if mn.startswith(('cmp', 'test', 'push')):  # dst not written
        return 'load' if ('(' in op) else 'none'
    if dst_mem and '(' in parts[-1]:
        if src_mem or not mn.startswith(('mov', 'vmov')):
            # RMW on memory counts as load+store; plain mov-to-mem is store
            return 'both' if not mn.startswith(('mov', 'vmov')) else 'store'
        return 'store'
    return 'load' if src_mem else 'none'


opc = collections.Counter(); width = collections.Counter()
regions = collections.Counter(); statw = collections.Counter()
dynls = collections.Counter(); statls = collections.Counter()
offs = [x[0] for x in funcs[fn]]
miss = 0
for off, c in hist.items():
    i = bisect.bisect_right(offs, off) - 1
    if i < 0 or off - offs[i] > 15:
        miss += c; continue
    _, mn, op = funcs[fn][i]
    opc[mn] += c
    w = 'zmm' if '%zmm' in op else 'ymm' if '%ymm' in op else 'xmm' if '%xmm' in op else 'gpr'
    width[w] += c
    dynls[memclass(mn, op)] += c
    regions[off // 512 * 512] += c
for _, mn, op in funcs[fn]:
    w = 'zmm' if '%zmm' in op else 'ymm' if '%ymm' in op else 'xmm' if '%xmm' in op else 'gpr'
    statw[w] += 1
    statls[memclass(mn, op)] += 1
print(f"# unmapped {miss}")
print("## dynamic opcode mix (% of selected)")
for m, c in opc.most_common(18):
    print(f"   {m:<16} {100.0*c/max(sel,1):6.2f}")
print("## register width: dynamic | static")
for k in ['zmm', 'ymm', 'xmm', 'gpr']:
    print(f"   {k:<6} dyn {100.0*width[k]/max(sel,1):6.2f}   static {100.0*statw[k]/max(len(funcs[fn]),1):6.2f}")
n = max(len(funcs[fn]), 1)
print("## mem class: dynamic | static  (load/store/both/none)")
for k in ['load', 'store', 'both', 'none']:
    print(f"   {k:<6} dyn {100.0*dynls[k]/max(sel,1):6.2f}   static {100.0*statls[k]/n:6.2f}")
if mode == 'all':
    print("## 512B regions (top 10): off, %sel, dominant mnemonics")
    for a, c in regions.most_common(10):
        lo = bisect.bisect_left(offs, a); hi = bisect.bisect_left(offs, a + 512)
        sub = collections.Counter(x[1] for x in funcs[fn][lo:hi])
        print(f"   +{hex(a):<8} {100.0*c/max(sel,1):6.2f}  " + ", ".join(f"{k}:{v}" for k, v in sub.most_common(7)))
