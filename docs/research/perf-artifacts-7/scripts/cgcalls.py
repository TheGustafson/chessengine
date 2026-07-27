#!/usr/bin/env python3
"""Sum call counts + inclusive-ish Ir per called function from a callgrind out file."""
import sys, re, collections

path = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else '.'
names = {}          # id -> name  (fn/cfn share the name table per file)
calls = collections.Counter()
ir_self = collections.Counter()
cur_cfn = None
cur_fn = None
pending_calls = 0
for line in open(path, errors='replace'):
    m = re.match(r'^c?fn=\((\d+)\)(?:\s+(.*))?$', line)
    if m:
        fid, nm = m.group(1), m.group(2)
        if nm:
            names[fid] = nm
        name = names.get(fid, f'?{fid}')
        if line.startswith('cfn='):
            cur_cfn = name
        else:
            cur_fn = name
        continue
    m = re.match(r'^calls=(\d+)', line)
    if m and cur_cfn is not None:
        calls[cur_cfn] += int(m.group(1))
        continue
    m = re.match(r'^[\d+*-]+ (\d+)', line)
    if m and cur_fn:
        ir_self[cur_fn] += int(m.group(1))

tot_ir = sum(ir_self.values())
print(f"# total self Ir {tot_ir:,}")
sel = [(n, c) for n, c in calls.items() if re.search(pat, n)]
sel.sort(key=lambda x: -x[1])
for n, c in sel[:40]:
    print(f"calls {c:>12,}  {n[:150]}")
print("## self-Ir top matches")
sel2 = [(n, c) for n, c in ir_self.items() if re.search(pat, n)]
sel2.sort(key=lambda x: -x[1])
for n, c in sel2[:40]:
    print(f"Ir    {c:>14,} ({100.0*c/tot_ir:5.2f}%)  {n[:145]}")
