#!/usr/bin/env python3
"""perf study 69 — the mispredict profile rolled up by SOURCE LINE.

  python3 tools/perf69/misroll69.py <brstacksym.txt> <dbg-binary> \
      <taken-branches-per-node> <taken-mispredicts-per-node> [top]

`missrc69.py` ranks machine sites; PGO duplicates one source line into four
monomorphisations, so the site view understates a shared line by up to 4x. This
sums every site that resolves to the same innermost `file:line` -- the view that
says which SOURCE CONSTRUCT to edit.

Same LBR caveat as `missrc69.py`: only TAKEN branches appear.
"""
import collections
import re
import subprocess
import sys

dump, dbg = sys.argv[1], sys.argv[2]
tkn_pn, misp_pn = float(sys.argv[3]), float(sys.argv[4])
top = int(sys.argv[5]) if len(sys.argv) > 5 else 30

seen = collections.Counter()
miss = collections.Counter()
tot = tmiss = 0
for line in open(dump, errors="replace"):
    for ent in line.split():
        f = ent.split("/")
        if len(f) < 3:
            continue
        tot += 1
        m = f[2] == "M"
        tmiss += m
        seen[f[0]] += 1
        miss[f[0]] += m

nm = subprocess.run(["nm", "-SC", dbg], capture_output=True, text=True).stdout
sym_va = {}
for ln in nm.splitlines():
    p = ln.split(None, 3)
    if len(p) == 4 and p[2].lower() in "tdw":
        sym_va.setdefault(p[3], int(p[0], 16))

pat = re.compile(r"^(.*)\+0x([0-9a-f]+)$")
addr_of = {}
for s in seen:
    m = pat.match(s)
    if m and m.group(1) in sym_va:
        addr_of[s] = sym_va[m.group(1)] + int(m.group(2), 16)

res = subprocess.run(
    ["addr2line", "-e", dbg, "-f", "-C", "-i", "-a"],
    input="\n".join(hex(a) for a in addr_of.values()),
    capture_output=True,
    text=True,
).stdout
frames, cur, pend = {}, None, []
for ln in res.splitlines():
    if ln.startswith("0x"):
        if cur is not None:
            frames[cur] = pend
        cur, pend = int(ln, 16), []
    else:
        pend.append(ln)
if cur is not None:
    frames[cur] = pend

roll_m = collections.Counter()
roll_s = collections.Counter()
roll_sites = collections.defaultdict(set)
for s, a in addr_of.items():
    f = frames.get(a, [])
    key = f[1].split("/")[-1] if len(f) >= 2 else "?"
    if len(f) >= 4:
        key += "  <-  " + f[3].split("/")[-1]
    roll_m[key] += miss[s]
    roll_s[key] += seen[s]
    roll_sites[key].add(s.rsplit("+0x", 1)[0])

sc_s, sc_m = tkn_pn / tot, misp_pn / max(tmiss, 1)
print(f"# {tot} LBR entries, {tmiss} mispredicted; taken {tkn_pn}/node, "
      f"taken-misp {misp_pn}/node")
print(f"{'misp/node':>10}{'%tknmis':>9}{'exec/node':>11}{'rate':>7}{'sym':>5}  source line (innermost <- caller)")
for k, mc in roll_m.most_common(top):
    print(f"{mc*sc_m:>10.4f}{100.0*mc/max(tmiss,1):>8.2f}%{roll_s[k]*sc_s:>11.4f}"
          f"{100.0*mc/max(roll_s[k],1):>6.1f}%{len(roll_sites[k]):>5}  {k}")
