#!/usr/bin/env python3
"""perf study 69 — THE BRANCH-MISPREDICT PROFILE, RESOLVED TO SOURCE.

  python3 tools/perf69/missrc69.py <brstacksym.txt> <dbg-binary> \
      <taken-branches-per-node> <taken-mispredicts-per-node> [top] [--sym S]

`mispagg69.py` gives `symbol+0xoff`; this resolves each site through the
line-tables PGO binary with `addr2line -i`, so the answer is a FILE:LINE and an
inline chain rather than an offset. The two per-node scalars come from
`perf stat -e ex_ret_brn_tkn:u,ex_ret_brn_tkn_misp:u` on the same frame and turn
sampled entry counts into absolute per-node rates.

⚠ AN LBR STACK HOLDS ONLY **TAKEN** BRANCHES. A conditional branch that is
never taken (an initialised `LazyLock` guard, a never-hit panic edge) has zero
entries here no matter how often it executes -- measured, not assumed: all ten
`attacks::TABLES` guard branches in `search::<NonPv>` have 0 of 418,976
entries. So this profile covers the `ex_ret_brn_tkn_misp` population (12.55 of
the 21.65 mispredicts per node on this frame); the other 42% are branches
predicted taken that fall through, and they are NOT in this histogram. Use
`opsat69.py` (IBS, dispatched ops) for anything never-taken.
"""
import collections
import re
import subprocess
import sys

dump, dbg = sys.argv[1], sys.argv[2]
tkn_pn, misp_pn = float(sys.argv[3]), float(sys.argv[4])
rest = [a for a in sys.argv[5:]]
want = None
if "--sym" in rest:
    want = rest[rest.index("--sym") + 1]
    rest = rest[: rest.index("--sym")]
top = int(rest[0]) if rest else 30

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
        if want and want not in f[0]:
            continue
        seen[f[0]] += 1
        miss[f[0]] += m

# symbol -> vaddr in the line-tables binary
nm = subprocess.run(["nm", "-SC", dbg], capture_output=True, text=True).stdout
sym_va = {}
for ln in nm.splitlines():
    p = ln.split(None, 3)
    if len(p) == 4 and p[2].lower() in "tdw":
        sym_va.setdefault(p[3], int(p[0], 16))

exec_per = tkn_pn / (tot / 1.0) * 1.0  # placeholder; real scale below
scale_seen = tkn_pn / tot * 1.0
scale_miss = misp_pn / max(tmiss, 1)

sites = miss.most_common(top)
addrs, keys = [], []
pat = re.compile(r"^(.*)\+0x([0-9a-f]+)$")
for s, _ in sites:
    m = pat.match(s)
    if m and m.group(1) in sym_va:
        addrs.append(hex(sym_va[m.group(1)] + int(m.group(2), 16)))
        keys.append(s)
    else:
        addrs.append(None)
        keys.append(s)
res = subprocess.run(
    ["addr2line", "-e", dbg, "-f", "-C", "-i", "-a"],
    input="\n".join(a for a in addrs if a),
    capture_output=True,
    text=True,
).stdout
# `addr2line -a` echoes the address ZERO-PADDED, so key the map by int value.
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

print(f"# {tot} LBR entries, {tmiss} mispredicted ({100*tmiss/tot:.2f}%)")
print(f"# scale: 1 entry = {scale_seen:.4f} taken-branch exec/node, "
      f"1 M-entry = {scale_miss:.5f} taken-mispredicts/node")
print(f"{'misp/node':>10}{'%tknmis':>9}{'exec/node':>11}{'rate':>7}  site / source")
for (s, mc), a in zip(sites, addrs):
    sc = seen[s]
    f = frames.get(int(a, 16) if a else -1, [])
    loc = ""
    if len(f) >= 2:
        chain = [f[i + 1].split("/")[-1] for i in range(0, len(f) - 1, 2)]
        loc = "  " + "  <-  ".join(chain[:3])
    print(f"{mc*scale_miss:>10.4f}{100.0*mc/max(sum(miss.values()),1):>8.2f}%"
          f"{sc*scale_seen:>11.4f}{100.0*mc/max(sc,1):>6.1f}%  {s}")
    if loc:
        print(f"{'':>38}{loc}")
