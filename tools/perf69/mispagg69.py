#!/usr/bin/env python3
"""perf study 69 — roll an LBR `brstacksym` dump up into a per-branch-SITE
mispredict histogram.

  python3 tools/perf69/mispagg69.py <brstacksym.txt> [top] [--sym SUBSTR]

Each LBR entry is `<from-sym+off>/<to-sym+off>/<M|P|->/...`. `from` is the
branch's own address, so the histogram is UNSKIDDED: no `branch-misses` sample
IP is involved anywhere.

Three numbers per site, and the third is the one that matters:

  seen   how often this branch appears anywhere in a sampled 16-deep stack
         (proportional to its EXECUTION rate)
  miss   how often it appears with the mispredict flag set
  rate   miss/seen — the branch's own mispredict probability

A site with a big `miss` share and a low `rate` is a hot correctly-predicted
branch that is simply executed a lot; a site with a high `rate` is a branch the
predictor cannot learn, and that is the only kind worth attacking.

Duplicate LBR entries inside ONE stack are counted once each (they are genuinely
separate executions of the same branch — a tight loop fills the stack with the
same site).
"""
import collections
import sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 40
want = None
if "--sym" in sys.argv:
    want = sys.argv[sys.argv.index("--sym") + 1]

seen = collections.Counter()
miss = collections.Counter()
tot_seen = tot_miss = 0
for line in open(path, errors="replace"):
    for ent in line.split():
        f = ent.split("/")
        if len(f) < 3:
            continue
        src, dst, pred = f[0], f[1], f[2]
        tot_seen += 1
        m = pred == "M"
        tot_miss += m
        if want and want not in src:
            continue
        seen[(src, dst)] += 1
        miss[(src, dst)] += m

# Roll (src,dst) pairs up by src: one branch instruction, possibly many targets
# (an indirect branch or a switch), and the SITE is what we can edit.
site_seen = collections.Counter()
site_miss = collections.Counter()
site_tgts = collections.defaultdict(set)
for (src, dst), c in seen.items():
    site_seen[src] += c
    site_miss[src] += miss[(src, dst)]
    site_tgts[src].add(dst)

sel_miss = sum(site_miss.values())
print(f"# entries {tot_seen}  mispredicted {tot_miss} ({100.0*tot_miss/max(tot_seen,1):.2f}%)")
if want:
    print(f"# filter '{want}': {sum(site_seen.values())} entries, {sel_miss} misses")
print(f"{'miss':>9}{'%ofmis':>8}{'seen':>10}{'rate':>8}{'tgt':>5}  site")
for src, mc in site_miss.most_common(top):
    sc = site_seen[src]
    print(
        f"{mc:>9}{100.0*mc/max(sel_miss,1):>7.2f}%{sc:>10}"
        f"{100.0*mc/max(sc,1):>7.1f}%{len(site_tgts[src]):>5}  {src}"
    )
