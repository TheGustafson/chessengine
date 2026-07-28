#!/usr/bin/env python3
"""Parse pstat7 rounds -> per-node rates + per-round ratios."""
import re, glob, os, statistics, sys

S = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else 'g1'


def nodes_of(path):
    txt = open(path).read()
    m = re.findall(r'Nodes searched\s*:\s*(\d+)', txt)
    n = int(m[-1]) if m else None
    m2 = re.findall(r'Nodes/second\s*:\s*(\d+)', txt)
    nps = int(m2[-1]) if m2 else None
    return n, nps


def counters_of(path):
    d = {}
    for line in open(path):
        parts = line.strip().split(',')
        if len(parts) >= 3 and parts[0].replace('.', '').isdigit():
            try:
                d[parts[2]] = int(parts[0])
            except ValueError:
                pass
        # wall time from 'duration_time'? -x, gives ns rows named duration_time on some versions
    m = re.search(r'^([\d.]+),,duration_time', open(path).read(), re.M)
    return d


engines = ['A', 'S18', 'S16']
rounds = sorted(set(int(re.search(r'-r(\d+)\.csv', f).group(1))
                    for f in glob.glob(f'{S}/runs/ps-{tag}-A-r*.csv')))
per = {e: {} for e in engines}
for e in engines:
    for r in rounds:
        c = counters_of(f'{S}/runs/ps-{tag}-{e}-r{r}.csv')
        n, nps = nodes_of(f'{S}/runs/out-{tag}-{e}-r{r}.txt')
        per[e][r] = (c, n, nps)

# events present
ev = sorted({k for e in engines for r in rounds for k in per[e][r][0]})
print(f"tag={tag} rounds={rounds}")
for e in engines:
    ns = [per[e][r][1] for r in rounds]
    print(f"== {e}: nodes {ns[0]} (spread {min(ns)}..{max(ns)})")
    for k in ev:
        vals = [per[e][r][0].get(k, 0) / per[e][r][1] for r in rounds if per[e][r][0].get(k)]
        if not vals:
            continue
        med = statistics.median(vals)
        print(f"   {k:<44} per-node median {med:12.2f}  (n={len(vals)}, "
              f"min {min(vals):.2f}, max {max(vals):.2f})")
    npss = [per[e][r][2] for r in rounds if per[e][r][2]]
    if npss:
        print(f"   NPS median {statistics.median(npss):.0f}  range {min(npss)}..{max(npss)}")

# per-round ratios A/S18 and A/S16 for each event
for other in ['S18', 'S16']:
    print(f"== ratios A/{other} (per round, then median)")
    for k in ev:
        rr = []
        for r in rounds:
            a = per['A'][r][0].get(k); b = per[other][r][0].get(k)
            an = per['A'][r][1]; bn = per[other][r][1]
            if a and b:
                rr.append((a / an) / (b / bn))
        if rr:
            print(f"   {k:<44} median {statistics.median(rr):.3f}  "
                  f"({', '.join(f'{x:.3f}' for x in rr)})")
    rr = []
    for r in rounds:
        a = per['A'][r][2]; b = per[other][r][2]
        if a and b:
            rr.append(a / b)
    if rr:
        print(f"   NPS ratio A/{other}: median {statistics.median(rr):.3f}  "
              f"({', '.join(f'{x:.3f}' for x in rr)})")
