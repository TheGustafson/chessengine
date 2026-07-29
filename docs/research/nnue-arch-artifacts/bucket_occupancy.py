#!/usr/bin/env python3
"""Output-bucket occupancy of ARCHON's real position distributions.

Prices the owner's "queens / no-queens routing" idea and the more general
"more output buckets" question, both of which are FREE at inference time
(only one bucket's weights are touched per eval).

Inputs: any file of FENs (one per line) or a .json list/dict keyed by FEN.
"""
import json, sys, collections

LAYOUT = [0]*9 + [1]*4 + [2]*4 + [3]*3 + [4]*3 + [5]*3 + [6]*3 + [7]*4  # sota_head.rs
assert len(LAYOUT) == 33

def fens(path):
    if path.endswith('.json'):
        d = json.load(open(path))
        return list(d.keys()) if isinstance(d, dict) else [x if isinstance(x,str) else x.get('fen') for x in d]
    return [l.split(';')[0].strip() for l in open(path) if l.strip() and not l.startswith('#')]

def feats(fen):
    b = fen.split()[0]
    pc = sum(1 for c in b if c.isalpha())
    q  = b.count('Q') + b.count('q')
    return pc, q

for path in sys.argv[1:]:
    try:
        F = [f for f in fens(path) if f and '/' in f]
    except Exception as e:
        print(f"{path}: SKIP ({e})"); continue
    tot = len(F)
    if not tot: continue
    bc = collections.Counter(); qb = collections.Counter(); nq = collections.Counter()
    for f in F:
        pc, q = feats(f)
        b = LAYOUT[min(pc,32)]
        bc[b] += 1
        qb[(b, min(q,2))] += 1
        nq[q] += 1
    print(f"\n{path}  n={tot}")
    print("  bucket : " + " ".join(f"{b:>6d}" for b in range(8)))
    print("  share %: " + " ".join(f"{100*bc[b]/tot:6.1f}" for b in range(8)))
    print("  queens on board: " + ", ".join(f"{k}Q {100*v/tot:.1f}%" for k,v in sorted(nq.items())))
    print("  P(has any queen | bucket):  " + " ".join(
        f"{100*(bc[b]-qb[(b,0)])/max(bc[b],1):6.1f}" for b in range(8)))
    # entropy / effective bucket count
    import math
    p = [bc[b]/tot for b in range(8) if bc[b]]
    H = -sum(x*math.log(x) for x in p)
    print(f"  effective buckets used = exp(H) = {math.exp(H):.2f} / 8")
