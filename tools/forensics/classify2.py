"""Failure-mode classifier, v2 — probe B removes pruning THROUGHOUT the tree.

v1 used MultiPV, which only unhides moves at the root's immediate children. The
game-18 mechanism (RFP discarding a refutation one ply down, inside the losing
move's subtree) was invisible to it and misclassified as EVAL. Probe B is now
all five eval gates disabled, which is what actually recovered that line.
"""
import json,os,re,subprocess,statistics as st
S=os.environ["S"]; NET=os.getcwd()+"/data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
BIN="engines/bin/archon-d47v-tune-pgo"
OFF=("setoption name RAZOR_BASE value 30000\nsetoption name RFP_MULT_BASE value 30000\n"
     "setoption name RFP_MULT_CAP value 30000\nsetoption name PFUT_BASE value 30000\n"
     "setoption name CFUT_BASE value 30000\nsetoption name QS_FUTILITY_MARGIN value 30000\n")
def probe(fen,depth,extra="",wait=7):
    s=f"uci\nsetoption name EvalFile value {NET}\n{extra}isready\nposition fen {fen}\ngo depth {depth}\n"
    out=subprocess.run(["bash","-c",f'{{ printf %s "$SC"; sleep {wait}; }} | timeout {wait+30} {BIN}'],
        capture_output=True,text=True,env={**os.environ,"SC":s}).stdout
    v=m=None
    for line in out.splitlines():
        if line.startswith(f"info depth {depth} "):
            g=re.search(r"score cp (-?\d+).* pv (\S+)",line)
            if g: v=int(g.group(1)); m=g.group(2)
    return v,m
d=json.load(open(f"{S}/blunders_wp.json"))
recs=[r for r in d["archon"] if r["class"] in ("search_miss","eval_instability","inaccuracy")]
rows=[]
for r in recs:
    dep=max(8,min(int(r["depth"]),16))
    a,am=probe(r["fen"],dep)
    b,bm=probe(r["fen"],dep,OFF)
    if a is None or b is None: continue
    moved = (am!=bm)
    kind="PRUNING" if (moved and b-a>=50) else ("PRUNING(move)" if moved else "not-pruning")
    rows.append((r["game"],dep,a,am,b,bm,kind))
    print(f"  g{r['game']:<4} d{dep:<3} normal {a:>6} {am:<6} gates-off {b:>6} {bm:<6}  {kind}",flush=True)
from collections import Counter
print(f"\n{len(rows)} errors: {dict(Counter(x[6] for x in rows))}")
