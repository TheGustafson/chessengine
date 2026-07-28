"""Failure-mode classifier, v2 — probe B removes pruning THROUGHOUT the tree.

v1 used MultiPV, which only unhides moves at the root's immediate children. The
game-18 mechanism (a refutation discarded one ply down, inside the losing
move's subtree) was invisible to it and misclassified as EVAL. Probe B is now
all five eval gates disabled, which is what actually recovered that line.

*** SUPERSEDED — DO NOT TRUST OUTPUT FROM THIS SCRIPT (2026-07-28). ***

Two independent measurement bugs, either of which alone invalidates the table:

1. `RFP_MULT_BASE/CAP = 30000` is NOT a disable — the trend discount
   `2934*improving*futility_mult/1024` is ~2.87x futility_mult, so at depth 1-2
   the margin goes hugely negative and RFP becomes an UNCONDITIONAL shallow
   prune. Use `RFP_MAX_DEPTH = 0` (see `uciprobe.GATES_OFF`, which now carries
   the verification).
2. `sleep {wait}` then EOF: our engine treats stdin EOF as a STOP, so a search
   that outruns the sleep is truncated and the last `info depth N` line parsed
   is a partial re-search. That is why this script and `depthsplit.py` reported
   DIFFERENT moves for the same position at the same depth with binaries later
   verified bit-identical. Use `uciprobe.Engine`, which reads to `bestmove`.

And one interpretation bug that outlives both: on this class of position the
chosen move is a CHAOTIC function of the pruning constants — `RAZOR_BASE
465->464` flips game 18 at depth 17. A per-position "arm X changes the move"
verdict carries almost no information without `uciprobe.NULL_ARMS` measured
beside it.
"""
import json,os,re,subprocess,statistics as st
S=os.environ["S"]; NET=os.getcwd()+"/engines/canonical/net.cenn"
BIN="engines/bin/archon-d47v-tune-pgo"
OFF=("setoption name RAZOR_BASE value 30000\nsetoption name RFP_MAX_DEPTH value 0\n"
     "setoption name PFUT_BASE value 30000\n"
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
