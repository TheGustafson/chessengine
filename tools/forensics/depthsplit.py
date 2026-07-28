"""Split the 'not-pruning' errors into DEPTH vs EVAL.
For each, search at the depth reached in-game, then +8 plies. If the move
changes AND matches what a deep SF18 says, more depth alone would have saved
it -> DEPTH/TIME. If it never changes, our eval is simply wrong -> EVAL.
"""
import json,os,re,subprocess
S=os.environ["S"]; NET=os.getcwd()+"/data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
def go(binary,fen,depth,net=True,wait=9):
    s="uci\n"+(f"setoption name EvalFile value {NET}\n" if net else "")
    s+=f"isready\nposition fen {fen}\ngo depth {depth}\n"
    out=subprocess.run(["bash","-c",f'{{ printf %s "$SC"; sleep {wait}; }} | timeout {wait+40} {binary}'],
        capture_output=True,text=True,env={**os.environ,"SC":s}).stdout
    m=None
    for line in out.splitlines():
        if line.startswith(f"info depth {depth} "):
            g=re.search(r" pv (\S+)",line)
            if g: m=g.group(1)
    return m
d=json.load(open(f"{S}/blunders_wp.json"))
recs=[r for r in d["archon"] if r["class"] in ("search_miss","eval_instability","inaccuracy")]
notpr={"17","22","24","28","34","3"}
print(f"{'game':<6}{'d':<4}{'ours@d':<8}{'ours@d+8':<10}{'SF deep':<9}verdict")
for r in recs:
    dep=max(8,min(int(r["depth"]),16))
    a=go("engines/bin/archon-d47v-pgo",r["fen"],dep)
    b=go("engines/bin/archon-d47v-pgo",r["fen"],dep+8,wait=14)
    s=go("engines/bin/stockfish18",r["fen"],dep+8,net=False,wait=14)
    if not a or not b or not s: continue
    if a==s: v="already right"
    elif b==s: v="DEPTH/TIME  (more depth alone fixes it)"
    else:     v="EVAL       (never converges to SF's move)"
    print(f"g{r['game']:<5}{dep:<4}{a:<8}{b:<10}{s:<9}{v}",flush=True)
