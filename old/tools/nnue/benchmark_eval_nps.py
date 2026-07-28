#!/usr/bin/env python3
"""Measure matched single-thread fixed-node search throughput for one evaluator."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import chess, chess.engine

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine",type=Path,required=True); ap.add_argument("--net",type=Path,required=True)
    ap.add_argument("--fens",type=Path,required=True); ap.add_argument("--nodes",type=int,default=200_000)
    ap.add_argument("--positions",type=int,default=8); ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    fens=[x.strip() for x in a.fens.read_text().splitlines() if x.strip() and not x.startswith("#")][:a.positions]
    engine=chess.engine.SimpleEngine.popen_uci(str(a.engine.resolve()))
    rows=[]
    try:
        engine.configure({"Threads":1,"Hash":16,"EvalFile":str(a.net.resolve())})
        for fen in fens:
            board=chess.Board(fen); began=time.perf_counter()
            info=engine.analyse(board,chess.engine.Limit(nodes=a.nodes)); elapsed=time.perf_counter()-began
            rows.append({"fen":fen,"nodes":int(info.get("nodes",0)),"nps":int(info.get("nps",0)),
                         "wall_nps":int(info.get("nodes",0)/elapsed),"elapsed":elapsed})
    finally: engine.quit()
    nps=[r["nps"] for r in rows if r["nps"]>0]
    report={"engine":str(a.engine.resolve()),"net":str(a.net.resolve()),"requested_nodes":a.nodes,
            "positions":len(rows),"geometric_mean_nps":math.exp(sum(map(math.log,nps))/len(nps)),
            "median_nps":sorted(nps)[len(nps)//2],"rows":rows}
    text=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text)
    print(text,end="")
if __name__=="__main__": main()
