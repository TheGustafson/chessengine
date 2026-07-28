"""Harvest EVERY >=100cp error we made in lost games -- the full gift corpus.

Catastrophic losses are the cheapest defects to fix because each leaves a
complete trace: a position, our move, the right move, a measurable gap, and a
reproducible search to bisect. This extracts all of them, detector-independent
(the oracle is asked about every move, not only the ones our eval noticed).
"""
import os,re,json,math,chess,chess.pgn,chess.engine
S=os.environ["S"]
A=re.compile(r"([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s")
moves=[]
for leg in ("time","nodes"):
    p=f"engines/runs/rematch-v1/{leg}/games.pgn"
    if not os.path.exists(p): continue
    with open(p,errors="ignore") as fh:
        while (g:=chess.pgn.read_game(fh)) is not None:
            aw="archon" in g.headers.get("White","").lower(); r=g.headers.get("Result","*")
            if not ((r=="1-0" and not aw) or (r=="0-1" and aw)): continue
            node=g
            while node.variations:
                node=node.variations[0]
                m=A.search(node.comment or "")
                if not m: continue
                b=node.parent.board()
                if (b.turn==chess.WHITE)==aw and not b.is_check():
                    moves.append((leg,g.headers.get("Round","?"),b.fen(),node.move.uci(),int(m.group(2))))
print(f"{len(moves)} ARCHON moves in lost games -> auditing all",flush=True)
eng=chess.engine.SimpleEngine.popen_uci("engines/bin/stockfish18"); eng.configure({"Threads":1,"Hash":256})
def wp(cp): return 1/(1+math.exp(-cp/170.0))
hits=[]
for i,(leg,rnd,fen,mv,dep) in enumerate(moves,1):
    b=chess.Board(fen)
    try:
        best=eng.analyse(b,chess.engine.Limit(nodes=100000))
        got =eng.analyse(b,chess.engine.Limit(nodes=100000),root_moves=[chess.Move.from_uci(mv)])
    except Exception: continue
    bs=best["score"].relative.score(mate_score=20000); ps=got["score"].relative.score(mate_score=20000)
    if bs is None or ps is None: continue
    loss=bs-ps
    if loss>=100:
        hits.append({"leg":leg,"game":rnd,"fen":fen,"played":mv,
                     "best":best["pv"][0].uci() if best.get("pv") else None,
                     "loss":loss,"wp_loss":round(wp(bs)-wp(ps),3),
                     "game_depth":dep,"best_score":bs})
    if i%400==0: print(f"  {i}/{len(moves)}  hits so far {len(hits)}",flush=True)
eng.quit()
hits.sort(key=lambda h:-h["wp_loss"])
json.dump(hits,open(f"{S}/gift_corpus.json","w"),indent=1)
open(f"{S}/gift_corpus.fens","w").write("\n".join(h["fen"] for h in hits)+"\n")
print(f"\n{len(hits)} errors >=100cp harvested -> gift_corpus.json")
print(f"{'game':<7}{'played':>8}{'best':>8}{'cp':>7}{'wp':>7}{'d':>4}")
for h in hits[:20]:
    print(f"g{h['game']:<6}{h['played']:>8}{str(h['best']):>8}{h['loss']:>7}{h['wp_loss']:>7.2f}{h['game_depth']:>4}")
