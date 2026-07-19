#!/usr/bin/env python3
"""Mine ARCHON's eval blindspots from the SF16 match: positions (after an ARCHON
move) where ARCHON's eval was non-losing but SF16's reply reveals it was already
bad — i.e. ARCHON's static eval + shallow search was OPTIMISTIC and wrong.

Signal: E_a = ARCHON's eval (ARCHON POV) right after ARCHON's move M. E_s' =
-(SF16's eval after its reply) (ARCHON POV, ~1 move later). drop = E_a - E_s'.
Large positive drop with E_a not-yet-losing = an ARCHON eval blindspot. Output
an EPD regression suite (does a future net judge these better?) + phase stats.
"""
import sys, re, statistics, collections
import chess, chess.pgn

PGN = sys.argv[1] if len(sys.argv) > 1 else "data/nnue/experiments/pgo-vs-sf16/games.pgn"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/nnue/testsuites/archon-eval-blindspots.epd"
ARCHON = "archon-pgo"
DROP = 1.5        # pawns: how much ARCHON over-rated the position
E_A_MIN = -0.50   # ARCHON thought it was at worst slightly worse ...
E_A_MAX = 1.50    # ... and at best only slightly better (exclude won-position compression)
E_S_MAX = -1.00   # ... but after SF16's reply ARCHON is actually LOSING
evalre = re.compile(r"([+-]?\d+\.\d+|[+-]?#-?\d+)/(\d+)\s+([\d.]+)s")

def parse(c):
    m = evalre.search(c or "")
    if not m: return None
    raw = m.group(1)
    val = (100.0*(-1 if "-" in raw else 1)) if "#" in raw else float(raw)
    return val, int(m.group(2))

def mat(board):
    v={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9}
    return sum(v[p.piece_type] for p in board.piece_map().values() if p.piece_type!=chess.KING)

def phase(m): return "opening" if m>=52 else "middlegame" if m>=24 else "endgame"

blind=[]
with open(PGN) as f:
    while True:
        g=chess.pgn.read_game(f)
        if g is None: break
        aw=g.headers.get("White","")==ARCHON
        board=g.board(); node=g
        prev=None  # (fen_after_archon_move, E_a, fullmove, material)
        while node.variations:
            node=node.variations[0]; mv=node.move
            is_arch=(board.turn==chess.WHITE)==aw
            pe=parse(node.comment)
            board.push(mv)
            if pe is None:
                prev=None; continue
            ev,depth=pe
            if is_arch:
                prev=(board.fen(), ev, board.fullmove_number, mat(board))  # ARCHON POV eval after its move
            else:
                # SF16 just replied; -ev is ARCHON POV of the position after SF16's move
                if prev:
                    fen,E_a,fm,m0=prev
                    E_s = -ev
                    drop = E_a - E_s
                    if drop>=DROP and E_A_MIN<=E_a<=E_A_MAX and E_s<=E_S_MAX:
                        blind.append((drop,E_a,E_s,fen,fm,m0))
                prev=None

blind.sort(reverse=True)
print(f"=== ARCHON eval blindspots vs SF16 ({len(blind)} positions, drop>={DROP}, E_a>={E_A_MIN}) ===")
ph=collections.Counter(phase(b[5]) for b in blind)
tot=max(1,len(blind))
for p in ("opening","middlegame","endgame"):
    print(f"  {p:12s} {ph[p]:4d}  ({100*ph[p]/tot:.0f}%)")
print(f"  drop magnitude: median {statistics.median([b[0] for b in blind]):.2f}  mean {statistics.mean([b[0] for b in blind]):.2f}")
print(f"  fullmove of blindspot: median {statistics.median([b[4] for b in blind]):.0f}")
print(f"  worst 5:")
for drop,E_a,E_s,fen,fm,m0 in blind[:5]:
    print(f"    mv{fm} {phase(m0):11s} ARCHON {E_a:+.2f} -> after SF16 reply {E_s:+.2f} (drop {drop:.2f})  {fen}")

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"w") as o:
    for drop,E_a,E_s,fen,fm,m0 in blind:
        o.write(f'{fen} ; archon_eval {E_a:+.2f} ; sf16_implied {E_s:+.2f} ; drop {drop:.2f} ; phase {phase(m0)}\n')
print(f"\nwrote {len(blind)} blindspot positions -> {OUT}")
print("Use: a regression suite — a better net should rate these closer to the SF16-implied value.")
