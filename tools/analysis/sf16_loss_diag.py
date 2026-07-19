#!/usr/bin/env python3
"""Diagnose WHERE ARCHON loses to SF16, from the equal-time match PGN.

Each move carries {eval/depth time}. We build a unified ARCHON-POV eval
trajectory (negate SF16's evals), then for each game classify result, the
decisive swing (ply where ARCHON-POV eval crosses -1.5 for good), the phase
(by material) at that swing, eval overoptimism (ARCHON eval stays >=0 while
already lost), and time spent on the decisive move.
"""
import sys, re, statistics
import chess, chess.pgn

PGN = sys.argv[1] if len(sys.argv) > 1 else "data/nnue/experiments/pgo-vs-sf16/games.pgn"
LOST = -1.5   # ARCHON-POV eval (pawns) below which we call ARCHON "losing"
ARCHON = "archon-pgo"

evalre = re.compile(r"([+-]?\d+\.\d+|[+-]?M?\d+|[+-]?#-?\d+)/(\d+)\s+([\d.]+)s")

def parse_eval(comment):
    """Return (eval_pawns, depth, secs) from a '{+1.12/18 0.800s}' comment, mover POV."""
    m = evalre.search(comment or "")
    if not m:
        return None
    raw, depth, secs = m.group(1), int(m.group(2)), float(m.group(3))
    if "M" in raw or "#" in raw:      # mate score
        sign = -1 if "-" in raw else 1
        val = 100.0 * sign
    else:
        val = float(raw)
    return val, depth, secs

def material(board):
    v = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}
    return sum(v[p.piece_type] for p in board.piece_map().values() if p.piece_type != chess.KING)

def phase_of(mat):
    # total non-king material starts at 78 (2*(1*8+3*4+5*2+9)); classify by what's left
    if mat >= 52: return "opening"
    if mat >= 24: return "middlegame"
    return "endgame"

results = {"W":0,"L":0,"D":0}
by_color = {"white":{"W":0,"L":0,"D":0}, "black":{"W":0,"L":0,"D":0}}
loss_swing_phase = {"opening":0,"middlegame":0,"endgame":0,"never":0}
loss_swing_fullmove = []
loss_swing_time = []
loss_overoptimism = []   # max ARCHON eval while already lost (>0 = optimistic in a lost game)
loss_plycount = []
windraw_plycount = []
loss_swing_depth = []
n = 0

with open(PGN) as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        white = game.headers.get("White","")
        archon_white = (white == ARCHON)
        res = game.headers.get("Result","*")
        if res == "*":
            continue
        # ARCHON-POV result
        if res == "1/2-1/2":
            outcome = "D"
        elif (res == "1-0") == archon_white:
            outcome = "W"
        else:
            outcome = "L"
        results[outcome] += 1
        by_color["white" if archon_white else "black"][outcome] += 1
        n += 1

        # Build ARCHON-POV eval trajectory over plies.
        board = game.board()
        traj = []  # (fullmove, archon_eval, secs, depth, material_after)
        node = game
        while node.variations:
            node = node.variations[0]
            mv = node.move
            mover_is_archon = (board.turn == chess.WHITE) == archon_white
            pe = parse_eval(node.comment)
            board.push(mv)
            if pe is None:
                continue
            ev, depth, secs = pe
            archon_ev = ev if mover_is_archon else -ev
            traj.append((board.fullmove_number, archon_ev, secs, depth, material(board), mover_is_archon))

        pc = int(game.headers.get("PlyCount", len(traj)))
        if outcome == "L":
            loss_plycount.append(pc)
            # decisive swing: first ply where ARCHON-POV eval <= LOST and never recovers above LOST/2
            swing = None
            for i,(fm,ae,secs,depth,mat,is_arch) in enumerate(traj):
                if ae <= LOST:
                    if all(t[1] <= LOST/2 for t in traj[i:]):
                        swing = (fm,ae,secs,depth,mat,is_arch)
                        break
            if swing is None:
                loss_swing_phase["never"] += 1  # lost without our evals ever flagging it (eval blindspot!)
            else:
                fm,ae,secs,depth,mat,is_arch = swing
                loss_swing_phase[phase_of(mat)] += 1
                loss_swing_fullmove.append(fm)
                if is_arch:
                    loss_swing_time.append(secs)
                    loss_swing_depth.append(depth)
                # overoptimism: max ARCHON eval AFTER the swing (still thinking it's ok while lost)
                after = [t[1] for t in traj if t[0] >= fm and t[5]]  # ARCHON's own evals after
                if after:
                    loss_overoptimism.append(max(after))
        else:
            windraw_plycount.append(pc)

def dist(xs):
    if not xs: return "n/a"
    xs = sorted(xs)
    return f"median {statistics.median(xs):.1f}  mean {statistics.mean(xs):.1f}  p25 {xs[len(xs)//4]:.1f}  p75 {xs[3*len(xs)//4]:.1f}"

print(f"=== ARCHON vs SF16 (pgo, 10+0.1) — {n} games ===")
w,l,d = results["W"],results["L"],results["D"]
score = (w + 0.5*d)/n*100
print(f"  W {w}  L {l}  D {d}   score {score:.1f}%   (raw Elo ~ {-400*__import__('math').log10(1/(score/100)-1):.0f})" if 0<score<100 else f"  W {w} L {l} D {d}")
print(f"  as White: {by_color['white']}   as Black: {by_color['black']}")
print()
print(f"=== of {l} LOSSES ===")
print(f"  decisive-swing PHASE (by material when ARCHON's eval crossed {LOST}):")
for ph in ("opening","middlegame","endgame","never"):
    c = loss_swing_phase[ph]
    bar = "#"*int(40*c/max(1,l))
    label = ph if ph!="never" else "never-flagged (ARCHON eval blindspot: lost w/o our eval warning)"
    print(f"    {label:16s} {c:3d} {bar}")
print(f"  decisive-swing fullmove #: {dist(loss_swing_fullmove)}")
print(f"  loss game length (plies):  {dist(loss_plycount)}")
print(f"  win/draw game length:      {dist(windraw_plycount)}")
print(f"  time spent on ARCHON's decisive blunder move (s): {dist(loss_swing_time)}")
print(f"  ARCHON depth on the decisive blunder move:        {dist(loss_swing_depth)}")
print(f"  overoptimism (max ARCHON eval AFTER it was already lost; >0 = eval too rosy): {dist(loss_overoptimism)}")
opt = [x for x in loss_overoptimism if x > 0]
print(f"    -> {len(opt)}/{len(loss_overoptimism)} lost games where ARCHON's eval stayed POSITIVE while already lost")
