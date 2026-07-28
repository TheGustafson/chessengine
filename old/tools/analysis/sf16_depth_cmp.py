#!/usr/bin/env python3
"""Does SF16 out-SEARCH ARCHON (depth gap -> speed lever) or out-EVAL it
(similar depth -> eval/fleet)? Compare per-move depth + time at equal TC,
overall and specifically at ARCHON's decisive-blunder positions."""
import sys, re, statistics
import chess, chess.pgn

PGN = sys.argv[1] if len(sys.argv) > 1 else "data/nnue/experiments/pgo-vs-sf16/games.pgn"
ARCHON = "archon-pgo"
evalre = re.compile(r"([+-]?\d+\.\d+|[+-]?#-?\d+)/(\d+)\s+([\d.]+)s")

def parse(comment):
    m = evalre.search(comment or "")
    if not m: return None
    raw = m.group(1)
    val = (100.0*(-1 if "-" in raw else 1)) if "#" in raw else float(raw)
    return val, int(m.group(2)), float(m.group(3))

a_depth, s_depth, a_time, s_time = [], [], [], []
# depth by game phase (fullmove buckets)
phase_depth = {"opening(<=15)":{"a":[],"s":[]}, "middle(16-40)":{"a":[],"s":[]}, "endgame(>40)":{"a":[],"s":[]}}
# paired: same position count where both searched — compare depth per fullmove
with open(PGN) as f:
    while True:
        g = chess.pgn.read_game(f)
        if g is None: break
        aw = g.headers.get("White","")==ARCHON
        board = g.board(); node = g
        while node.variations:
            node = node.variations[0]
            is_arch = (board.turn==chess.WHITE)==aw
            p = parse(node.comment)
            fm = board.fullmove_number
            board.push(node.move)
            if not p: continue
            _,depth,secs = p
            (a_depth if is_arch else s_depth).append(depth)
            (a_time if is_arch else s_time).append(secs)
            ph = "opening(<=15)" if fm<=15 else "middle(16-40)" if fm<=40 else "endgame(>40)"
            phase_depth[ph]["a" if is_arch else "s"].append(depth)

def stat(xs): return f"median {statistics.median(xs):5.1f}  mean {statistics.mean(xs):5.2f}" if xs else "n/a"
print(f"=== depth & time per move at equal TC (10+0.1), {len(a_depth)} ARCHON / {len(s_depth)} SF16 moves ===")
print(f"  ARCHON depth: {stat(a_depth)}")
print(f"  SF16   depth: {stat(s_depth)}")
print(f"  depth gap (SF16 - ARCHON): {statistics.median(s_depth)-statistics.median(a_depth):+.1f} median")
print(f"  ARCHON time/move: {stat(a_time)}s")
print(f"  SF16   time/move: {stat(s_time)}s")
print()
print("=== depth by phase (are we shallower everywhere, or only somewhere?) ===")
for ph,d in phase_depth.items():
    if d["a"] and d["s"]:
        print(f"  {ph:15s}  ARCHON {statistics.median(d['a']):4.1f}   SF16 {statistics.median(d['s']):4.1f}   gap {statistics.median(d['s'])-statistics.median(d['a']):+.1f}")
print()
print("INTERPRETATION:")
gap = statistics.median(s_depth)-statistics.median(a_depth)
if gap >= 3:
    print(f"  SF16 searches ~{gap:.0f} ply DEEPER at equal time -> a SEARCH/SPEED gap.")
    print("  The speed levers (smallnet gate, PGO) + search efficiency (LMR/pruning tuning)")
    print("  directly attack this. Depth is nodes: closing it needs more nodes/sec OR a")
    print("  smaller tree per depth. This VALIDATES the smallnet-gate speed program.")
else:
    print(f"  Depth gap is small ({gap:+.0f} ply) -> NOT primarily a search-depth gap.")
    print("  At similar depth SF16 still wins -> EVAL QUALITY is the deficit (the fleet lever),")
    print("  or SF16's tree shape (selectivity) differs. Speed alone won't close it.")
