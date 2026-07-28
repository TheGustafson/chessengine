"""Where does the RANKING of two candidate moves flip, in each engine?

WHY this rung exists.  A blunder can fail in two structurally different ways
and they need opposite fixes:

  RESOLUTION  both engines rank the moves the same way until some depth, and
              the refutation simply needs more plies than we had.  Fix: search
              efficiency, time management, or the specific reduction on that
              line.
  PREFERENCE  the reference engine ranks them correctly from a SHALLOW depth
              and we rank them backwards for many plies.  Fix: eval.  No amount
              of search shaping will help, because our own tree is telling us
              the wrong move is better.

`searchmoves` gives each candidate the root to itself, so what comes back is
the search's honest value for that move with root ordering and root pruning
removed.  Ranking is frame-free -- it compares two numbers from the SAME
engine -- which is what lets our internal ~0.58x frame be compared with
Stockfish's at all (playbook: never compare raw cp across engines).

    .venv/bin/python tools/forensics/rank_ladder.py FEN PLAYED BEST [--max-depth 20]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-tune-pgo"
SF = REPO + "/engines/bin/stockfish18"


def forced_ladder(eng: Engine, fen: str, move: str, max_depth: int) -> dict[int, int]:
    eng.newgame()
    r = eng.go(fen=fen, depth=max_depth, searchmoves=[move])
    return {d: r["ladder"][d]["score"] for d in r["ladder"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fen")
    ap.add_argument("played")
    ap.add_argument("best")
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--ours", default=OURS)
    args = ap.parse_args()

    rows = {}
    for name, path, net in (("ARCHON", args.ours, NET), ("SF18", SF, None)):
        e = Engine(path, net=net, hash_mb=64, name=name)
        if net:
            e.assert_net_live()
        rows[name] = (forced_ladder(e, args.fen, args.played, args.max_depth),
                      forced_ladder(e, args.fen, args.best, args.max_depth))
        e.close()

    print(f"position {args.fen}")
    print(f"played={args.played}  best={args.best}   "
          f"(+ = that engine prefers BEST; ranking is frame-free)")
    print(f"{'d':>3} | {'ARCHON played':>13}{'best':>7}{'diff':>7} | "
          f"{'SF18 played':>13}{'best':>7}{'diff':>7}")
    flip = {}
    for d in range(1, args.max_depth + 1):
        cells = []
        for name in ("ARCHON", "SF18"):
            p, b = rows[name][0].get(d), rows[name][1].get(d)
            if p is None or b is None:
                cells.append(f"{'-':>13}{'-':>7}{'-':>7}")
                continue
            cells.append(f"{p:>13}{b:>7}{b - p:>+7}")
            if b > p:
                flip.setdefault(name, d)
            else:
                flip.pop(name, None)      # must HOLD the correct ranking
        print(f"{d:>3} | {cells[0]} | {cells[1]}")

    print()
    for name in ("ARCHON", "SF18"):
        f = flip.get(name)
        print(f"{name:<7} ranks BEST above PLAYED from depth "
              f"{f if f else 'never'}")
    a, s = flip.get("ARCHON"), flip.get("SF18")
    if a and s:
        gap = a - s
        print(f"\nverdict: {'PREFERENCE' if gap >= 4 else 'RESOLUTION'} "
              f"(ranking gap {gap:+d} plies)")


if __name__ == "__main__":
    main()
