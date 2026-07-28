"""Depth reached at a FIXED NODE budget, by game phase, ours vs Stockfish.

WHY this and not seconds.  In real games we reach 18.7 plies to Stockfish's
18.3 in the opening (equal time) but 17.7 to their 19.2 at moves 41-50 and
20.0 to their 22.5 at 51+.  It is tempting -- I did it -- to blame the clock,
since we arrive at move 45 with 35% of their time.  But at moves 41-50 they
spend only 13% more time than us and gain 1.5 plies, and 13% more time cannot
buy 1.5 plies at any plausible branching factor.  So most of that gap is not
time, it is TREE SHAPE, and the way to see tree shape without the clock in the
way is to give both engines the same node budget and read the depth.

Fixed nodes is deterministic, so this is load-immune and can run beside a
match (playbook 1.3) -- unlike any measurement in seconds.

    .venv/bin/python tools/forensics/depth_per_node.py [--nodes 200000]
        [--per-bucket 40] [--pgn ...]
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-pgo"
SF = REPO + "/engines/bin/stockfish18"


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def collect(pattern, per_bucket, seed):
    import chess
    import chess.pgn

    by = collections.defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        with open(path, errors="ignore") as fh:
            while (g := chess.pgn.read_game(fh)) is not None:
                node = g
                while node.variations:
                    node = node.variations[0]
                    b = node.board()
                    if b.is_game_over() or b.is_check():
                        continue
                    if len(list(b.legal_moves)) < 2:
                        continue
                    by[bucket(npieces(b.fen()))].append(b.fen())
    rng = random.Random(seed)
    return {k: rng.sample(v, min(per_bucket, len(v))) for k, v in by.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="engines/runs/rematch-v1/time/games.pgn")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--per-bucket", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    pos = collect(args.pgn, args.per_bucket, args.seed)
    ours = Engine(OURS, net=NET, hash_mb=16, name="archon")
    ours.assert_net_live()
    sf = Engine(SF, net=None, hash_mb=16, name="sf18")

    print(f"depth reached at {args.nodes} nodes (Hash 16, as in the match)")
    print(f"{'pieces':<8}{'n':>4}{'ARCHON':>9}{'SF18':>8}{'gap':>7}{'agree':>8}")
    for k in ("21+", "13-20", "8-12", "<=7"):
        fens = pos.get(k) or []
        if not fens:
            continue
        a, s, agree = [], [], 0
        for f in fens:
            ours.newgame()
            ra = ours.go(fen=f, nodes=args.nodes)
            sf.newgame()
            rs = sf.go(fen=f, nodes=args.nodes)
            if not ra["ladder"] or not rs["ladder"]:
                continue
            a.append(max(ra["ladder"]))
            s.append(max(rs["ladder"]))
            agree += (ra["best"] == rs["best"])
        if not a:
            continue
        ma, ms = statistics.mean(a), statistics.mean(s)
        print(f"{k:<8}{len(a):>4}{ma:>9.2f}{ms:>8.2f}{ma-ms:>+7.2f}"
              f"{100*agree/len(a):>7.0f}%", flush=True)
    ours.close()
    sf.close()


if __name__ == "__main__":
    main()
