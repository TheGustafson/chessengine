"""What does a top-1 disagreement actually COST, per phase?

WHY this has to come before chasing the 8-12 deficit.  Top-1 agreement is a
rate, which makes it robust to mate scores -- but it is binary, and in sparse
positions many moves are near-equal.  A 7-point top-1 gap at 8-12 pieces is
worth chasing only if the moves we pick instead are actually worse; if each
disagreement costs 5 cp it is a metric artefact and the real deficit is
somewhere else.

So for every position where our move differs from the deep oracle's, ask the
oracle what OUR move is worth at the same budget, and report the distribution of
the loss in win probability as well as centipawns.  Win probability is the unit
that converts to results; centipawns in a sparse position do not.

Also reported: the move-kind breakdown of the oracle's choice on the
disagreements versus on the agreements.  A systematic bias -- we miss quiet king
moves, or pawn pushes -- points at specific machinery, where an undifferentiated
gap points nowhere.

    .venv/bin/python tools/forensics/disagree_cost.py [--oracle deep-top1.json]
        [--nodes 200000] [--limit 0]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-pgo"
SF = REPO + "/engines/bin/stockfish18"


def wp(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / 170.0))


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def kind(board, mv):
    import chess

    tags = []
    if board.is_capture(mv):
        tags.append("cap")
    b2 = board.copy()
    b2.push(mv)
    if b2.is_check():
        tags.append("chk")
    if mv.promotion:
        tags.append("promo")
    if not tags:
        pt = board.piece_at(mv.from_square)
        name = chess.piece_name(pt.piece_type) if pt else "?"
        return f"quiet-{name}"
    return "+".join(tags)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="tools/nnue/calib/deep-top1.json")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import chess

    o = json.load(open(args.oracle))
    rows = list(o["best"].items())
    if args.limit:
        rows = rows[: args.limit]
    onodes = o["nodes"]

    ours = Engine(OURS, net=NET, hash_mb=16, name="archon")
    ours.assert_net_live()
    sf = Engine(SF, net=None, hash_mb=64, name="sf18")

    loss = collections.defaultdict(list)
    agree = collections.Counter()
    total = collections.Counter()
    kd = collections.defaultdict(collections.Counter)
    ka = collections.defaultdict(collections.Counter)

    for fen, (best, bcp) in rows:
        b = bucket(npieces(fen))
        total[b] += 1
        ours.newgame()
        got = ours.go(fen=fen, nodes=args.nodes)["best"]
        board = chess.Board(fen)
        if got == best:
            agree[b] += 1
            ka[b][kind(board, chess.Move.from_uci(best))] += 1
            continue
        kd[b][kind(board, chess.Move.from_uci(best))] += 1
        sf.newgame()
        r = sf.go(fen=fen, nodes=onodes, searchmoves=[got])
        if not r["ladder"]:
            continue
        ocp = r["ladder"][max(r["ladder"])]["score"]
        loss[b].append((min(bcp, 20000) - min(ocp, 20000), wp(bcp) - wp(ocp)))

    ours.close()
    sf.close()

    print(f"oracle @ {onodes} nodes, ours @ {args.nodes}\n")
    print(f"{'pieces':<8}{'n':>5}{'top1':>8}{'disagree':>10}"
          f"{'med cp':>9}{'mean cp':>9}{'med wp':>9}{'mean wp':>9}")
    for k in ("21+", "13-20", "8-12", "<=7"):
        if not total[k]:
            continue
        L = loss[k]
        if not L:
            continue
        cps = [x[0] for x in L]
        wps = [x[1] for x in L]
        print(f"{k:<8}{total[k]:>5}{100*agree[k]/total[k]:>7.1f}%{len(L):>10}"
              f"{statistics.median(cps):>9.0f}{statistics.mean(cps):>9.0f}"
              f"{statistics.median(wps):>9.3f}{statistics.mean(wps):>9.3f}")

    print("\nmove-kind of the ORACLE's choice, disagreements vs agreements")
    for k in ("8-12", "<=7", "13-20", "21+"):
        if not kd[k]:
            continue
        tot_d = sum(kd[k].values())
        tot_a = sum(ka[k].values()) or 1
        print(f"\n  {k}  (n_dis={tot_d})")
        for kind_name, c in kd[k].most_common(6):
            share_d = 100 * c / tot_d
            share_a = 100 * ka[k][kind_name] / tot_a
            flag = "  <-- over-represented" if share_d > share_a + 6 else ""
            print(f"    {kind_name:<16}{share_d:>6.1f}% of misses  "
                  f"{share_a:>6.1f}% of hits{flag}")


if __name__ == "__main__":
    main()
