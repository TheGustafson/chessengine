#!/usr/bin/env python3
"""Mine a loss corpus from an ARCHON-vs-SF18 match, oracle-scored by SF18.

WHY NOT a self-eval detector. The previous generation of this
(`tools/blunder_forensics.py`) only fired when OUR OWN eval collapsed within two
plies, which gives it roughly **9% recall** — a move whose refutation lands ten
plies later is invisible to it. A detector-independent audit found ~1.5% of our
moves in lost games lose >=100 cp, about ONE per lost game. So this scores EVERY
ARCHON move in every lost game with the oracle and never consults our own eval
to decide what is interesting.

METHOD, per ARCHON move in a lost game:
  1. SF18 searches the position freely  -> its best move and score
  2. SF18 searches with `searchmoves <our move>` -> the score OF OUR MOVE
Both at the same node budget, both from SF's own perspective, so no cross-engine
frame conversion is involved anywhere (CLAUDE.md R2).

  cp_loss = sf_best_score - sf_our_move_score      (>= 0 by construction)
  wp_loss = win_prob(sf_best) - win_prob(our move)

**Win probability, not centipawns, is the ranking key.** Summing cp assumes
small inaccuracies add linearly into outcomes; they mostly do not — they are
absorbed — while one large error at a decision point converts a result.

Output: `losses.json`, one row per scored move. Classification into
OURS / BOTH / near-tie comes later (classify_losses.py) because it needs OUR
engine at a budget, which is a separate and slower pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, REPO  # noqa: E402


def win_prob(cp: float) -> float:
    """Logistic win probability from SF-printed cp. The 1/350 slope is the usual
    engine-community fit; only DIFFERENCES of this are used, so the exact
    constant matters far less than using probability at all."""
    return 1.0 / (1.0 + math.exp(-cp / 350.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--engine-name", default="archon",
                    help="the PGN name tag of OUR engine")
    ap.add_argument("--oracle", default="engines/bin/stockfish18")
    ap.add_argument("--nodes", type=int, default=120_000,
                    help="oracle budget per probe; 120k is the tier used for the "
                         "D47V corpus, so results stay comparable")
    ap.add_argument("--min-ply", type=int, default=16,
                    help="skip the book/opening plies")
    ap.add_argument("--out", default="searchopt/positions/losses.json")
    ap.add_argument("--limit-games", type=int, default=0)
    a = ap.parse_args()
    os.chdir(REPO)
    # python-chess is imported HERE, not at module level, so `--help` works in a plain
    # interpreter. It is not in the repo venv by design; run under uv.
    global chess
    try:
        import chess, chess.pgn                                        # noqa: F401
    except ModuleNotFoundError:
        print("needs python-chess. Run it as:\n"
              "  uv run --no-project --with chess python " + sys.argv[0] + " ...",
              file=sys.stderr)
        return 1

    # 1. collect the positions to score: every ARCHON move from a game ARCHON LOST
    jobs = []
    with open(a.pgn) as fh:
        g_i = 0
        while True:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            w, b, res = g.headers.get("White",""), g.headers.get("Black",""), g.headers.get("Result","*")
            if a.engine_name not in (w, b):
                continue
            we_are_white = (w == a.engine_name)
            lost = (res == "0-1" and we_are_white) or (res == "1-0" and not we_are_white)
            if not lost:
                continue
            g_i += 1
            if a.limit_games and g_i > a.limit_games:
                break
            board = g.board()
            for ply, mv in enumerate(g.mainline_moves()):
                ours = (board.turn == chess.WHITE) == we_are_white
                if ours and ply >= a.min_ply and not board.is_game_over():
                    jobs.append({"game": g_i, "ply": ply, "fen": board.fen(),
                                 "played": mv.uci(), "result": res})
                board.push(mv)
    print(f"{g_i} lost games -> {len(jobs)} ARCHON moves to score", flush=True)
    if not jobs:
        return 1

    # 2. oracle-score each: free search, then the same search forced onto our move
    s = Engine(a.oracle, net=None, hash_mb=64, name="sf18")
    rows = []
    for i, j in enumerate(jobs, 1):
        s.newgame()
        free = s.go(fen=j["fen"], nodes=a.nodes)
        if not free["ladder"]:
            continue
        top = free["ladder"][max(free["ladder"])]
        best_mv, best_cp = top["move"], top["score"]
        if best_mv == j["played"]:
            our_cp = best_cp                      # we played the oracle's move
        else:
            s.newgame()
            forced = s.go(fen=j["fen"], nodes=a.nodes, searchmoves=[j["played"]])
            if not forced["ladder"]:
                continue
            our_cp = forced["ladder"][max(forced["ladder"])]["score"]
        # both scores are SF's, from the mover's perspective
        rows.append({**j, "best": best_mv, "best_cp": best_cp, "our_cp": our_cp,
                     "cp_loss": best_cp - our_cp,
                     "wp_loss": win_prob(best_cp) - win_prob(our_cp)})
        if i % 50 == 0:
            print(f"  [{i}/{len(jobs)}] scored", flush=True)
            json.dump(rows, open(a.out, "w"))
    s.close()
    # PROVENANCE STAMP: a corpus is net-specific. Positions mined from one net say
    # nothing about where a different net discards moves, and after the D48X promotion
    # I only noticed the D47V corpus was stale by reasoning about it. Now it announces.
    import hashlib
    net = os.path.realpath("engines/canonical/net.cenn")
    h = hashlib.sha256(open(net, "rb").read(1 << 20)).hexdigest()[:16] if os.path.exists(net) else "?"
    json.dump({"_provenance": {"net": os.path.basename(net), "net_head_sha256": h,
                               "pgn": a.pgn, "oracle_nodes": a.nodes},
               "rows": rows}, open(a.out, "w"))

    big = [r for r in rows if r["wp_loss"] >= 0.10]
    print(f"\n{len(rows)} moves scored -> {a.out}")
    print(f"  wp_loss >= 0.10 : {len(big)}  ({100*len(big)/max(1,len(rows)):.1f}%)")
    print(f"  cp_loss >= 100  : {sum(1 for r in rows if r['cp_loss'] >= 100)}")
    print(f"  per lost game   : {len(big)/max(1,g_i):.2f} big errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
