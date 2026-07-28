"""Does carried search state make us play worse than a cold search would?

WHY this control is the only honest one.  11 of 29 depth-fixable errors are
found by a COLD search at or below the depth the game itself reached, which looks
like search-state contamination -- TT, history and correction tables carried from
earlier moves making the in-game search worse than a fresh one.

But it is not evidence on its own.  These positions have a chaotic argmax (a ±1
change to an unrelated constant flips them), and the corpus SELECTED positions
where the game got it wrong.  If the choice is near-coin-flip, a cold search
would land on the right move a good fraction of the time purely by chance, with
no state effect at all.

The discriminating experiment holds the position and the depth fixed and varies
ONLY the state:

  cold     `ucinewgame`, then search the position directly
  replay   feed the real game's moves from the start, searching at every one of
           OUR moves, so the TT, histories and correction tables arrive in the
           same condition the real game left them in -- then read what we play

If replay errs where cold succeeds, state is the mechanism.  If they agree, the
"should have found it" bucket is chance and the mode does not exist.

Uses the [FEN] header when present and feeds `position fen ... moves ...`, so the
engine sees the true game ply -- a startpos replay of a book-opening game
desynchronises silently and every ply-dependent term is then wrong.

    .venv/bin/python tools/forensics/replay_vs_cold.py CLASSIFIED.json
        [--pgn 'engines/runs/corpus-v2/*/games.pgn'] [--limit 0]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-pgo"


def index_games(patterns):
    """round -> LIST of (start-fen, [uci moves], archon_is_white).

    A list, not one entry: fastchess `-repeat -games 2` plays every opening
    TWICE with colours swapped, so a round number names two games.  Keeping only
    the first silently lost 26 of 44 rows to "fen_not_in_game" and halved the
    power of the comparison.
    """
    import chess.pgn

    out = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, errors="ignore") as fh:
                while (g := chess.pgn.read_game(fh)) is not None:
                    rnd = g.headers.get("Round", "?")
                    mv, node = [], g
                    while node.variations:
                        node = node.variations[0]
                        mv.append(node.move.uci())
                    out.setdefault(rnd, []).append(
                        (g.headers.get("FEN"), mv,
                         "archon" in g.headers.get("White", "").lower()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("classified")
    ap.add_argument("--pgn", action="append",
                    default=["engines/runs/corpus-v2/*/games.pgn"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verdicts", default="DEPTH,DEPTH_vs_EVAL,SEARCH_REJECTS,EVAL")
    args = ap.parse_args()

    want = set(args.verdicts.split(","))
    rows = [r for r in json.load(open(args.classified)) if r["verdict"] in want]
    if args.limit:
        rows = rows[: args.limit]
    games = index_games(args.pgn)

    import chess

    eng = Engine(OURS, net=NET, hash_mb=16, name="archon")   # 16 MB, as in the match
    eng.assert_net_live()

    tally = collections.Counter()
    print(f"{'game':<6}{'best':>7}{'d':>4}{'cold':>7}{'replay':>8}{'ply':>5}  outcome")
    for r in rows:
        cands = games.get(str(r["game"])) or []
        if not cands:
            tally["no_game"] += 1
            continue
        start_fen = moves = aw = None
        target = None
        for sf, mv, a in cands:                 # try BOTH games of the round
            b = chess.Board(sf) if sf else chess.Board()
            for i, u in enumerate(mv):
                if b.fen() == r["fen"]:
                    target, start_fen, moves, aw = i, sf, mv, a
                    break
                try:
                    b.push(chess.Move.from_uci(u))
                except Exception:  # noqa: BLE001
                    break
            if target is not None:
                break
        if target is None:
            tally["fen_not_in_game"] += 1
            continue
        d = int(r["game_depth"])
        head = f"fen {start_fen}" if start_fen else "startpos"

        eng.newgame()
        cold = eng.go(fen=r["fen"], depth=d)["best"]

        # replay: search at every one of OUR moves up to the target, no newgame
        eng.newgame()
        rep = None
        for i in range(0, target + 1):
            if (i % 2 == 0) != aw:
                continue                      # opponent move: just advance
            pos = f"position {head}" + (" moves " + " ".join(moves[:i]) if i else "")
            eng.send(pos)
            eng.send(f"go depth {d}")
            while True:
                line = eng.p.stdout.readline()
                if not line or line.startswith("bestmove"):
                    rep = line.split()[1] if line.startswith("bestmove") else rep
                    break
        c_ok, r_ok = cold == r["best"], rep == r["best"]
        key = ("cold_ok_replay_bad" if c_ok and not r_ok else
               "both_ok" if c_ok and r_ok else
               "replay_ok_cold_bad" if r_ok else "both_bad")
        tally[key] += 1
        print(f"g{str(r['game']):<5}{r['best']:>7}{d:>4}{cold:>7}{str(rep):>8}"
              f"{target:>5}  {key}", flush=True)
    eng.close()

    print("\n=== STATE EFFECT ===")
    tot = sum(tally[k] for k in ("cold_ok_replay_bad", "both_ok",
                                 "replay_ok_cold_bad", "both_bad"))
    for k, n in tally.most_common():
        pct = f"{100*n/tot:5.1f}%" if tot and k not in ("no_game", "fen_not_in_game") else "     "
        print(f"{n:>4}  {pct}  {k}")
    if tot:
        a = tally["cold_ok_replay_bad"]
        b = tally["replay_ok_cold_bad"]
        print(f"\ncold beats replay on {a}, replay beats cold on {b}.")
        print("A real state effect means a >> b. If a ~= b it is the chaotic argmax,")
        print("and the 'should have found it' bucket does not exist as a mode.")


if __name__ == "__main__":
    main()
