"""Does carried search state (TT + histories) degrade our move choice?

The definitive version, with the two controls that decide it.

**Why a control set is mandatory.** The error corpus selects positions where the
GAME erred, and the game carried state.  A state-carrying replay is therefore
closer to the condition that produced the error than a cold search is, and would
reproduce it more often for that reason alone -- with carried state being
perfectly harmless.  So the same comparison is run on positions where the game
played the ORACLE's move.  Selection predicts replay agrees MORE than cold there;
a real state effect predicts it agrees less.

**Why per-move depths matter.** An earlier version searched every preceding move
at the TARGET's depth, which on a game whose early moves were shallow accumulates
far more history and TT traffic than the game ever did -- inflating the effect.
This one replays each of our moves at the depth the PGN records FOR THAT MOVE, so
the state arrives in roughly the condition the game left it.

**Fidelity check.** On the error set the game played a specific wrong move.  If
the replay reproduces that move much more often than a cold search does, its
state resembles the game's and the comparison means something.  If both
reproduce it equally, the replay is a third condition and its numbers describe
the harness rather than the engine.

    .venv/bin/python tools/forensics/state_effect.py CLASSIFIED.json
        [--pgn ...] [--control 44] [--per-move-depth/--fixed-depth]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-pgo"
SF = REPO + "/engines/bin/stockfish18"
ANNOT = re.compile(r"([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s")


def load_games(patterns):
    """round -> list of (start_fen, moves, archon_white, {ply: depth}).

    A LIST because fastchess `-repeat -games 2` puts two games under one round
    number; keeping only the first silently dropped 26 of 44 rows once already.
    """
    import chess.pgn

    out = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, errors="ignore") as fh:
                while (g := chess.pgn.read_game(fh)) is not None:
                    mv, dep, node, i = [], {}, g, 0
                    while node.variations:
                        node = node.variations[0]
                        mv.append(node.move.uci())
                        m = ANNOT.search(node.comment or "")
                        if m:
                            dep[i] = int(m.group(2))
                        i += 1
                    out.setdefault(g.headers.get("Round", "?"), []).append(
                        (g.headers.get("FEN"), mv,
                         "archon" in g.headers.get("White", "").lower(), dep))
    return out


def locate(fen, cands):
    import chess

    for sf, mv, aw, dep in cands:
        b = chess.Board(sf) if sf else chess.Board()
        for i, u in enumerate(mv):
            if b.fen() == fen:
                return sf, mv, aw, dep, i
            try:
                b.push(chess.Move.from_uci(u))
            except Exception:  # noqa: BLE001
                break
    return None


class Probe:
    def __init__(self, hash_mb=16):
        self.e = Engine(OURS, net=NET, hash_mb=hash_mb, name="archon")
        self.e.assert_net_live()

    def _go(self, pos, depth):
        self.e.send(pos)
        self.e.send(f"go depth {depth}")
        while True:
            line = self.e.p.stdout.readline()
            if not line:
                return None
            if line.startswith("bestmove"):
                return line.split()[1]

    def cold(self, fen, depth):
        self.e.newgame()
        return self._go(f"position fen {fen}", depth)

    def replay(self, start_fen, moves, aw, dep, target, fallback, per_move):
        self.e.newgame()
        head = f"fen {start_fen}" if start_fen else "startpos"
        got = None
        for i in range(target + 1):
            if (i % 2 == 0) != aw:
                continue
            d = (dep.get(i, fallback) if per_move else fallback)
            pos = f"position {head}" + (" moves " + " ".join(moves[:i]) if i else "")
            got = self._go(pos, max(4, min(int(d), 30)))
        return got

    def close(self):
        self.e.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("classified")
    ap.add_argument("--pgn", action="append",
                    default=["engines/runs/corpus-v2/*/games.pgn"])
    ap.add_argument("--control", type=int, default=44)
    ap.add_argument("--fixed-depth", action="store_true",
                    help="replay every prior move at the TARGET's depth (the old, "
                         "state-inflating behaviour) instead of each move's own")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    per_move = not args.fixed_depth

    import chess
    import chess.pgn

    games = load_games(args.pgn)
    rows = [r for r in json.load(open(args.classified))
            if r["verdict"] in ("DEPTH", "DEPTH_vs_EVAL", "SEARCH_REJECTS", "EVAL")]

    err = []
    for r in rows:
        got = locate(r["fen"], games.get(str(r["game"]), []))
        if got:
            err.append((r["fen"], r["best"], r["played"], int(r["game_depth"]), *got))

    # --- control population: the game played the ORACLE's move ---------------
    pool = []
    for rnd, cands in games.items():
        for sf, mv, aw, dep in cands:
            b = chess.Board(sf) if sf else chess.Board()
            for i, u in enumerate(mv):
                if (i % 2 == 0) == aw and not b.is_check() and i in dep:
                    pool.append((b.fen(), u, dep[i], sf, mv, aw, dep, i))
                try:
                    b.push(chess.Move.from_uci(u))
                except Exception:  # noqa: BLE001
                    break
    random.Random(args.seed).shuffle(pool)
    sfe = Engine(SF, net=None, hash_mb=16, name="sf18")
    ctrl = []
    for fen, played, d, sf, mv, aw, dep, i in pool:
        if len(ctrl) >= args.control:
            break
        if d < 10 or d > 22:
            continue
        sfe.newgame()
        rr = sfe.go(fen=fen, nodes=1_000_000)
        if not rr["ladder"]:
            continue
        if rr["ladder"][max(rr["ladder"])]["move"] == played:
            ctrl.append((fen, played, played, d, sf, mv, aw, dep, i))
    sfe.close()

    p = Probe()
    print(f"replay depth policy: {'per-move (faithful)' if per_move else 'fixed at target depth'}")
    for label, data in (("ERROR set  (game played a WRONG move)", err),
                        ("CONTROL set (game played the ORACLE's move)", ctrl)):
        c_ok = r_ok = c_game = r_game = 0
        for fen, best, played, d, sf, mv, aw, dep, t in data:
            cb = p.cold(fen, d)
            rb = p.replay(sf, mv, aw, dep, t, d, per_move)
            c_ok += (cb == best)
            r_ok += (rb == best)
            c_game += (cb == played)
            r_game += (rb == played)
        n = len(data)
        print(f"\n{label}  n={n}")
        print(f"  agrees with ORACLE   cold {c_ok:>3}/{n}   replay {r_ok:>3}/{n}")
        print(f"  reproduces GAME move cold {c_game:>3}/{n}   replay {r_game:>3}/{n}"
              f"   <- fidelity", flush=True)
    p.close()
    print("\nReal state effect: replay < cold on BOTH sets.")
    print("Selection artefact:  replay >= cold on the control set.")
    print("Faithful replay:     replay reproduces the game's move far more than cold.")


if __name__ == "__main__":
    main()
