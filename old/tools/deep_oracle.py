#!/usr/bin/env python3
"""Build a deep, uncensored oracle over an arbitrary FEN list.

WHY, in one paragraph. `screen.py` measures mean centipawn loss as agreement
with an oracle, and the oracle we have tops out at 64k nodes per move. Its own
error, measured by scoring each tier's best move under the deepest tier, is
~1.9 cp by extrapolation (`screen.py --oracle-audit`). That floor sits inside
every screen and cannot be searched away, so as an arm approaches it every
delta compresses and the instrument stops resolving. A deeper tier is the only
real fix.

WHY searchmoves RATHER THAN MultiPV. `move_quality.py --build-oracle` asks for
the top K lines, so any move outside K has an unknown loss that gets floored --
and the moves that fall outside K are the bad ones, so the censoring flatters
exactly the arms that deserve to look worst. Restricting the root to one move
at a time gives EVERY legal move its own full-budget search, which is both
uncensored and what the `.ceobs` corpus already does, so the output slots into
the same tiered format `obs_to_oracle.py` produces and `screen.py --oracle`
consumes.

It is also far cheaper than it sounds, because the budget is per move and the
position count is small: a 135-position loss corpus at ~35 legal moves and
256k nodes is ~1.2 G nodes, about a quarter hour of one core.

    python3 tools/deep_oracle.py data/losscorpus/lossmine-0727.fens \
        --tiers 64000,256000 --out tools/nnue/calib/oracle-loss.json
    python3 tools/screen.py --oracle tools/nnue/calib/oracle-loss.json \
        --nodes 25000 'arm:PARAM=V'

Screening on a loss-derived corpus asks a sharper question than screening on
random positions: not "is this arm better on average" but "is it better where
we actually lost games". Pair it with a deep tier and the two main weaknesses
of the standard screen -- generic positions, a saturated judge -- are both
addressed at once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import chess
    import chess.engine
except ModuleNotFoundError:  # pragma: no cover
    _venv = REPO / ".venv" / "bin" / "python3"
    if _venv.exists() and Path(sys.executable).resolve() != _venv.resolve():
        os.execv(str(_venv), [str(_venv), *sys.argv])
    raise

SF18 = "engines/bin/stockfish18"


def score_position(eng, board: chess.Board, nodes: int) -> dict[str, int]:
    """Every legal move, scored by its own root-restricted search.

    Scores are root-relative from the side to move, matching the `.ceobs`
    `arm_mode: searchmoves` convention, so loss for a played move is
    `max(scores) - scores[played]` exactly as `obs_to_oracle.py` intends.
    """
    out: dict[str, int] = {}
    for mv in board.legal_moves:
        info = eng.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[mv])
        sc = info["score"].relative
        # Mates are clipped rather than dropped: a mate score is a real
        # ordering signal, and dropping the move would silently censor exactly
        # the decisive ones. 20000 keeps it above every material score while
        # staying inside i32 arithmetic downstream.
        out[mv.uci()] = sc.score(mate_score=20000)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fens")
    ap.add_argument("--tiers", default="64000,256000")
    ap.add_argument("--out", default="tools/nnue/calib/oracle-deep.json")
    ap.add_argument("--engine", default=SF18)
    ap.add_argument("--hash", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N positions only")
    args = ap.parse_args()

    os.chdir(REPO)
    fens = [l.strip() for l in open(args.fens) if l.strip()]
    if args.limit:
        fens = fens[: args.limit]
    tiers = [int(t) for t in args.tiers.split(",")]

    boards = []
    for f in fens:
        try:
            b = chess.Board(f)
        except ValueError:
            continue
        if b.is_game_over() or not any(b.legal_moves):
            continue
        boards.append((f, b))
    moves_total = sum(b.legal_moves.count() for _, b in boards)
    print(f"{len(boards)} positions, {moves_total} legal moves, tiers {tiers}")
    print(f"budget ~{sum(tiers) * moves_total / 1e9:.2f} G nodes total")

    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    result: dict[str, dict[str, dict[str, int]]] = {str(t): {} for t in tiers}
    try:
        eng.configure({"Threads": 1, "Hash": args.hash})
        for t in tiers:
            t0 = time.time()
            for i, (fen, board) in enumerate(boards, 1):
                result[str(t)][fen] = score_position(eng, board, t)
                if i % 20 == 0 or i == len(boards):
                    el = time.time() - t0
                    print(f"  tier {t}: {i}/{len(boards)}  "
                          f"{el:.0f}s elapsed, ~{el / i * (len(boards) - i):.0f}s left",
                          flush=True)
    finally:
        eng.quit()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"source": f"deep_oracle.py {args.fens} tiers={args.tiers} "
                             f"engine={os.path.basename(args.engine)}",
                   "tiers": result}, fh)
    print(f"\nwrote {args.out}")
    print(f"audit it:  python3 tools/screen.py --oracle {args.out} --oracle-audit")
    print("  (the deepest tier reads 0.00 by construction; what matters is how")
    print("   much the SHALLOWER tiers lose against it -- that ratio is what")
    print("   estimates the deepest tier's own remaining error.)")


if __name__ == "__main__":
    main()
