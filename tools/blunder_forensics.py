#!/usr/bin/env python3
"""Why does our own evaluation collapse 10x more often than Stockfish's?

THE OBSERVATION. Over the 120-game rematch, `match_report` counted own-eval
drops of >= 2 pawns across a side's consecutive moves: **ARCHON 52, SF18 5**.
An engine of near-parity net quality and near-parity speed should not walk into
ten times as many positions where its own assessment falls apart. That ratio is
the loudest unexplained signal we have, and no aggregate we have computed --
mean cp loss, taxonomy, depth by phase, clock trajectory -- can explain it,
because they all average over exactly the tail that produces it.

WHAT A COLLAPSE ACTUALLY MEANS is ambiguous, and the ambiguity is the whole
point. We played a move; after the reply our own eval is two pawns worse. There
are three completely different causes and they send you to three different
places:

  search_miss       the move really was bad -- a deep oracle agrees. We did not
                    see the refutation. This is SEARCH SHAPE: horizon, pruning,
                    or ordering dropped the refuting line.
  eval_instability  the move was objectively fine. Our eval was simply wrong
                    BEFORE, and the opponent's reply corrected it. This is the
                    net, not the search.
  already_lost      every move loses; the "collapse" is just the position
                    resolving. Not a mistake at all, and counting it as one
                    inflates the ledger.

Separating them needs ground truth at the position, which is what the deep
oracle is for: every legal move searched separately (`searchmoves`), so nothing
is censored and the played move always has a score.

    python3 tools/blunder_forensics.py engines/runs/rematch-v1/time/games.pgn \
        --us archon --them sf18 --nodes 256000

For a search_miss it also reports the REFUTATION -- the opponent's best reply
and whether it is a capture, a check, or a quiet move. A refutation set
dominated by quiet moves says something very different from one dominated by
captures: captures point at qsearch and the horizon, quiets point at pruning
having discarded the line before it could be seen.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import chess
    import chess.engine
    import chess.pgn
except ModuleNotFoundError:  # pragma: no cover
    _venv = REPO / ".venv" / "bin" / "python3"
    if _venv.exists() and Path(sys.executable).resolve() != _venv.resolve():
        os.execv(str(_venv), [str(_venv), *sys.argv])
    raise

# python-chess strips the braces from PGN comments, so a `\{...\}` pattern
# silently matches nothing. Anchor on the eval/depth/time shape instead.
ANNOT = re.compile(r"([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s")


def ev(tok: str) -> float:
    if "M" in tok.upper():
        return -20.0 if tok.lstrip().startswith("-") else 20.0
    return float(tok)


def wp(pawns: float) -> float:
    """Pawns -> win probability. Logistic, ~1.7 pawn scale.

    A raw-pawn threshold is WRONG for this job and the first version used one.
    +16.2 -> +12.8 is a 3.4 pawn "collapse" between two totally won positions:
    in win-probability terms it is a move from 1.000 to 0.999, i.e. nothing
    happened. Sparse endgames carry far more extreme evals than middlegames, so
    a pawn threshold manufactures blunders preferentially at low piece counts --
    which is exactly the spurious concentration the first run reported.
    """
    return 1.0 / (1.0 + math.exp(-pawns / 1.7))


def collapses(pgn: str, us: str, drop: float, scale: float):
    """Positions where OUR eval fell by `drop` pawns across our own moves.

    Mate-score transitions are excluded: a +M3 -> -3.2 swing is score encoding,
    not a two-pawn error, and they polluted the first version of this ledger.
    """
    out = []
    with open(pgn, errors="ignore") as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            us_white = us.lower() in g.headers.get("White", "").lower()
            seq, node = [], g
            while node.variations:
                node = node.variations[0]
                m = ANNOT.search(node.comment or "")
                b = node.parent.board()
                if m and (b.turn == chess.WHITE) == us_white:
                    seq.append((b.fen(), node.move, ev(m.group(1)) * scale,
                                int(m.group(2))))
            for i in range(len(seq) - 1):
                if abs(seq[i][2]) > 25 or abs(seq[i + 1][2]) > 25:
                    continue
                if wp(seq[i][2]) - wp(seq[i + 1][2]) >= drop:
                    out.append({"game": g.headers.get("Round", "?"),
                                "fen": seq[i][0], "move": seq[i][1].uci(),
                                "before": round(seq[i][2], 2),
                                "after": round(seq[i + 1][2], 2),
                                "depth": seq[i][3]})
    return out


def adjudicate(eng, rec, nodes: int) -> dict:
    """Score every legal move at `nodes` each; classify the collapse."""
    board = chess.Board(rec["fen"])
    scores = {}
    for mv in board.legal_moves:
        info = eng.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[mv])
        scores[mv.uci()] = info["score"].relative.score(mate_score=20000)
    best = max(scores.values())
    played = scores.get(rec["move"], best)
    rec["true_loss"] = best - played
    rec["best_move"] = max(scores, key=scores.get)
    rec["best_score"] = best
    # already_lost: even the best move is losing badly, so nothing was thrown
    # away here and calling it a blunder is a counting error.
    if best <= -300:
        rec["class"] = "already_lost"
    elif rec["true_loss"] >= 100:
        rec["class"] = "search_miss"
    elif rec["true_loss"] < 50:
        rec["class"] = "eval_instability"
    else:
        rec["class"] = "inaccuracy"

    if rec["class"] == "search_miss":
        board.push(chess.Move.from_uci(rec["move"]))
        reply = eng.play(board, chess.engine.Limit(nodes=nodes)).move
        if reply is not None:
            rec["refutation"] = reply.uci()
            rec["refutation_kind"] = (
                "capture" if board.is_capture(reply)
                else "check" if board.gives_check(reply) else "quiet")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--us", default="archon")
    ap.add_argument("--them", default="sf18")
    ap.add_argument("--oracle", default="engines/bin/stockfish18")
    ap.add_argument("--nodes", type=int, default=256000)
    ap.add_argument("--drop", type=float, default=0.15,
                    help="WIN-PROBABILITY drop, not pawns. 0.15 ~ giving away a "
                         "sixth of the game. A pawn threshold manufactures "
                         "blunders in blowout positions, which are commonest "
                         "when the board is sparse.")
    ap.add_argument("--our-scale", type=float, default=0.64)
    ap.add_argument("--their-scale", type=float, default=1.07)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.chdir(REPO)

    sides = [(args.us, args.our_scale), (args.them, args.their_scale)]
    eng = chess.engine.SimpleEngine.popen_uci(args.oracle)
    try:
        eng.configure({"Threads": 1, "Hash": 256})
        allrecs = {}
        for name, scale in sides:
            recs = collapses(args.pgn, name, args.drop, scale)
            if args.limit:
                recs = recs[: args.limit]
            print(f"\n{name}: {len(recs)} own-eval collapses >= {args.drop} win-prob"
                  f"  (adjudicating at {args.nodes} nodes/move)", flush=True)
            for i, r in enumerate(recs, 1):
                adjudicate(eng, r, args.nodes)
                if i % 10 == 0:
                    print(f"  {i}/{len(recs)}", flush=True)
            allrecs[name] = recs
    finally:
        eng.quit()

    print(f"\n{'engine':<10}{'n':>5}  " + "  ".join(f"{c:>17}" for c in
          ("search_miss", "eval_instability", "inaccuracy", "already_lost")))
    for name, _ in sides:
        c = Counter(r["class"] for r in allrecs[name])
        n = len(allrecs[name])
        print(f"{name:<10}{n:>5}  " + "  ".join(
            f"{c[k]:>7} ({100*c[k]/max(n,1):>4.0f}%)" for k in
            ("search_miss", "eval_instability", "inaccuracy", "already_lost")))

    for name, _ in sides:
        sm = [r for r in allrecs[name] if r["class"] == "search_miss"]
        if sm:
            k = Counter(r.get("refutation_kind", "?") for r in sm)
            print(f"\n{name} refutations ({len(sm)} search misses): {dict(k)}")
            print(f"  median true loss {sorted(r['true_loss'] for r in sm)[len(sm)//2]} cp,"
                  f" median depth reached {sorted(r['depth'] for r in sm)[len(sm)//2]}")

    print("\nsearch_miss is SEARCH SHAPE -- the move was really bad and we did not")
    print("see it. eval_instability is the NET -- the move was fine and our own")
    print("assessment was wrong beforehand. The RATIO between the two is the")
    print("finding; the raw collapse count conflates them and cannot be read.")
    print("Refutation kind narrows it further: captures point at qsearch and the")
    print("horizon, quiets point at pruning discarding the line before it was seen.")

    if args.out:
        Path(args.out).write_text(json.dumps(allrecs, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
