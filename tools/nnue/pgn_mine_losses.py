#!/usr/bin/env python3
"""Mine high-value training positions from ARCHON-vs-SF18 games.

The idea (user, 2026-07-19): SF18 crushes ARCHON — that is the *value*. The
positions where ARCHON loses are exactly where our eval is most wrong, so they
carry the highest information per position for improving the net. We mine them
quickly from bulk games, then deep-label with SF18 (50k nodes) — some of the
highest-value data in the corpus.

This reads a fastchess PGN (evals tracked as `{score/depth ...}` comments) and
emits, for each game ARCHON LOST (and optionally drew-from-worse), the positions
where it is ARCHON's turn to move — i.e. the decision points that led to the
loss. It skips the opening book plies and already-decided positions, and (by
default) keeps only positions where ARCHON's *own* reported eval was not yet
losing (the blind spots: "thought it was fine, then lost").

Output is TSV `FEN\twdl` where wdl is the WHITE-relative game result
(1.0/0.5/0.0) — identical shape to the fleet's `c1_orig.tsv`, so the same
deep-label + WDL-merge + bf-convert path builds the corpus. De-duplicates FENs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import chess
import chess.pgn

# fastchess writes eval as a comment "{+0.35/12 0.123s}" -> python-chess strips
# the braces, so node.comment is "+0.35/12 0.123s". Capture the signed score
# token that precedes "/depth" (pawns like 0.35, or mate like M3).
_EVAL_RE = re.compile(r"([+-]?)(\d+\.\d+|M\d+|\d+)/\d+")


def parse_eval_cp(comment: str, side_to_move_white: bool) -> int | None:
    """White-relative centipawns from a fastchess eval comment, or None.

    fastchess reports the score from the SIDE-TO-MOVE's perspective; we return
    it White-relative so thresholds are unambiguous.
    """
    m = _EVAL_RE.search(comment or "")
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    tok = m.group(2)
    if tok.startswith("M"):
        cp = 30000  # mate: treat as a large decided magnitude
    elif "." in tok:
        cp = int(round(float(tok) * 100))
    else:
        cp = int(tok)
    stm_cp = sign * cp
    return stm_cp if side_to_move_white else -stm_cp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pgn", type=Path, help="fastchess PGN with eval comments")
    ap.add_argument("--archon-name", default="ARCHON",
                    help="engine name (PGN White/Black tag) that is our engine")
    ap.add_argument("--out", type=Path, required=True, help="TSV FEN\\twdl output")
    ap.add_argument("--min-ply", type=int, default=16,
                    help="skip the first N plies (book/opening; default 16 = 8 moves)")
    ap.add_argument("--decided-cp", type=int, default=800,
                    help="skip positions where |ARCHON eval| exceeds this (already lost/won)")
    ap.add_argument("--optimism-cp", type=int, default=-75,
                    help="keep only positions where ARCHON's own eval (its perspective) was "
                         ">= this — the blind spots it did not see coming. Set very low "
                         "(e.g. -100000) to keep every losing-game position.")
    ap.add_argument("--include-worse-draws", action="store_true",
                    help="also mine ARCHON-to-move positions from DRAWN games where ARCHON "
                         "was materially worse by its own eval")
    ap.add_argument("--max-per-game", type=int, default=0,
                    help="cap positions kept per game (0 = no cap); takes the worst-eval ones")
    args = ap.parse_args()

    if not args.pgn.is_file():
        sys.exit(f"missing PGN: {args.pgn}")

    seen: set[str] = set()
    kept = 0
    games = 0
    archon_losses = 0
    with args.pgn.open() as handle, args.out.open("w") as out:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games += 1
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            result = game.headers.get("Result", "*")
            if args.archon_name not in (white, black):
                continue
            archon_white = args.archon_name == white
            # White-relative outcome.
            wdl = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}.get(result)
            if wdl is None:
                continue
            archon_score = wdl if archon_white else 1.0 - wdl  # 1 win / .5 draw / 0 loss
            is_loss = archon_score == 0.0
            is_draw = archon_score == 0.5
            if not is_loss and not (args.include_worse_draws and is_draw):
                continue
            if is_loss:
                archon_losses += 1

            per_game: list[tuple[int, str, float]] = []  # (archon_eval, fen, wdl)
            board = game.board()
            node = game
            ply = 0
            while node.variations:
                node = node.variations[0]
                move = node.move
                mover_white = board.turn  # side to move BEFORE this move
                archon_to_move = mover_white == archon_white
                # ARCHON's reported eval for THIS position (its perspective), from
                # the comment attached to the move it is about to make.
                if archon_to_move and ply >= args.min_ply:
                    fen = board.fen()
                    wr_cp = parse_eval_cp(node.comment, mover_white)
                    if wr_cp is not None:
                        # ARCHON-perspective eval = White-relative * (white?1:-1)
                        archon_cp = wr_cp if archon_white else -wr_cp
                        if abs(archon_cp) <= args.decided_cp and archon_cp >= args.optimism_cp:
                            if fen not in seen:
                                per_game.append((archon_cp, fen, wdl))
                    elif args.optimism_cp <= -100000:
                        # no-eval fallback only when keeping everything
                        if fen not in seen:
                            per_game.append((0, fen, wdl))
                board.push(move)
                ply += 1

            if args.max_per_game > 0 and len(per_game) > args.max_per_game:
                per_game.sort(key=lambda t: t[0])  # most-optimistic-yet-lost first
                per_game = per_game[: args.max_per_game]
            for _cp, fen, w in per_game:
                if fen in seen:
                    continue
                seen.add(fen)
                out.write(f"{fen}\t{w}\n")
                kept += 1

    print(f"games={games} archon_losses={archon_losses} positions_kept={kept} "
          f"unique={len(seen)} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
