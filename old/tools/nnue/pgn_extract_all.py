#!/usr/bin/env python3
"""Extract EVERY position from ARCHON-vs-SF18 games (no eval/loss filtering).

Rationale (user, 2026-07-20): we are data-hungry; these on-policy positions are
expensive to generate and SF18 gives every one a truthful deep label. Don't throw
data away. So: take every board state along every game (both sides to move),
past a tiny book skip, dedup by FEN, emit FEN\\twdl (White-relative game result)
— same shape as the fleet's tsv, so the deep-label + WDL-merge + bf-convert path
is unchanged.

  pgn_extract_all.py <pgn>... --out FEN_wdl.tsv [--min-ply 8]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import chess, chess.pgn

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pgn", type=Path, nargs="+", help="fastchess PGN(s)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-ply", type=int, default=8,
                    help="skip the first N plies (immediate book duplicates; default 8)")
    args = ap.parse_args()
    seen: set[str] = set()
    games = kept = 0
    with args.out.open("w") as out:
        for pgn_path in args.pgn:
            if not pgn_path.is_file():
                print(f"skip missing {pgn_path}", file=sys.stderr); continue
            with pgn_path.open() as handle:
                while True:
                    game = chess.pgn.read_game(handle)
                    if game is None:
                        break
                    games += 1
                    wdl = {"1-0":1.0,"1/2-1/2":0.5,"0-1":0.0}.get(game.headers.get("Result","*"))
                    if wdl is None:
                        continue
                    board = game.board(); node = game; ply = 0
                    while node.variations:
                        node = node.variations[0]
                        if ply >= args.min_ply:
                            fen = board.fen()
                            if fen not in seen:
                                seen.add(fen)
                                out.write(f"{fen}\t{wdl}\n"); kept += 1
                        board.push(node.move); ply += 1
    print(f"games={games} positions_kept={kept} unique={len(seen)} -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
