#!/usr/bin/env python3
"""Derive an ARCHON-original 49-position bench suite from ARCHON's own games.

2026-08 originality audit, item 6b (docs/audit/2026-08-originality/
00-synthesis.md §2; foundation-infra.md F2): the shipped BENCH_POSITIONS is a
49/49 byte-identical subsequence of Stockfish's benchmark.cpp Defaults. This
tool builds a replacement from positions ARCHON actually played, so the suite
is our data end to end.

Composition (mirrors the SF suite's phase mix without reusing any position):
    10 opening        (29..=32 pieces)
    18 middlegame     (17..=28 pieces)
    14 endgame        ( 8..=16 pieces)
     5 late endgame   ( 3..= 7 pieces)
     2 known-hard     (the first two entries of searchopt/positions/hard.json,
                       diagnosed real ARCHON losses)
    -- 49 total

Selection is DETERMINISTIC, no tuning knobs:
  * games are read in sorted path order from the given PGNs (ARCHON was a
    player in every game; the [FEN] header seeds each replay — the book-FEN
    trap rule);
  * every position after ply 6 of a game is a candidate for its piece-count
    phase, provided the side to move has a legal move;
  * within a phase, candidates are ordered by sha256(fen) — a fixed,
    content-derived shuffle — and taken greedily subject to: at most one
    position per source game (across the whole suite), and side-to-move
    alternation inside the phase (so the suite cannot skew white);
  * fullmove/halfmove counters ride along in the FEN unchanged.

Output: the 49 FEN lines, plus a ready-to-paste Rust slice body.

Usage (from the repo root):
    python3 tools/bench/derive_bench_suite.py \
        'data/nnue/selfplay/sfdev5k-a-c*/games.pgn' [--max-games 4000]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import chess
import chess.pgn

PHASES = (  # (name, quota, min_pieces, max_pieces)
    ("opening", 10, 29, 32),
    ("middlegame", 18, 17, 28),
    ("endgame", 14, 8, 16),
    ("late-endgame", 5, 3, 7),
)
HARD_COUNT = 2
MIN_PLY = 6


def candidates_from_pgns(paths: list[Path], max_games: int):
    """Yield (game_id, fen, piece_count, stm_white) for every candidate."""
    game_id = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while game_id < max_games:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                board = game.board()  # honors the [FEN] header
                ply = 0
                for move in game.mainline_moves():
                    board.push(move)
                    ply += 1
                    if ply < MIN_PLY:
                        continue
                    if board.is_game_over():
                        break
                    yield (game_id, board.fen(), len(board.piece_map()),
                           board.turn == chess.WHITE)
                game_id += 1
        if game_id >= max_games:
            break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pgn_glob")
    ap.add_argument("--max-games", type=int, default=4000)
    ap.add_argument("--hard-json", type=Path,
                    default=Path("searchopt/positions/hard.json"))
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.pgn_glob))
    if not paths:
        raise SystemExit(f"no PGNs match {args.pgn_glob!r}")

    by_phase: dict[str, list] = {name: [] for name, *_ in PHASES}
    for game_id, fen, pieces, stm_white in candidates_from_pgns(
            paths, args.max_games):
        for name, _quota, lo, hi in PHASES:
            if lo <= pieces <= hi:
                by_phase[name].append((game_id, fen, stm_white))
                break

    used_games: set[int] = set()
    seen_fens: set[str] = set()
    suite: list[tuple[str, str]] = []  # (phase, fen)
    for name, quota, _lo, _hi in PHASES:
        pool = sorted(by_phase[name],
                      key=lambda c: hashlib.sha256(c[1].encode()).hexdigest())
        want_white = True
        taken = 0
        # Two passes: strict stm alternation first, then fill any shortfall
        # ignoring stm (tiny phases on small games files).
        for strict in (True, False):
            for game_id, fen, stm_white in pool:
                if taken >= quota:
                    break
                if game_id in used_games or fen in seen_fens:
                    continue
                if strict and stm_white != want_white:
                    continue
                suite.append((name, fen))
                used_games.add(game_id)
                seen_fens.add(fen)
                want_white = not stm_white
                taken += 1
        if taken < quota:
            raise SystemExit(
                f"phase {name}: only {taken}/{quota} positions found — "
                f"widen --max-games or the PGN glob")

    hard = json.loads(args.hard_json.read_text())[:HARD_COUNT]
    for entry in hard:
        suite.append(("known-hard", entry["fen"]))

    assert len(suite) == sum(q for _, q, *_ in PHASES) + HARD_COUNT == 49

    print(f"# derived from {len(paths)} PGN files, "
          f"first {args.max_games} games, by tools/bench/derive_bench_suite.py")
    for phase, fen in suite:
        print(f"{fen}  # {phase}")
    print()
    print("// ---- Rust slice body (src/bench_positions_archon.rs) ----")
    for phase, fen in suite:
        print(f'    "{fen}", // {phase}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
