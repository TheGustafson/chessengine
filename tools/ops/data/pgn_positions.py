#!/usr/bin/env python3
"""tools/ops/data/pgn_positions.py — stage 2: PGN games -> candidate positions.

PURPOSE
  Walk the PGNs written by gen_games.sh and emit one row per CANDIDATE training
  position, carrying everything a later stage needs to (a) label it, (b) join a
  real WDL column, and (c) trace it back to the game it came from.

  This is the stage that decides WHAT the corpus is a sample OF. It does not
  label and it does not filter on score — the D46 quiet filter needs the raw
  oracle score and the oracle best move, neither of which exists yet, so it runs
  at the LABEL layer (contract §1: "filter at the layer that still has raw
  scores + best moves").

USAGE
  .venv/bin/python tools/ops/data/pgn_positions.py \
      data/nnue/selfplay/<tag>/games.pgn --out <tag>/positions.tsv [--per-game 12]

OUTPUT  TSV with a header line, one row per candidate position:
    fen  ply  result  stm  played  own_cp  plies_to_end  termination  game_id

    result        WHITE-relative game result 1.0/0.5/0.0 (the WDL column)
    ply           ABSOLUTE game ply, derived from the FEN header's fullmove
                  counter, NOT from the start of the PGN movetext. The D46
                  ply>=16 criterion is about how far into a GAME a position is;
                  a book position at move 8 is already ply 14.
    own_cp        the mover's own printed eval from the fastchess comment, in
                  WHITE-relative ARCHON-frame cp (see the frame note below).
                  Diagnostic only — never a training label.
    plies_to_end  distance to the end of the game. Lets a study ask whether
                  positions near an adjudication carry different label quality.
    termination   'normal' | 'adjudication' — adjudicated results are the
                  engines' OPINION, not a played-out fact, so any WDL study
                  must be able to slice them off.

SAMPLING POLICY — why uniform-across-the-game
  Positions are sampled at evenly spaced indices across each game's eligible
  plies, so game phases appear in the proportion we actually play them. The
  receipt is tools/forensics/build_oracle.py: the previous oracle sampled
  without this discipline, came out 100% >=21-piece positions, and was blind to
  82% of our damage. Consecutive plies are also near-duplicates in feature
  space, so taking every ply would buy correlated rows, not information.

FRAME
  fastchess comments hold the mover's PRINTED score in PAWNS and from the SIDE
  TO MOVE's perspective. ARCHON prints ~2.07x Stockfish's printed cp
  (uciprobe.ARCHON_PER_SF_PRINTED), so the same comment means different things
  depending on which engine moved. This tool therefore emits own_cp in the
  frame of WHOEVER MOVED and records the mover's engine name in the manifest;
  it never mixes the two into one number. Cross-engine comparisons happen at
  the study layer, through uciprobe.to_sf_frame().
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import chess
import chess.pgn

# fastchess move comment: "{+2.25/12 0.030s, n=18696}"  or  "{-M12/25 0.1s}"
# python-chess strips the braces, so we match the bare body.
EVAL_RE = re.compile(r"^([+-]?)(M?\d+(?:\.\d+)?)/(\d+)")


def parse_comment_cp(comment: str, white_to_move: bool):
    """Printed pawns -> WHITE-relative printed centipawns, in the MOVER's frame.

    Returns None when the comment carries no score (book moves, or a mate score
    we deliberately refuse to turn into a number: a mate is not on the same
    scale as a cp value and averaging the two is how 86k mate labels once got
    into an aux corpus).
    """
    if not comment:
        return None
    m = EVAL_RE.match(comment.strip())
    if not m:
        return None
    sign, body, _depth = m.groups()
    if body.startswith("M"):
        return None
    cp = round(float(body) * 100)
    if sign == "-":
        cp = -cp
    return cp if white_to_move else -cp


def initial_ply(board: chess.Board) -> int:
    """Absolute game ply of a position, from its own FEN counters."""
    return 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)


RESULT_MAP = {"1-0": "1.0", "0-1": "0.0", "1/2-1/2": "0.5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-game", type=int, default=12,
                    help="positions sampled per game, evenly spaced (0 = all)")
    ap.add_argument("--min-ply", type=int, default=16,
                    help="skip plies below this ABSOLUTE game ply (D46 uses 16)")
    ap.add_argument("--drop-last", type=int, default=0,
                    help="drop the final N plies of each game")
    ap.add_argument("--dedup", action="store_true", default=True)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    args = ap.parse_args()

    seen = set()
    n_games = n_cand = n_kept = n_nores = 0
    # Explicit D46 accounting: this stage applies the two criteria that need
    # neither a score nor a best move (ply, in-check), purely so we do not pay
    # an oracle search for a position that is already disqualified. The counts
    # are recorded so the FULL five-criterion drop rate is reconstructable by
    # adding these to stage 3's — otherwise stage 3's rate looks deceptively
    # low and the "did the filter run?" self-check loses its meaning.
    n_drop_ply = n_drop_check = 0
    term_counts: dict[str, int] = {}
    out = open(args.out, "w")
    out.write("fen\tply\tresult\tstm\tplayed\town_cp\tplies_to_end\ttermination\tgame_id\n")

    for path in args.pgn:
        tag = Path(path).parent.name
        with open(path) as fh:
            gi = 0
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                gi += 1
                n_games += 1
                res = RESULT_MAP.get(game.headers.get("Result", "*"))
                if res is None:                      # unfinished game: no WDL
                    n_nores += 1
                    continue
                term = game.headers.get("Termination", "?")
                term_counts[term] = term_counts.get(term, 0) + 1
                gid = f"{tag}:{gi}"

                board = game.board()
                base = initial_ply(board)
                rows = []
                for node in game.mainline():
                    ply = initial_ply(board)
                    mv = node.move
                    # A position is a candidate only if it is a legal, non-
                    # terminal position with a move actually played from it.
                    if ply < args.min_ply:
                        n_drop_ply += 1
                    elif board.is_check():
                        n_drop_check += 1
                    else:
                        rows.append((board.fen(), ply, board.turn,
                                     mv.uci(),
                                     parse_comment_cp(node.comment,
                                                      board.turn == chess.WHITE)))
                    board.push(mv)
                end_ply = initial_ply(board)
                if args.drop_last:
                    rows = [r for r in rows if end_ply - r[1] > args.drop_last]
                n_cand += len(rows)

                # Even spacing across the game's eligible plies.
                if args.per_game and len(rows) > args.per_game:
                    step = len(rows) / args.per_game
                    rows = [rows[int(i * step)] for i in range(args.per_game)]

                for fen, ply, turn, played, own_cp in rows:
                    if args.dedup:
                        key = " ".join(fen.split()[:4])
                        if key in seen:
                            continue
                        seen.add(key)
                    out.write(f"{fen}\t{ply}\t{res}\t"
                              f"{'w' if turn == chess.WHITE else 'b'}\t{played}\t"
                              f"{'' if own_cp is None else own_cp}\t"
                              f"{end_ply - ply}\t{term}\t{gid}\n")
                    n_kept += 1
                _ = base
    out.close()

    plies_seen = n_cand + n_drop_ply + n_drop_check
    stats = dict(games=n_games, no_result=n_nores,
                 plies_seen=plies_seen, candidates=n_cand,
                 d46_stage2_drops=dict(ply=n_drop_ply, check=n_drop_check),
                 d46_stage2_drop_pct=round(
                     100.0 * (n_drop_ply + n_drop_check) / max(plies_seen, 1), 2),
                 kept=n_kept, unique=len(seen) if args.dedup else None,
                 per_game=args.per_game, min_ply=args.min_ply,
                 terminations=term_counts,
                 kept_per_game=round(n_kept / max(n_games, 1), 2))
    sys.stderr.write(json.dumps(stats, indent=2) + "\n")
    sp = Path(args.out).with_suffix(".stats.json")
    sp.write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
