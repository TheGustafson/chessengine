#!/usr/bin/env python3
"""Walk ARCHON's and Stockfish's searches side by side on one position.

Aggregate statistics said our net agrees with SF18 everywhere, our searched
eval agrees on ordinary positions, and yet on the positions that lose games
we are ~1.5 pawns adrift with a 3-4x wider tree. Averages cannot say WHY. This
prints, per iteration, what each engine believes and how it got there:

  * best move and eval at every depth, in a common frame
  * the PV, so a divergence can be read as a concrete line
  * the depth at which each engine FIRST finds the move the other ends on
  * where OUR move ordering ranks the opponent's final choice

The last item is the actionable one. If we eventually agree with SF18's move
but ordered it late, the defect is move ORDERING, not evaluation or pruning —
and ordering is what determines how much of the tree we can safely skip. Our
first-move cutoff rate is 89.4% against Stockfish's typical 93-95%, and that
gap compounds with depth into exactly the width ratio measured here.

    python3 tools/tree_walk.py "FEN" --max-depth 16
    python3 tools/tree_walk.py --from-json losses.json --index 0
"""

from __future__ import annotations

import argparse
import json

import chess
import chess.engine

ARCHON = "target/release/chessengine"
NET = "data/nnue/nets/screens/d47v-a4b2-cal400.cenn"
SF18 = "engines/bin/stockfish18"
OUR_SCALE = 0.64          # our UCI cp is raw internal, ~0.64x SF18's frame


def ladder(binary, fen, max_depth, options=None, scale=1.0):
    """{depth: (eval_common_frame, best_move_uci, pv_san, nodes)}"""
    eng = chess.engine.SimpleEngine.popen_uci(binary)
    out = {}
    try:
        for k, v in (options or {}).items():
            eng.configure({k: v})
        board = chess.Board(fen)
        for d in range(4, max_depth + 1):
            info = eng.analyse(board, chess.engine.Limit(depth=d))
            pv = info.get("pv") or []
            sc = info["score"].relative
            cp = sc.score()
            ev = None if cp is None else (cp / 100.0) / scale
            san = board.variation_san(pv[:6]) if pv else ""
            out[d] = (ev, pv[0].uci() if pv else None, san, info.get("nodes"))
    finally:
        eng.quit()
    return out


def our_move_rank(fen: str, target_uci: str, depth: int) -> str:
    """Where our search ranks `target` — via MultiPV, which orders by value.

    Not the same as move-ordering position in the move loop, but it answers
    the question that matters: does our search consider this move seriously?
    """
    eng = chess.engine.SimpleEngine.popen_uci(ARCHON)
    try:
        eng.configure({"EvalFile": NET})
        try:
            eng.configure({"MultiPV": 8})
        except chess.engine.EngineError:
            return "n/a (no MultiPV)"
        infos = eng.analyse(chess.Board(fen), chess.engine.Limit(depth=depth), multipv=8)
        for i, info in enumerate(infos, 1):
            pv = info.get("pv")
            if pv and pv[0].uci() == target_uci:
                sc = info["score"].relative.score()
                return f"rank {i}/8" + (f" ({sc/100.0/OUR_SCALE:+.2f})" if sc is not None else "")
        return "outside top 8"
    finally:
        eng.quit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fen", nargs="?")
    ap.add_argument("--from-json")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=16)
    args = ap.parse_args()

    played = None
    if args.from_json:
        blob = json.load(open(args.from_json))
        cls = [l for l in blob["losses"] if "fen" in l]
        rec = cls[args.index]
        fen, played = rec["fen"], rec["our_san"]
        print(f"loss {args.index + 1}/{len(cls)}  round {rec['round']}  "
              f"move {rec['fullmove']}  tags {','.join(rec['tags'])}")
        print(f"we played {played}   our eval {rec['our_eval_common']:+.2f}   "
              f"their eval {-rec['their_eval']:+.2f} (our POV)")
    else:
        fen = args.fen
    print(f"FEN {fen}\n")

    ours = ladder(ARCHON, fen, args.max_depth, {"EvalFile": NET}, OUR_SCALE)
    theirs = ladder(SF18, fen, args.max_depth)

    print(f"{'d':>3} | {'ARCHON':>7} {'move':>6} {'nodes':>9} | "
          f"{'SF18':>7} {'move':>6} {'nodes':>8} | agree")
    for d in sorted(ours):
        oe, om, _, on = ours[d]
        te, tm, _, tn = theirs.get(d, (None, None, "", None))
        agree = "yes" if (om and tm and om == tm) else ""
        print(f"{d:>3} | {oe if oe is None else f'{oe:+7.2f}'} {om or '-':>6} "
              f"{on or 0:>9,} | {te if te is None else f'{te:+7.2f}'} {tm or '-':>6} "
              f"{tn or 0:>8,} | {agree}")

    fin_o = ours[max(ours)][1]
    fin_t = theirs[max(theirs)][1]
    print(f"\nfinal: ours {fin_o}   theirs {fin_t}")
    print(f"  our PV    {ours[max(ours)][2]}")
    print(f"  their PV  {theirs[max(theirs)][2]}")
    if fin_o != fin_t:
        first = next((d for d in sorted(ours) if ours[d][1] == fin_t), None)
        print(f"\nWe never settle on their move." if first is None
              else f"\nWe held their move at depth {first} then abandoned it.")
        print(f"  our search's opinion of {fin_t}: "
              f"{our_move_rank(fen, fin_t, min(12, args.max_depth))}")
        print("  If it ranks well but we still play something else, the defect is")
        print("  evaluation of the resulting position; if it ranks poorly or is")
        print("  outside the top 8, the defect is ORDERING/selectivity.")


if __name__ == "__main__":
    main()
