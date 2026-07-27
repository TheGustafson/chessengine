#!/usr/bin/env python3
"""Turn lost games into a targeted position corpus — and price the mistake.

WHY. Every screen so far has judged tunables on a generic position set. But
the positions that decide games are not generic: they are the handful where
we actually went wrong. This walks a match PGN, finds the move that lost each
game, and emits those positions as a corpus — so a tunable can be screened
where it would have mattered instead of on average.

It extends `loss_forensics.py`, which classifies losses from the PGN's eval
trail but never looks at a board. Without the position you can say "this was
a misjudgment" and nothing more; with it you can ask what we played, what was
best, and whether a candidate configuration finds the better move.

TWO CORRECTIONS THIS TOOL MAKES THAT THE TRAIL-ONLY VERSION CANNOT:

1. **Eval frames differ.** ARCHON reports RAW INTERNAL cp over UCI, which
   measures ~0.64x Stockfish's frame (`eval-frame-scale-model-2026-07-27.md`
   section 11). A threshold of "-0.5 means we thought we were fine" is
   therefore ~-0.78 real pawns for us and -0.5 for them. Applying one
   threshold to both engines mis-classifies every close call. `--our-scale`
   converts our evals to the opponent's frame before any comparison.

2. **Depth is not comparable across engines.** Measured in these very games,
   ARCHON reports depth 19-25 while SF18 reports 14-19 on the same positions
   -- and loses. Our plies are cheaper: we prune harder, so a ply means less.
   Any "we searched shallower" conclusion drawn from the depth field is
   backwards here. Time and nodes are the honest axes; depth is recorded but
   flagged.

OUTPUT
  --fens FILE   positions at the point of no return, one FEN per line, ready
                for `obs_to_oracle.py`/`screen.py` as a loss-derived corpus
  --json FILE   per-loss detail: the FEN, the move we played, both engines'
                evals in a COMMON frame, the disagreement, phase and clocks

    python3 tools/loss_positions.py games.pgn --us archon --our-scale 0.64 \
        --fens loss-positions.txt --json losses.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics as st

import chess
import chess.pgn

# `eval/depth time` — python-chess STRIPS the surrounding braces from a PGN
# comment, so matching on `\{...\}` silently finds nothing and every game
# parses as "unclassified". Anchor on the eval/depth/time shape instead.
# Eval may be a mate score (+M3 / -M12).
ANNOT = re.compile(r"([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s")


def parse_eval(tok: str) -> float:
    """Annotation eval in pawns; mates clipped to +-20 so one game cannot
    dominate a mean and the ordering stays correct."""
    if "M" in tok.upper():
        sign = -1.0 if tok.lstrip().startswith("-") else 1.0
        return sign * 20.0
    return float(tok)


def games(path: str):
    with open(path, errors="ignore") as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            yield g


def phase_of(fullmove: int) -> str:
    return "opening" if fullmove <= 12 else ("middlegame" if fullmove <= 32 else "endgame")


def analyse(path: str, us: str, our_scale: float, pnr_hi: float, pnr_lo: float):
    losses, n_games, n_loss, n_win, n_draw = [], 0, 0, 0, 0
    for game in games(path):
        n_games += 1
        hdr = game.headers
        white = hdr.get("White", "")
        our_white = us.lower() in white.lower()
        res = hdr.get("Result", "*")
        if res == "1/2-1/2":
            n_draw += 1
            continue
        if res not in ("1-0", "0-1"):
            continue
        we_won = (res == "1-0") == our_white
        if we_won:
            n_win += 1
            continue
        n_loss += 1

        # Walk the game once, recording (board-before-move, move, eval, depth,
        # time, mover) so the position at any ply is available later.
        trail = []
        board = game.board()
        node = game
        while node.variations:
            node = node.variation(0)
            m = ANNOT.search(node.comment or "")
            ev = dep = tm = None
            if m:
                ev, dep, tm = parse_eval(m.group(1)), int(m.group(2)), float(m.group(3))
            mover_is_us = (board.turn == chess.WHITE) == our_white
            trail.append({
                "fen": board.fen(), "move": node.move.uci(),
                "san": board.san(node.move),
                "eval": ev, "depth": dep, "time": tm,
                "ours": mover_is_us, "fullmove": board.fullmove_number,
            })
            board.push(node.move)

        # Winner's eval trail, in the WINNER's frame already (it is the
        # opponent, so no rescale). Point of no return: first ply where it
        # reaches pnr_hi and never falls back below pnr_lo.
        theirs = [t for t in trail if not t["ours"] and t["eval"] is not None]
        ours = [t for t in trail if t["ours"] and t["eval"] is not None]
        if not theirs or not ours:
            continue
        w = [t["eval"] for t in theirs]
        pnr = next((i for i, e in enumerate(w)
                    if e >= pnr_hi and all(x >= pnr_lo for x in w[i:])), None)
        if pnr is None:
            losses.append({"round": hdr.get("Round", "?"), "tags": ["unclassified"]})
            continue

        # OUR move immediately before the winner's pnr evaluation: that is the
        # decision that let it happen, and its position is what we want.
        our_before = [t for t in ours if t["fullmove"] <= theirs[pnr]["fullmove"]]
        if not our_before:
            continue
        blunder = our_before[-1]

        # Common frame: convert our eval into theirs before comparing.
        our_ev_common = blunder["eval"] * (1.0 / our_scale) if our_scale else blunder["eval"]
        their_ev = theirs[pnr]["eval"]          # their POV, positive = good for them
        # Our eval is our POV; flip theirs to our POV to measure disagreement.
        disagreement = our_ev_common - (-their_ev)

        tags = []
        if our_ev_common >= -0.5:
            tags.append("misjudgment")
        after = [t["eval"] for t in ours if t["fullmove"] > blunder["fullmove"]][:3]
        if after and (blunder["eval"] - min(after)) * (1.0 / our_scale) >= 1.5:
            tags.append("tactical")
        first07 = next((i for i, e in enumerate(w) if e >= 0.7), pnr)
        if pnr - first07 >= 6 and "tactical" not in tags:
            tags.append("grind")
        if not tags:
            tags.append("collapse")

        losses.append({
            "round": hdr.get("Round", "?"),
            "fen": blunder["fen"],
            "our_move": blunder["move"],
            "our_san": blunder["san"],
            "fullmove": blunder["fullmove"],
            "phase": phase_of(blunder["fullmove"]),
            "our_eval_raw": blunder["eval"],
            "our_eval_common": round(our_ev_common, 2),
            "their_eval": their_ev,
            "disagreement": round(disagreement, 2),
            "our_depth": blunder["depth"],
            "their_depth": theirs[pnr]["depth"],
            "our_time_s": blunder["time"],
            "their_time_s": theirs[pnr]["time"],
            "tags": tags,
            "term": hdr.get("Termination", ""),
        })
    return losses, {"games": n_games, "wins": n_win, "draws": n_draw, "losses": n_loss}


def eval_bias(path: str, us: str, our_scale: float):
    """Our eval vs the opponent's, on EVERY move of EVERY game.

    Measuring optimism only at the point of no return is selection-biased:
    the winner sees the win first by definition, so we would look optimistic
    there even with a perfect eval. Both engines annotate every move they
    play, so the same position is never scored by both -- but across a long
    match the two eval streams cover the same distribution, and comparing
    consecutive plies (our eval, then their reply's eval negated) gives a
    per-ply disagreement that is not conditioned on the outcome.

    Positive = we think we are BETTER than the opponent thinks we are.
    """
    pairs, by_phase = [], {}
    for game in games(path):
        our_white = us.lower() in game.headers.get("White", "").lower()
        board, node, prev = game.board(), game, None
        while node.variations:
            node = node.variation(0)
            m = ANNOT.search(node.comment or "")
            mover_is_us = (board.turn == chess.WHITE) == our_white
            if m:
                ev = parse_eval(m.group(1))
                if mover_is_us:
                    prev = (ev * (1.0 / our_scale) if our_scale else ev,
                            board.fullmove_number)
                elif prev is not None:
                    # Their eval is their POV; negated it is our POV for
                    # essentially the same position, one ply later.
                    d = prev[0] - (-ev)
                    pairs.append(d)
                    by_phase.setdefault(phase_of(prev[1]), []).append(d)
                    prev = None
            board.push(node.move)
    return pairs, by_phase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--us", default="archon")
    ap.add_argument("--our-scale", type=float, default=0.64,
                    help="our eval frame relative to the opponent's; our UCI "
                    "cp is RAW INTERNAL and measures ~0.64x SF18's")
    ap.add_argument("--pnr-hi", type=float, default=2.0)
    ap.add_argument("--pnr-lo", type=float, default=1.0)
    ap.add_argument("--fens")
    ap.add_argument("--json")
    args = ap.parse_args()

    losses, tally = analyse(args.pgn, args.us, args.our_scale, args.pnr_hi, args.pnr_lo)
    cls = [l for l in losses if "fen" in l]
    print(f"{tally['games']} games: {tally['wins']}W {tally['draws']}D "
          f"{tally['losses']}L -> {len(cls)} losses localised\n")

    if cls:
        from collections import Counter
        tags = Counter(t for l in cls for t in l["tags"])
        print("loss taxonomy:")
        for t, c in tags.most_common():
            print(f"  {t:<14} {c:>3}  ({c/len(cls):.0%})")
        print("\nphase of the losing move:")
        for p, c in Counter(l["phase"] for l in cls).most_common():
            print(f"  {p:<14} {c:>3}  ({c/len(cls):.0%})")
        dis = [l["disagreement"] for l in cls]
        print(f"\neval disagreement at the losing move (our POV, common frame):")
        print(f"  median {st.median(dis):+.2f} pawns   mean {st.fmean(dis):+.2f}")
        print("  positive = we thought we were BETTER than we were")
        od = [l["our_depth"] for l in cls if l["our_depth"]]
        td = [l["their_depth"] for l in cls if l["their_depth"]]
        if od and td:
            print(f"\ndepth at the losing move: ours {st.median(od):.0f} vs "
                  f"theirs {st.median(td):.0f}")
            print("  NOTE: not comparable across engines -- our plies are "
                  "cheaper (we prune harder).")

    pairs, by_phase = eval_bias(args.pgn, args.us, args.our_scale)
    if pairs:
        print(f"\neval bias across ALL {len(pairs)} move pairs (not just losses,")
        print(" so free of the point-of-no-return selection bias):")
        print(f"  median {st.median(pairs):+.2f} pawns   mean {st.fmean(pairs):+.2f}")
        for ph in ("opening", "middlegame", "endgame"):
            v = by_phase.get(ph)
            if v:
                print(f"  {ph:<12} median {st.median(v):+.2f}  (n={len(v)})")
        print("  positive = we think we are BETTER than the opponent does")

    if args.fens and cls:
        with open(args.fens, "w") as fh:
            for l in cls:
                fh.write(l["fen"] + "\n")
        print(f"\nwrote {len(cls)} loss positions -> {args.fens}")
    if args.json:
        json.dump({"tally": tally, "losses": losses}, open(args.json, "w"), indent=1)
        print(f"wrote detail -> {args.json}")


if __name__ == "__main__":
    main()
