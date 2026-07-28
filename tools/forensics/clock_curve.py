"""Reconstruct both sides' clocks from a match PGN, per game.

WHY per game and not per move-number.  Averaging time-spent at move N across
games of different lengths and cumulating it gives a NEGATIVE time-remaining
column -- which is impossible, and is proof of the method rather than of a
forfeit. The game set changes with N, so the sum of means is not the mean of
sums. Simulate each game's clock and average the remaining time only over the
games that actually reached that move.

Use it to arm-verify any time-management change: a TM arm that does not move this
curve is inert, and a 700-round match on an inert arm is a clean, expensive null.
It also reports depth by move bucket, because the point of holding clock back is
to convert it into plies where they matter.

    .venv/bin/python tools/forensics/clock_curve.py PGN --a arm --b base
        [--tc 10+0.1]
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import statistics
import sys

ANNOT = re.compile(r"([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--a", required=True, help="substring of the arm's engine name")
    ap.add_argument("--b", required=True, help="substring of the baseline's name")
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--min-games", type=int, default=20)
    args = ap.parse_args()

    base_s, _, inc_s = args.tc.partition("+")
    BASE, INC = float(base_s), float(inc_s or 0)

    import chess
    import chess.pgn

    left = collections.defaultdict(lambda: collections.defaultdict(list))
    dep = collections.defaultdict(lambda: collections.defaultdict(list))
    low = collections.Counter()
    tot = collections.Counter()
    wdl = collections.Counter()
    games = 0

    with open(args.pgn, errors="ignore") as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            w = g.headers.get("White", "")
            b = g.headers.get("Black", "")
            if args.a in w:
                a_white = True
            elif args.a in b:
                a_white = False
            else:
                continue
            games += 1
            r = g.headers.get("Result", "*")
            if r == "1/2-1/2":
                wdl["D"] += 1
            elif (r == "1-0") == a_white:
                wdl["W"] += 1
            else:
                wdl["L"] += 1
            clk = {"a": BASE, "b": BASE}
            node = g
            while node.variations:
                node = node.variations[0]
                m = ANNOT.search(node.comment or "")
                if not m:
                    continue
                pb = node.parent.board()
                who = "a" if ((pb.turn == chess.WHITE) == a_white) else "b"
                clk[who] = clk[who] - float(m.group(3)) + INC
                mv = pb.fullmove_number
                left[who][mv].append(clk[who])
                dep[who][mv].append(int(m.group(2)))
                tot[who] += 1
                if clk[who] < 1.0:
                    low[who] += 1

    if not games:
        raise SystemExit(f"no games matched --a {args.a!r}")
    n = sum(wdl.values())
    s = (wdl["W"] + 0.5 * wdl["D"]) / n
    var = (wdl["W"] * (1 - s) ** 2 + wdl["D"] * (0.5 - s) ** 2
           + wdl["L"] * (0 - s) ** 2) / n
    sd = math.sqrt(var / n)
    elo = -400 * math.log10(1 / s - 1) if 0 < s < 1 else 0.0
    dsd = 400 / (math.log(10) * s * (1 - s)) * sd if 0 < s < 1 else 0.0
    print(f"{games} games, tc={args.tc}")
    print(f"arm W{wdl['W']} D{wdl['D']} L{wdl['L']}  {100*s:.1f}%  "
          f"Elo {elo:+.1f} +- {dsd:.1f}\n")

    print(f"{'move':>5}{'n':>6}{'arm left':>10}{'base left':>11}{'diff':>8}"
          f"{'arm dep':>9}{'base dep':>10}{'ddep':>7}")
    for mv in (10, 20, 25, 30, 33, 36, 40, 45, 50, 60, 70):
        A, B = left["a"].get(mv), left["b"].get(mv)
        if not A or not B or len(A) < args.min_games:
            continue
        da, db = statistics.mean(dep["a"][mv]), statistics.mean(dep["b"][mv])
        print(f"{mv:>5}{len(A):>6}{statistics.mean(A):>10.2f}"
              f"{statistics.mean(B):>11.2f}{statistics.mean(A)-statistics.mean(B):>+8.2f}"
              f"{da:>9.1f}{db:>10.1f}{da-db:>+7.2f}")
    print()
    for k, lab in (("a", "arm"), ("b", "base")):
        print(f"{lab:<5} {100*low[k]/max(1,tot[k]):.1f}% of moves played with <1.0s left")
    print("\nA TM arm intended to hold clock back must show a POSITIVE diff column")
    print("and, if the thesis is right, a positive ddep at the late moves.")


if __name__ == "__main__":
    main()
