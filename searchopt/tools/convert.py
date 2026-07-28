#!/usr/bin/env python3
"""Does the CONVERSION asymmetry survive at equal nodes? A paired read.

The corpus headline is "we convert a won book 12%, SF converts 84%". That is
measured at 10+0.1, where we also carry a ~1.9x speed handicap, and it is
partly a restatement of the Elo gap itself -- the stronger engine converts, the
weaker one does not, which is what Elo MEANS. To be a localisation rather than
a tautology it has to survive at equal nodes, and it has to be read PAIRED:
each position is played twice with colours swapped, so the same advantage is
handed to each engine in turn.

    P(advantaged side wins | advantaged side is ARCHON)
    P(advantaged side wins | advantaged side is SF18)

Same positions, same budget, same book. If the two are close, "cannot convert"
is not where the Elo lives and the folder should stop organising around it.

Needs `plateau-advside.json` (SF18 @ 1M, white POV) to say which side is better
-- the FEN alone does not say, and guessing from the source game's colour is
exactly the PGN-replay trap that has bitten this repo before.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

GAME = re.compile(r'\[Event[\s\S]*?(?=\[Event|\Z)')
TAG = re.compile(r'\[(\w+) "([^"]*)"\]')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="engines/runs/plateau-110k/games.pgn")
    ap.add_argument("--adv", default="searchopt/positions/plateau-advside.json")
    ap.add_argument("--us", default="archon")
    ap.add_argument("--min-cp", type=float, default=50.0,
                    help="only positions the judge calls decided by this much")
    a = ap.parse_args()

    adv = json.load(open(a.adv))
    txt = open(a.pgn, errors="ignore").read()

    # per (fen, advantaged-owner) -> list of results for that owner
    tally = {"archon": [0, 0, 0], "sf18": [0, 0, 0]}   # win / draw / loss for the advantaged side
    used = set()
    for blk in GAME.findall(txt):
        h = dict(TAG.findall(blk))
        if "FEN" not in h or h.get("Result", "*") == "*":
            continue
        fen = h["FEN"]
        if fen not in adv:
            continue
        cp = adv[fen]["cp_white"]
        if abs(cp) < a.min_cp:
            continue
        adv_white = cp > 0
        owner = h["White"] if adv_white else h["Black"]
        owner = "archon" if owner == a.us else "sf18"
        res = h["Result"]
        if res == "1/2-1/2":
            idx = 1
        else:
            adv_won = (res == "1-0") == adv_white
            idx = 0 if adv_won else 2
        tally[owner][idx] += 1
        used.add(fen)

    print(f"{a.pgn}\npositions used: {len(used)}  "
          f"(judge |cp| >= {a.min_cp:.0f}, SF18 @1M, white POV)\n")
    print(f"{'advantaged side':<18}{'n':>6}{'W':>6}{'D':>6}{'L':>6}{'converts':>11}"
          f"{'score':>9}")
    for k in ("archon", "sf18"):
        w, d, l = tally[k]
        n = w + d + l
        if not n:
            continue
        print(f"{k:<18}{n:>6}{w:>6}{d:>6}{l:>6}{100*w/n:>10.0f}%"
              f"{100*(w+0.5*d)/n:>8.1f}%")
    wa, da, la = tally["archon"]
    ws, ds, ls = tally["sf18"]
    na, ns = wa + da + la, ws + ds + ls
    if na and ns:
        print(f"\nconversion gap at equal nodes: "
              f"{100*ws/ns - 100*wa/na:+.0f} points "
              f"(the 10+0.1 corpus figure was +72: 12% vs 84%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
