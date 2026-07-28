#!/usr/bin/env python3
"""Read a match. Refuses to let you over-read a partial one.

    python3 searchopt/tools/match.py engines/runs/corpus-d48x
    python3 searchopt/tools/match.py engines/runs/*/          # all of them

WHY the nagging. On 2026-07-28 `rfp-threat0` read **-15.3 Elo at n=600** and
finished at **-3.5 +- 14.1 at n=1185**. I quoted the -15 mid-run more than once.
A partial match is not a small version of the final result; it is a different
number, and the CI is the only thing that says so.

So this always prints the interval, always prints how many games are still
needed for a given resolution, and labels anything whose CI spans zero as
INCONCLUSIVE regardless of how tempting the point estimate looks.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import statistics
import sys

GAME = re.compile(r'\[White "([^"]+)"\][\s\S]*?\[Black "([^"]+)"\][\s\S]*?\[Result "([^"]+)"\]')


def elo(p):
    return -400 * math.log10(1 / p - 1) if 0 < p < 1 else float("nan")


def read(path):
    try:
        games = GAME.findall(open(path).read())
    except FileNotFoundError:
        return None
    if not games:
        return None
    names = sorted({w for w, _, _ in games} | {b for _, b, _ in games})
    if len(names) != 2:
        return None
    a, b = names
    res = []
    for w, _, r in games:
        v = 1.0 if r == "1-0" else (0.0 if r == "0-1" else 0.5)
        res.append(v if w == a else 1 - v)
    n = len(res)
    s = sum(res) / n
    sd = statistics.pstdev(res) / math.sqrt(n) if n > 1 else 0.5
    return {"a": a, "b": b, "n": n, "score": s, "elo": elo(s),
            "lo": elo(max(1e-9, s - 2 * sd)), "hi": elo(min(1 - 1e-9, s + 2 * sd)),
            "w": sum(1 for x in res if x == 1), "d": sum(1 for x in res if x == 0.5),
            "l": sum(1 for x in res if x == 0), "sd": sd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--target-ci", type=float, default=10.0,
                    help="Elo half-width you want; prints the games still needed")
    a = ap.parse_args()
    rows = []
    for r in a.runs:
        for p in ([r] if os.path.isfile(r) else
                  glob.glob(os.path.join(r, "**", "games.pgn"), recursive=True)):
            d = read(p)
            if d:
                d["path"] = p
                rows.append(d)
    if not rows:
        print("no readable match found")
        return 1
    rows.sort(key=lambda d: -d["n"])
    print(f"{'run':<28} {'n':>6} {'W-D-L':>14} {'score':>7} {'Elo':>8} "
          f"{'95% CI':>18}  verdict")
    for d in rows:
        run = os.path.basename(os.path.dirname(d["path"]))[:27]
        half = (d["hi"] - d["lo"]) / 2
        span0 = d["lo"] <= 0 <= d["hi"]
        need = ""
        if half > a.target_ci and d["n"] > 1:
            n_need = int(d["n"] * (half / a.target_ci) ** 2)
            need = f" (+{n_need - d['n']:,} games for +-{a.target_ci:.0f})"
        verdict = "INCONCLUSIVE" + need if span0 else ("GAIN" if d["elo"] > 0 else "LOSS")
        print(f"{run:<28} {d['n']:>6} {d['w']:>4}-{d['d']}-{d['l']:<4} "
              f"{d['score']*100:6.2f}% {d['elo']:+8.1f} "
              f"[{d['lo']:+7.1f},{d['hi']:+7.1f}]  {verdict}")
        print(f"{'':<28} {d['a']} vs {d['b']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
