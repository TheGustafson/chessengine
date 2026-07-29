#!/usr/bin/env python3
"""Per-engine NPS from a fastchess PGN written with `-pgnout nps=true nodes=true`.

Why this exists (2026-07-29). "How fast are we vs Stockfish" kept being answered
with single-position `go movetime` probes, which measure one position's node mix
on an idle box. A real match measures the thing that actually matters: sustained
throughput over a realistic position distribution, with both engines contending
for the same machine.

**The aggregate is Sum(nodes)/Sum(time), NOT the mean of per-move nps.** A mean
over moves weights a 0.05 s book-ish reply the same as a 3 s critical think, and
those have very different nps (short searches are dominated by the shallow, cheap
plies and read HIGH). The ratio you want is total work over total time. Median is
reported alongside because it says something different -- the typical move --
and when the two disagree the distribution is telling you something.

Usage:
    python3 tools/nps_from_pgn.py engines/runs/nps-ratio/games.pgn
    python3 tools/nps_from_pgn.py games.pgn --min-time 0.2   # heavier filter
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict

# {-2.07/19 0.225s, n=179470, nps=797644}
COMMENT = re.compile(r"\{([^}]*)\}")
FIELDS = re.compile(r"(?P<time>[0-9.]+)s,\s*n=(?P<nodes>\d+),\s*nps=(?P<nps>\d+)")
HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]')


def games(path):
    """Yield (white, black, [comment_text, ...]) per game, in move order."""
    white = black = None
    body: list[str] = []
    in_body = False
    for line in open(path, errors="ignore"):
        m = HEADER.match(line)
        if m:
            if in_body and white:
                yield white, black, COMMENT.findall(" ".join(body))
                body, in_body = [], False
            k, v = m.group(1), m.group(2)
            if k == "White":
                white = v
            elif k == "Black":
                black = v
        elif line.strip():
            in_body = True
            body.append(line.strip())
    if in_body and white:
        yield white, black, COMMENT.findall(" ".join(body))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--min-time", type=float, default=0.1,
                    help="drop moves searched for less than this (s). Very short "
                         "searches read HIGH and are mostly book/forced replies.")
    a = ap.parse_args()

    nodes = defaultdict(float)
    secs = defaultdict(float)
    per_move = defaultdict(list)
    dropped = defaultdict(int)
    ngames = 0

    for white, black, comments in games(a.pgn):
        ngames += 1
        # Move comments alternate W,B,W,B... from the start of the movetext.
        for i, c in enumerate(comments):
            f = FIELDS.search(c)
            if not f:
                continue
            eng = white if i % 2 == 0 else black
            t = float(f.group("time"))
            if t < a.min_time:
                dropped[eng] += 1
                continue
            nodes[eng] += int(f.group("nodes"))
            secs[eng] += t
            per_move[eng].append(int(f.group("nps")))

    if not nodes:
        print("no nps data — was the PGN written with `-pgnout nps=true`?", file=sys.stderr)
        return 1

    print(f"{ngames} games, moves with time >= {a.min_time}s\n")
    print(f"{'engine':<16} {'aggregate Mnps':>15} {'median Mnps':>12} "
          f"{'moves':>7} {'dropped':>8} {'Gnodes':>8}")
    agg = {}
    for eng in sorted(nodes, key=lambda e: -nodes[e] / max(secs[e], 1e-9)):
        a_nps = nodes[eng] / secs[eng]
        agg[eng] = a_nps
        print(f"{eng:<16} {a_nps/1e6:>15.3f} "
              f"{statistics.median(per_move[eng])/1e6:>12.3f} "
              f"{len(per_move[eng]):>7} {dropped[eng]:>8} {nodes[eng]/1e9:>8.2f}")

    names = list(agg)
    if len(names) > 1:
        print("\npairwise aggregate ratios:")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                print(f"  {x} / {y} = {agg[x]/agg[y]:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
