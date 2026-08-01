#!/usr/bin/env python3
"""tools/perf70/pgn_ptnml.py — recover `Ptnml(0-2)` from a fastchess PGN.

fastchess APPENDS to its `-pgnout` file but its stdout summary goes through
`tee`, which TRUNCATES. So a re-run with the same output directory destroys the
summary while leaving every game intact — which is how study 70 lost a
completed 250-game segment's numbers but not its games.

Pairs games by `[Round]` (the `-repeat` opening pair), scores from the
CANDIDATE's point of view, and prints a `Ptnml(0-2):` line that
`pool_sprt.py` reads verbatim.

  pgn_ptnml.py games.pgn --candidate candidate [--limit N]

`--limit N` takes only the first N games, which is how you separate an
original run from a re-run that appended to the same file.
"""

import argparse
import collections
import re
import sys

PT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": None}


def games(path, limit=None):
    """(round, white, black, result) per game, in file order."""
    cur, out = {}, []
    for line in open(path, errors="replace"):
        m = re.match(r'\[(\w+)\s+"(.*)"\]', line.strip())
        if m:
            cur[m.group(1)] = m.group(2)
            if m.group(1) == "Result":
                out.append((cur.get("Round", ""), cur.get("White", ""), cur.get("Black", ""),
                            m.group(2)))
                cur = {}
                if limit and len(out) >= limit:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--candidate", default="candidate")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    rounds = collections.defaultdict(list)
    for rnd, white, black, res in games(a.pgn, a.limit):
        pts = PT.get(res)
        if pts is None:
            continue
        # Score from the candidate's side.
        if white == a.candidate:
            rounds[rnd].append(pts)
        elif black == a.candidate:
            rounds[rnd].append(1.0 - pts)
    counts = [0] * 5
    partial = 0
    for pair in rounds.values():
        if len(pair) != 2:
            partial += 1
            continue
        counts[int(round(sum(pair) * 2))] += 1
    n = sum(counts)
    if n == 0:
        raise SystemExit("pgn_ptnml: no complete pairs (is --candidate right?)")
    score = sum(c * p for c, p in zip(counts, (0, 0.25, 0.5, 0.75, 1.0))) / n
    print(f"Games: {2 * n}, Points: ({100 * score:.2f} %)")
    print(f"Ptnml(0-2): [{', '.join(str(c) for c in counts)}]")
    if partial:
        print(f"# {partial} incomplete pair(s) dropped", file=sys.stderr)


if __name__ == "__main__":
    main()
