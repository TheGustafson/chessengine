"""Is our static eval worse than Stockfish's, and does it get worse as the
board empties?

WHY ask.  The net-only suite reports d47v at near-parity with SF18 -- but that
suite, like every other instrument in this repo, was scored on an opening-heavy
population, and 82% of the win probability we lose to move errors is spent
below 21 pieces.  "Near parity on average" and "accurate in the phase where we
lose" are different claims and only one of them has been tested.

Frame problem and how it is dodged.  Our internal centipawn is ~0.64x
Stockfish's, so |static - truth| is not comparable between engines.  Two
scale-free statistics are reported instead:

  spearman   rank correlation between an engine's static eval and the deep
             search value.  Immune to any monotone rescaling, so no frame
             conversion is needed or possible to get wrong.
  resid      RMS error after fitting the single best linear scale per engine
             per bucket, expressed in SF centipawns.  This ASKS the data for
             the frame instead of assuming 0.64, which also means the fitted
             slope is a free readout of the frame in each phase.

Ground truth is the deep search value from the tiered oracle (its top tier),
not the other engine's opinion -- comparing two evals to each other cannot say
which is wrong.

    .venv/bin/python tools/forensics/eval_by_phase.py [--oracle ORACLE.json]
        [--per-bucket 250]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "/home/nick/projects/chessengine"
NET = REPO + "/engines/canonical/net.cenn"
OURS = REPO + "/engines/bin/archon-d47v-pgo"
SF = REPO + "/engines/bin/stockfish18"


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def our_statics(fens):
    out = subprocess.run([OURS, "nnue-eval", NET], input="\n".join(fens) + "\n",
                         capture_output=True, text=True, cwd=REPO).stdout.split()
    return [int(x) for x in out]


def sf_statics(fens):
    """One process, `eval` per position -- stm-relative like ours."""
    import chess

    inp = "".join(f"position fen {f}\neval\n" for f in fens) + "quit\n"
    out = subprocess.run([SF], input=inp, capture_output=True, text=True,
                         cwd=REPO).stdout
    vals, i = [], 0
    for line in out.splitlines():
        if "Final evaluation" in line:
            tok = line.split()[2]
            try:
                v = int(round(float(tok) * 100))
            except ValueError:
                v = None
            # SF prints white-relative; convert to side-to-move
            if v is not None and not chess.Board(fens[i]).turn:
                v = -v
            vals.append(v)
            i += 1
    return vals


def fit_stats(xs, ys):
    """Best single scale k for y ~ k*x, its RMS residual, and Spearman rho."""
    import statistics

    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 8:
        return None
    sx = sum(x * y for x, y in pairs)
    sxx = sum(x * x for x, _ in pairs)
    k = sx / sxx if sxx else 0.0
    resid = (sum((y - k * x) ** 2 for x, y in pairs) / len(pairs)) ** 0.5

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    rho = statistics.correlation(rx, ry) if len(pairs) > 2 else 0.0
    return dict(n=len(pairs), k=k, resid=resid, rho=rho)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="tools/nnue/calib/oracle-phase.json")
    ap.add_argument("--per-bucket", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--max-abs-truth", type=int, default=1000,
                    help="drop positions whose oracle value is in the mate band. "
                         "WHY this is not optional: in sparse positions the deep "
                         "oracle routinely returns +-20000, which dominates any "
                         "linear fit -- the first run of this tool reported RMS "
                         "residuals of 1855 and 4232 cp and fitted slopes of 2.18 "
                         "and 4.33, i.e. it was measuring mate distances, not eval "
                         "quality. 0 disables the filter.")
    args = ap.parse_args()

    d = json.load(open(args.oracle))
    tiers = d["tiers"]
    top = tiers[max(tiers, key=lambda t: int(t))]

    by = collections.defaultdict(list)
    for fen, moves in top.items():
        if len(moves) < 2:
            continue
        truth = max(moves.values())
        if args.max_abs_truth and abs(truth) > args.max_abs_truth:
            continue                      # mate band: measures distance, not eval
        by[bucket(npieces(fen))].append((fen, truth))
    rng = random.Random(args.seed)

    print(f"oracle {args.oracle}, truth = deepest tier best-move value")
    print(f"{'pieces':<8}{'n':>5} | {'ARCHON rho':>11}{'resid':>8}{'slope':>7}"
          f" | {'SF18 rho':>9}{'resid':>8}{'slope':>7}")
    for k in ("21+", "13-20", "8-12", "<=7"):
        rows = by.get(k) or []
        if len(rows) < 10:
            continue
        rows = rng.sample(rows, min(args.per_bucket, len(rows)))
        fens = [f for f, _ in rows]
        truth = [t for _, t in rows]
        a = fit_stats(our_statics(fens), truth)
        s = fit_stats(sf_statics(fens), truth)
        if not a or not s:
            continue
        print(f"{k:<8}{a['n']:>5} | {a['rho']:>11.3f}{a['resid']:>8.0f}{a['k']:>7.2f}"
              f" | {s['rho']:>9.3f}{s['resid']:>8.0f}{s['k']:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
