"""Is our QSEARCH -- the value the search actually backs up -- worse than
Stockfish's?

WHY this branch is the one left.  The equal-node quality gap is ~78 Elo and it is
not depth-limited (the headroom is flat over a 16x node range while the top-1 gap
WIDENS).  So at comparable depth Stockfish's answer is simply better.  Static
eval is at parity in every phase -- but that was measured at the ROOT, and a
search does not back up root evals.  It backs up QSEARCH values at the leaves.

Two engines with identical static evals can have very different leaf values: the
qsearch resolves captures and checks, and its delta pruning, SEE pruning, check
generation and stand-pat rules all change what comes back.  If ours correlates
worse with deep truth, every backed-up value in the tree is noisier, which is
exactly "converts depth into correct moves less well" and is invisible to a
root-static comparison.

Frame-free, same as the static comparison: Spearman against the deep oracle value
(immune to any monotone rescale) plus the RMS residual after fitting one scale per
engine per bucket, so no frame conversion can be got wrong.  Mate-band truths are
excluded -- in sparse positions they dominate a linear fit and measure mate
distance rather than value quality.

Depth 1 is the probe: one ply plus qsearch, the shallowest thing both engines
will report. Depth 4 is included to show whether the gap opens or closes with a
little real search on top.

    .venv/bin/python tools/forensics/leaf_quality.py [--oracle ...] [--per-bucket 200]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-pgo"
SF = REPO + "/engines/bin/stockfish18"


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def fit(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 8:
        return None
    sxx = sum(x * x for x, _ in pairs)
    k = (sum(x * y for x, y in pairs) / sxx) if sxx else 0.0
    resid = (sum((y - k * x) ** 2 for x, y in pairs) / len(pairs)) ** 0.5

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    rho = statistics.correlation(ranks([p[0] for p in pairs]),
                                 ranks([p[1] for p in pairs]))
    return dict(n=len(pairs), k=k, resid=resid, rho=rho)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="tools/nnue/calib/oracle-phase-clip.json")
    ap.add_argument("--per-bucket", type=int, default=200)
    ap.add_argument("--depths", default="1,4")
    ap.add_argument("--max-abs-truth", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    tiers = json.load(open(args.oracle))["tiers"]
    top = tiers[max(tiers, key=lambda t: int(t))]
    by = collections.defaultdict(list)
    for fen, moves in top.items():
        if len(moves) < 2:
            continue
        truth = max(moves.values())
        if args.max_abs_truth and abs(truth) > args.max_abs_truth:
            continue
        by[bucket(npieces(fen))].append((fen, truth))
    rng = random.Random(args.seed)

    ours = Engine(OURS, net=NET, hash_mb=16, name="archon")
    ours.assert_net_live()
    sf = Engine(SF, net=None, hash_mb=16, name="sf18")

    for d in [int(x) for x in args.depths.split(",")]:
        print(f"\n=== depth {d} (leaf value: {'qsearch-dominated' if d <= 1 else 'shallow search'}) ===")
        print(f"{'pieces':<8}{'n':>5} | {'ARCHON rho':>11}{'resid':>8} | "
              f"{'SF18 rho':>9}{'resid':>8} | {'rho gap':>8}")
        for k in ("21+", "13-20", "8-12", "<=7"):
            rows = by.get(k) or []
            if len(rows) < 20:
                continue
            rows = rng.sample(rows, min(args.per_bucket, len(rows)))
            av, sv, tv = [], [], []
            for fen, truth in rows:
                ours.newgame()
                ra = ours.go(fen=fen, depth=d)
                sf.newgame()
                rs = sf.go(fen=fen, depth=d)
                if not ra["ladder"] or not rs["ladder"]:
                    continue
                av.append(ra["ladder"][max(ra["ladder"])]["score"])
                sv.append(rs["ladder"][max(rs["ladder"])]["score"])
                tv.append(truth)
            a, s = fit(av, tv), fit(sv, tv)
            if not a or not s:
                continue
            print(f"{k:<8}{a['n']:>5} | {a['rho']:>11.3f}{a['resid']:>8.0f} | "
                  f"{s['rho']:>9.3f}{s['resid']:>8.0f} | {a['rho']-s['rho']:>+8.3f}",
                  flush=True)
    ours.close()
    sf.close()
    print("\nA rho gap that is ~0 at depth 1 says our leaf values are fine and the")
    print("deficit is in what the search DOES with them.  A negative gap says our")
    print("qsearch is the defect, and every backed-up value in the tree inherits it.")


if __name__ == "__main__":
    main()
