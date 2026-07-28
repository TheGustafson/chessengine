"""The quality headroom against Stockfish at a REALISTIC budget.

WHY this number has to be redone.  The ~78 Elo "equal-node move quality" term in
the model came from cp headroom measured at 12,500 nodes (+1.24 cp aggregate,
+2.81 in the middlegame).  But top-1 at 200k says we are at PARITY in the
middlegame and behind only at 8-12 pieces -- so the headroom at the budget we
actually play may be far smaller than 78 Elo, which would make SPEED the dominant
term rather than a co-equal one.

Method: both engines pick a move at the arm budget; the oracle's stored best value
is the truth; each engine's chosen move is priced by a `searchmoves` probe at a
budget several times the arm's.  Positions where an engine already played the
oracle's move cost it nothing and skip the probe, which is what makes this
affordable.  The probe budget is the same for both engines, so the COMPARISON is
fair even though it is shallower than the 4M truth.

Reported in win probability as well as centipawns: wp is the unit that converts
to results, and in sparse positions cp does not.

    .venv/bin/python tools/forensics/headroom200k.py [--nodes 200000]
        [--probe 1600000] [--workers 6] [--limit 0]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-pgo"
SF = REPO + "/engines/bin/stockfish18"


def wp(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / 170.0))


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def worker(job):
    rows, nodes, probe = job
    ours = Engine(OURS, net=NET, hash_mb=16, name="archon")
    sf = Engine(SF, net=None, hash_mb=64, name="sf18")
    out = []
    for fen, best, bcp in rows:
        ours.newgame()
        a = ours.go(fen=fen, nodes=nodes)["best"]
        sf.newgame()
        s = sf.go(fen=fen, nodes=nodes)["best"]
        vals = {}
        for mv in {a, s}:
            if mv == best:
                vals[mv] = bcp
                continue
            sf.newgame()
            r = sf.go(fen=fen, nodes=probe, searchmoves=[mv])
            if not r["ladder"]:
                vals[mv] = None
            else:
                vals[mv] = r["ladder"][max(r["ladder"])]["score"]
        if vals.get(a) is None or vals.get(s) is None:
            continue
        out.append((fen, a == best, s == best,
                    min(bcp, 20000) - min(vals[a], 20000),
                    min(bcp, 20000) - min(vals[s], 20000),
                    wp(bcp) - wp(vals[a]), wp(bcp) - wp(vals[s])))
    ours.close()
    sf.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="tools/nnue/calib/deep-top1.json")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--probe", type=int, default=1_600_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    o = json.load(open(args.oracle))
    rows = [(f, v[0], v[1]) for f, v in o["best"].items()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} positions | arms @ {args.nodes} | loss probe @ {args.probe} "
          f"| truth @ {o['nodes']}", flush=True)

    n = args.workers
    with mp.Pool(n) as pool:
        parts = pool.map(worker, [(rows[i::n], args.nodes, args.probe) for i in range(n)])
    res = [r for p in parts for r in p]

    agg = collections.defaultdict(list)
    for fen, aok, sok, acp, scp, awp, swp in res:
        agg[bucket(npieces(fen))].append((aok, sok, acp, scp, awp, swp))

    print(f"\n{'pieces':<8}{'n':>5} | {'A top1':>7}{'S top1':>7} | "
          f"{'A cp':>7}{'S cp':>7}{'head':>7} | {'A wp':>7}{'S wp':>7}{'head':>8}")
    tot = []
    for k in ("21+", "13-20", "8-12", "<=7"):
        v = agg.get(k) or []
        if len(v) < 20:
            continue
        tot += v
        n_ = len(v)
        print(f"{k:<8}{n_:>5} | {100*sum(x[0] for x in v)/n_:>6.1f}%"
              f"{100*sum(x[1] for x in v)/n_:>6.1f}% | "
              f"{statistics.mean(x[2] for x in v):>7.1f}"
              f"{statistics.mean(x[3] for x in v):>7.1f}"
              f"{statistics.mean(x[2] for x in v)-statistics.mean(x[3] for x in v):>7.1f} | "
              f"{statistics.mean(x[4] for x in v):>7.3f}"
              f"{statistics.mean(x[5] for x in v):>7.3f}"
              f"{statistics.mean(x[4] for x in v)-statistics.mean(x[5] for x in v):>8.3f}",
              flush=True)
    if tot:
        n_ = len(tot)
        acp = statistics.mean(x[2] for x in tot)
        scp = statistics.mean(x[3] for x in tot)
        awp = statistics.mean(x[4] for x in tot)
        swp = statistics.mean(x[5] for x in tot)
        sd = statistics.stdev([x[2] - x[3] for x in tot]) / math.sqrt(n_)
        print(f"\nALL      {n_:>5} | {100*sum(x[0] for x in tot)/n_:>6.1f}%"
              f"{100*sum(x[1] for x in tot)/n_:>6.1f}% | {acp:>7.1f}{scp:>7.1f}"
              f"{acp-scp:>7.1f} | {awp:>7.3f}{swp:>7.3f}{awp-swp:>8.3f}")
        print(f"\nheadroom {acp-scp:+.2f} +- {sd:.2f} cp at {args.nodes} nodes")
        print(f"  for comparison, at 12,500 nodes on the openings corpus it was "
              f"+1.24 +- 0.30 cp,\n  and the ~78 Elo quality term was derived from that.")


if __name__ == "__main__":
    main()
