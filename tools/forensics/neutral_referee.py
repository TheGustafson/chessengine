"""Headroom against Stockfish, judged by a THIRD engine.

WHY this check is necessary.  Every cross-engine number tonight was refereed by
Stockfish.  Top-1 agreement against an SF oracle was shown to reward being SF
(+15.1 points of swing for SF against +5.2 for us when the oracle deepened), and
the loss metric is not immune either: the probe that prices each engine's chosen
move is an SF search, and SF@200k tends to choose moves that SF@1.6M also likes,
because their errors are correlated.  That understates SF's measured loss by an
unknown amount.

`engines/bin/reckless` is an unrelated engine (different codebase, different net,
top-2 SPCC).  Refereeing both sides with it removes the shared-error term:
whatever Reckless gets wrong, it gets wrong for both engines equally.

Both metrics become meaningful again under a neutral referee:
  top-1  agreement with Reckless's best move -- now unbiased, since neither
         engine is Reckless
  loss   truth minus the referee's value of the chosen move, in ONE frame

If the +3.79 cp headroom survives, it is real.  If it collapses, it was the
shared-error term and the model's central number goes with it.

    .venv/bin/python tools/forensics/neutral_referee.py [--nodes 200000]
        [--probe 1600000] [--limit 1200] [--workers 6]
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
REF = REPO + "/engines/bin/reckless"
CLIP = 1000


def wp(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / 170.0))


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def worker(job):
    fens, nodes, probe = job
    a_eng = Engine(OURS, net=NET, hash_mb=16, name="archon")
    s_eng = Engine(SF, net=None, hash_mb=16, name="sf18")
    r_eng = Engine(REF, net=None, hash_mb=64, name="reckless")
    out = []
    cl = lambda v: max(-CLIP, min(CLIP, v))
    for fen in fens:
        a_eng.newgame()
        ra = a_eng.go(fen=fen, nodes=nodes)
        s_eng.newgame()
        rs = s_eng.go(fen=fen, nodes=nodes)
        if not ra["ladder"] or not rs["ladder"]:
            continue
        a, s = ra["best"], rs["best"]
        r_eng.newgame()
        rr = r_eng.go(fen=fen, nodes=probe)
        if not rr["ladder"]:
            continue
        top = rr["ladder"][max(rr["ladder"])]
        best, bcp = top["move"], top["score"]
        vals = {best: bcp}
        ok = True
        for mv in {a, s}:
            if mv in vals:
                continue
            r_eng.newgame()
            p = r_eng.go(fen=fen, nodes=probe, searchmoves=[mv])
            if not p["ladder"]:
                ok = False
                break
            vals[mv] = p["ladder"][max(p["ladder"])]["score"]
        if not ok:
            continue
        out.append((fen, a == best, s == best,
                    cl(bcp) - cl(vals[a]), cl(bcp) - cl(vals[s]),
                    wp(cl(bcp)) - wp(cl(vals[a])), wp(cl(bcp)) - wp(cl(vals[s]))))
    for e in (a_eng, s_eng, r_eng):
        e.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="tools/nnue/calib/deep-top1.json")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--probe", type=int, default=1_600_000)
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    fens = list(json.load(open(args.oracle))["best"].keys())[: args.limit]
    print(f"{len(fens)} positions | arms @ {args.nodes} | "
          f"NEUTRAL referee: reckless @ {args.probe}", flush=True)

    n = args.workers
    with mp.Pool(n) as pool:
        parts = pool.map(worker, [(fens[i::n], args.nodes, args.probe) for i in range(n)])
    res = [r for p in parts for r in p]
    if not res:
        raise SystemExit("no results")

    agg = collections.defaultdict(list)
    for fen, aok, sok, acp, scp, awp, swp in res:
        agg[bucket(npieces(fen))].append((aok, sok, acp, scp, awp, swp))

    print(f"\n{'pieces':<8}{'n':>5} | {'A top1':>7}{'S top1':>7} | "
          f"{'A cp':>7}{'S cp':>7}{'head':>7} | {'A wp':>7}{'S wp':>7}{'head':>8}")
    tot = []
    for k in ("21+", "13-20", "8-12", "<=7"):
        v = agg.get(k) or []
        if len(v) < 15:
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
    n_ = len(tot)
    acp = statistics.mean(x[2] for x in tot)
    scp = statistics.mean(x[3] for x in tot)
    awp = statistics.mean(x[4] for x in tot)
    swp = statistics.mean(x[5] for x in tot)
    sd = statistics.stdev([x[2] - x[3] for x in tot]) / math.sqrt(n_)
    wsd = statistics.stdev([x[4] - x[5] for x in tot]) / math.sqrt(n_)
    at1 = 100 * sum(x[0] for x in tot) / n_
    st1 = 100 * sum(x[1] for x in tot) / n_
    print(f"\nALL      {n_:>5} | {at1:>6.1f}%{st1:>6.1f}% | {acp:>7.1f}{scp:>7.1f}"
          f"{acp-scp:>7.1f} | {awp:>7.3f}{swp:>7.3f}{awp-swp:>8.3f}")
    print(f"\nNEUTRAL headroom {acp-scp:+.2f} +- {sd:.2f} cp   |   "
          f"{awp-swp:+.4f} +- {wsd:.4f} wp")
    print(f"top-1 gap {at1-st1:+.1f} points, now unbiased (neither engine is the referee)")
    print("\nSF-refereed comparison said +3.79 +- 0.74 cp and a 10.2-point top-1 gap.")
    print("A much smaller neutral number means those carried a shared-error term.")


if __name__ == "__main__":
    main()
