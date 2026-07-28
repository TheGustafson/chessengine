"""A DEEP top-1 oracle: the best move only, at a budget that outranks the arms.

WHY this exists.  Every screen in this repo scores arms at 12,500 nodes against
an oracle whose deepest tier is 64k -- and 12,500 nodes reaches depth ~9.9 where
our real games at 10+0.1 reach 17-20.  Measured on the same corpus, the verdict
FLIPS between those regimes: at 12.5k we lose top-1 in every phase, at 200k we
match or beat Stockfish in the middlegame and lose only at 8-12 pieces.  So the
screen's budget describes an engine we do not play.

Screening at a realistic 200k is impossible against a 64k oracle -- the arm would
be three times deeper than its judge.  A full every-legal-move oracle at 1M+ is
unaffordable (30 moves x 12,000 positions).  But TOP-1 AGREEMENT needs only the
best move, which is ONE deep search per position, and top-1 is the robust
statistic anyway: it is a rate, so a single mate score cannot move it the way it
moves a cp mean.

Output: {"nodes": N, "best": {fen: [uci, cp]}}

    .venv/bin/python tools/forensics/deep_top1.py --from-oracle ORACLE.json
        [--nodes 4000000] [--positions 3000] [--workers 6]
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine  # noqa: E402

REPO = "/home/nick/projects/chessengine"
SF = REPO + "/engines/bin/stockfish18"
CLIP = 1000


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def worker(job):
    fens, nodes, engine = job
    sf = Engine(engine, net=None, hash_mb=64, name="ref")
    out = {}
    for fen in fens:
        sf.newgame()
        r = sf.go(fen=fen, nodes=nodes)
        if not r["ladder"]:
            continue
        top = r["ladder"][max(r["ladder"])]
        # Clip into the decisive band.  A phase-representative corpus contains
        # endgames, where the oracle routinely returns +-20000, and an unclipped
        # store made the downstream cp headroom read -11.76 +- 35.37 -- the mean
        # was measuring mate DISTANCE.  The best MOVE is unaffected either way;
        # only the value is clipped.  (Same fix the phase corpus needed; it was
        # not carried over here, which is exactly how a known trap recurs.)
        out[fen] = [top["move"], max(-CLIP, min(CLIP, top["score"]))]
    sf.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-oracle", required=True,
                    help="reuse an existing corpus's FEN list")
    ap.add_argument("--nodes", type=int, default=4_000_000)
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--engine", default=SF,
                    help="referee engine. Use engines/bin/reckless for a NEUTRAL\n"
                         "oracle -- an SF oracle rewards arms that resemble SF,\n"
                         "which does not cancel for arms that change calibration.")
    ap.add_argument("--out", default="tools/nnue/calib/deep-top1.json")
    args = ap.parse_args()

    tiers = json.load(open(args.from_oracle))["tiers"]
    fens = list(tiers[max(tiers, key=lambda t: int(t))].keys())
    random.Random(args.seed).shuffle(fens)
    fens = fens[: args.positions]
    dist = collections.Counter(bucket(npieces(f)) for f in fens)
    print(f"{len(fens)} positions @ {args.nodes} nodes on {args.workers} workers")
    for k in ("<=7", "8-12", "13-20", "21+"):
        print(f"  {k:<8}{dist[k]:>6}  {100*dist[k]/max(1,len(fens)):5.1f}%")

    n = args.workers
    with mp.Pool(n) as pool:
        parts = pool.map(worker, [(fens[i::n], args.nodes, args.engine) for i in range(n)])
    best = {}
    for p in parts:
        best.update(p)
    json.dump({"nodes": args.nodes, "source": args.from_oracle,
               "engine": os.path.basename(args.engine), "best": best},
              open(args.out, "w"))
    print(f"{len(best)} positions -> {args.out}")


if __name__ == "__main__":
    main()
