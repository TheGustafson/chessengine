"""Build a PHASE-REPRESENTATIVE screen oracle from real games.

WHY this exists.  `tools/nnue/calib/oracle-obs.json` -- the corpus behind every
tunable screen this project has ever run -- is 9,992 positions and 100% of them
have 21+ pieces; spot-checking the FENs, they are all around move 6-8.  Against
that, the errors we actually make in lost games distribute like this, measured
on the gift corpus:

    pieces      share of errors    share of WIN PROBABILITY lost    in oracle
    <= 7            66.7%                  36.8%                      0%
    8-12            18.3%                   8.4%                      0%
    13-20           11.1%                  36.9%                      0%
    21+              4.0%                  17.9%                    100%

So the instrument samples the phase that holds ~18% of our damage and is blind
to the other 82%.  That is not merely low power -- it can be ANTI-correlated:
HINDSIGHT_EVAL_SUM=111 screened at +0.03 cp on this corpus and lost 17.4 Elo in
real games.  A screen is only as good as its population, and nobody had ever
plotted this one.

Positions come from our OWN games (all results, not just losses -- conditioning
the population on the outcome would bias every future screen toward whatever we
were doing while losing), sampled uniformly across each game so every phase is
represented in proportion to how often we play it.

Schema matches obs_to_oracle.py exactly, so `screen.py --oracle` accepts it:
    {"source": ..., "tiers": {"<nodes>": {fen: {uci: cp_from_side_to_move}}}}

    .venv/bin/python tools/forensics/build_oracle.py \\
        --pgn 'engines/runs/rematch-v1/*/games.pgn' --positions 2500 \\
        --out tools/nnue/calib/oracle-phase.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine  # noqa: E402

REPO = "/home/nick/projects/chessengine"
SF = REPO + "/engines/bin/stockfish18"
TIERS = (4000, 16000, 64000)


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def collect(patterns, per_game, seed):
    import chess
    import chess.pgn

    rng = random.Random(seed)
    seen, out = set(), []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, errors="ignore") as fh:
                while (g := chess.pgn.read_game(fh)) is not None:
                    fens, node = [], g
                    while node.variations:
                        node = node.variations[0]
                        b = node.board()
                        # a position with <2 legal moves measures nothing, and
                        # in-check positions are evasion-only: neither exercises
                        # the quiet-move pruning a tunable screen is testing
                        if b.is_game_over() or b.is_check():
                            continue
                        if len(list(b.legal_moves)) < 2:
                            continue
                        fens.append(b.fen())
                    if not fens:
                        continue
                    # uniform across the game -> phases appear in the
                    # proportion we actually play them
                    for f in rng.sample(fens, min(per_game, len(fens))):
                        if f not in seen:
                            seen.add(f)
                            out.append(f)
    return out


def worker(chunk):
    import chess

    sf = Engine(SF, net=None, hash_mb=128, name="sf18")
    res = []
    for fen in chunk:
        b = chess.Board(fen)
        moves = [m.uci() for m in b.legal_moves]
        per_tier = {}
        ok = True
        for t in TIERS:
            scores = {}
            for mv in moves:
                sf.newgame()
                r = sf.go(fen=fen, nodes=t, searchmoves=[mv])
                if not r["ladder"]:
                    ok = False
                    break
                scores[mv] = r["ladder"][max(r["ladder"])]["score"]
            if not ok:
                break
            per_tier[str(t)] = scores
        if ok:
            res.append((fen, per_tier))
    sf.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", action="append", required=True)
    ap.add_argument("--positions", type=int, default=2500)
    ap.add_argument("--per-game", type=int, default=12)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--out", default="tools/nnue/calib/oracle-phase.json")
    args = ap.parse_args()

    fens = collect(args.pgn, args.per_game, args.seed)
    random.Random(args.seed).shuffle(fens)
    fens = fens[: args.positions]
    dist = collections.Counter(bucket(npieces(f)) for f in fens)
    print(f"{len(fens)} positions sampled")
    for k in ("<=7", "8-12", "13-20", "21+"):
        print(f"  {k:<8}{dist[k]:>6}  {100*dist[k]/max(1,len(fens)):5.1f}%")
    est = sum(len(f.split()[0]) for f in fens)  # crude
    print(f"scoring every legal move at tiers {TIERS} on {args.workers} workers "
          f"(~{est} board chars, this takes a while)", flush=True)

    n = args.workers
    with mp.Pool(n) as pool:
        parts = pool.map(worker, [fens[i::n] for i in range(n)])

    tiers: dict[str, dict] = {str(t): {} for t in TIERS}
    for part in parts:
        for fen, per_tier in part:
            for t, sc in per_tier.items():
                tiers[t][fen] = sc
    json.dump({"source": "games-phase-representative",
               "pgn": args.pgn, "seed": args.seed, "tiers": tiers},
              open(args.out, "w"))
    for t in tiers:
        print(f"tier {t}: {len(tiers[t])} positions, "
              f"{sum(len(m) for m in tiers[t].values())} scored moves")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
