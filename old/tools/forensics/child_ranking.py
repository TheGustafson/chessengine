"""Can our eval rank the MOVES of a position?  (Not: can it rank positions.)

WHY this is the measurement that was missing.  Every eval comparison tonight --
and the net-only suite before it -- correlates an engine's eval with deep truth
ACROSS a set of positions.  That is what a training loss measures, and by it we
are at parity with Stockfish in every phase.

But a search never compares two unrelated positions.  It compares the CHILDREN of
one position, and what it needs is for the eval to order those children the way a
deep search would.  An eval can be excellent across positions -- capturing
material, king safety, structure -- while being nearly useless WITHIN a position,
if the differences between sibling moves are small relative to its noise.  Those
are different properties and only the second one drives move choice.

Method: for each position the tiered oracle already scores EVERY legal move, so
the per-move truth is free.  Evaluate every child statically with each engine and
take the Spearman against that truth, per position, then report the distribution.
Also the top-1 rate of the raw static eval -- how often the statically best child
IS the oracle's best move -- which is the quantity a 1-ply search would get right.

Frame-free by construction: a rank correlation within a position cannot be moved
by any monotone rescaling, so the 0.46x frame difference cannot touch it.

    .venv/bin/python tools/forensics/child_ranking.py [--oracle ...] [--per-bucket 150]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import subprocess
import sys

REPO = "/home/nick/projects/chessengine"
NET = REPO + "/engines/canonical/net.cenn"
OURS = REPO + "/engines/bin/archon-d47v-pgo"
SF = REPO + "/engines/bin/stockfish18"


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    if len(a) < 3:
        return None
    try:
        return statistics.correlation(ranks(a), ranks(b))
    except statistics.StatisticsError:
        return None


def our_statics(fens):
    out = subprocess.run([OURS, "nnue-eval", NET], input="\n".join(fens) + "\n",
                         capture_output=True, text=True, cwd=REPO).stdout.split()
    return [int(x) for x in out]


def sf_statics(fens):
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
            if v is not None and not chess.Board(fens[i]).turn:
                v = -v
            vals.append(v)
            i += 1
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="tools/nnue/calib/oracle-phase-clip.json")
    ap.add_argument("--per-bucket", type=int, default=150)
    ap.add_argument("--min-moves", type=int, default=6)
    ap.add_argument("--max-abs-truth", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    import chess

    tiers = json.load(open(args.oracle))["tiers"]
    top = tiers[max(tiers, key=lambda t: int(t))]
    by = collections.defaultdict(list)
    for fen, moves in top.items():
        if len(moves) < args.min_moves:
            continue
        if args.max_abs_truth and max(abs(v) for v in moves.values()) > args.max_abs_truth:
            continue
        by[bucket(npieces(fen))].append((fen, moves))
    rng = random.Random(args.seed)

    print(f"{'pieces':<8}{'n':>5}{'mv/pos':>8} | {'ARCHON rho':>11}{'SF18 rho':>10}"
          f"{'gap':>8} | {'A top1':>8}{'S top1':>8}{'gap':>7}")
    tot = collections.defaultdict(list)
    for k in ("21+", "13-20", "8-12", "<=7"):
        rows = by.get(k) or []
        if len(rows) < 20:
            continue
        rows = rng.sample(rows, min(args.per_bucket, len(rows)))
        arho, srho, ahit, shit, n, nm = [], [], 0, 0, 0, []
        for fen, moves in rows:
            b = chess.Board(fen)
            ucis, kids = [], []
            for u in moves:
                try:
                    m = chess.Move.from_uci(u)
                    if m not in b.legal_moves:
                        continue
                except Exception:  # noqa: BLE001
                    continue
                b.push(m)
                kids.append(b.fen())
                b.pop()
                ucis.append(u)
            if len(ucis) < args.min_moves:
                continue
            truth = [moves[u] for u in ucis]
            # children are the OPPONENT to move: negate to get our point of view
            av = [-v for v in our_statics(kids)]
            sv = [-v if v is not None else None for v in sf_statics(kids)]
            if len(av) != len(ucis) or len(sv) != len(ucis) or any(v is None for v in sv):
                continue
            ra, rs = spearman(av, truth), spearman(sv, truth)
            if ra is None or rs is None:
                continue
            arho.append(ra)
            srho.append(rs)
            best = max(range(len(truth)), key=lambda i: truth[i])
            ahit += (max(range(len(av)), key=lambda i: av[i]) == best)
            shit += (max(range(len(sv)), key=lambda i: sv[i]) == best)
            n += 1
            nm.append(len(ucis))
        if not n:
            continue
        tot["a"] += arho
        tot["s"] += srho
        tot["ah"].append(ahit)
        tot["sh"].append(shit)
        tot["n"].append(n)
        print(f"{k:<8}{n:>5}{statistics.mean(nm):>8.1f} | "
              f"{statistics.mean(arho):>11.3f}{statistics.mean(srho):>10.3f}"
              f"{statistics.mean(arho)-statistics.mean(srho):>+8.3f} | "
              f"{100*ahit/n:>7.1f}%{100*shit/n:>7.1f}%"
              f"{100*(ahit-shit)/n:>+7.1f}", flush=True)
    if tot["a"]:
        N = sum(tot["n"])
        print(f"\nALL      {N:>5}          | "
              f"{statistics.mean(tot['a']):>11.3f}{statistics.mean(tot['s']):>10.3f}"
              f"{statistics.mean(tot['a'])-statistics.mean(tot['s']):>+8.3f} | "
              f"{100*sum(tot['ah'])/N:>7.1f}%{100*sum(tot['sh'])/N:>7.1f}%"
              f"{100*(sum(tot['ah'])-sum(tot['sh']))/N:>+7.1f}")
    print("\nAcross-position parity plus WITHIN-position deficit would mean the net is")
    print("trained for the wrong objective: good at valuing positions, weak at the")
    print("move-ordering signal a search actually consumes.")


if __name__ == "__main__":
    main()
