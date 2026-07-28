"""How many NODES does each engine need to find the right move?

WHY nodes and not plies.  A pruning gate exists to trade accuracy-per-depth for
depth-per-second, so "turning gate X off finds the move at a shallower depth"
is very nearly a tautology -- it is true of almost every gate on almost every
position, and it cannot tell you whether the gate is a defect or is doing its
job.  Nodes-to-find is the axis that converts to time, and therefore the only
one that can say a gate is COSTING us anything.

It is also the direct measurement of "our search shape is wrong": run the same
position through both engines and compare the cost of reaching the same
conclusion.  On game 18 that is 618 nodes for Stockfish against 410,613 for us
-- a 664x deficit that no amount of eval work would close, and which the ply
axis reports as a mere "16 plies".

`d_find`/`n_find` require the engine to CHOOSE the move and never abandon it:
a move touched at depth 9 and dropped at 10 was never found, and the in-game
search would not have played it.

    .venv/bin/python tools/forensics/find_cost.py CORPUS.json [--limit N]
        [--max-depth 24] [--gates pfut,lmr]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_depth import GATES, engine_defaults  # noqa: E402
from uciprobe import Engine  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-tune-pgo"
SF = REPO + "/engines/bin/stockfish18"


def find(eng: Engine, fen: str, target: str, max_depth: int):
    """(depth, nodes) at which `target` is first chosen and never abandoned."""
    eng.newgame()
    lad = eng.go(fen=fen, depth=max_depth)["ladder"]
    first = None
    for d in sorted(lad):
        if lad[d]["move"] == target:
            if first is None:
                first = d
        else:
            first = None
    if first is None:
        return None, None
    return first, lad[first]["nodes"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--min-cp", type=int, default=100)
    ap.add_argument("--max-cp", type=int, default=2000)
    ap.add_argument("--verify-nodes", type=int, default=1_000_000)
    ap.add_argument("--gates", default="pfut,lmr,rfp,razor",
                    help="which gates to also measure with, off")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.corpus))
            if args.min_cp <= r["loss"] <= args.max_cp][: args.limit]
    gates = {k: GATES[k] for k in args.gates.split(",") if k}
    dflts = engine_defaults(OURS)

    ours = Engine(OURS, name="archon", hash_mb=64)
    ours.assert_net_live()
    sf = Engine(SF, net=None, hash_mb=64, name="sf18")

    print(f"{'game':<6}{'played':>7}{'best':>7}{'SFd':>5}{'SFn':>10}"
          f"{'oursd':>6}{'oursn':>10}{'ratio':>8}" +
          "".join(f"{g:>10}" for g in gates))
    out, ratios = [], []
    for r in rows:
        sf.newgame()
        b = sf.go(fen=r["fen"], nodes=args.verify_nodes)
        if not b["ladder"]:
            continue
        target = b["ladder"][max(b["ladder"])]["move"]
        if target == r["played"]:
            continue

        sd, sn = find(sf, r["fen"], target, args.max_depth)
        for _n, (k, _off) in gates.items():
            ours.setopt(k, dflts[k])
        od, on = find(ours, r["fen"], target, args.max_depth)
        rec = dict(game=r["game"], fen=r["fen"], played=r["played"], target=target,
                   sf_d=sd, sf_n=sn, our_d=od, our_n=on)
        ratio = (on / sn) if (on and sn) else None
        rec["ratio"] = ratio
        if ratio:
            ratios.append(ratio)

        cells = []
        for name, (k, off) in gates.items():
            for _n2, (k2, _o2) in gates.items():
                ours.setopt(k2, dflts[k2])
            ours.setopt(k, off)
            gd, gn = find(ours, r["fen"], target, args.max_depth)
            rec[name + "_n"] = gn
            rec[name + "_d"] = gd
            cells.append((f"{gn/on:.2f}x" if (gn and on) else ("x" if gn is None else "-")).rjust(10))
        for _n, (k, _off) in gates.items():
            ours.setopt(k, dflts[k])

        out.append(rec)
        print(f"g{str(r['game']):<5}{r['played']:>7}{target:>7}"
              f"{str(sd or '-'):>5}{str(sn or '-'):>10}"
              f"{str(od or '-'):>6}{str(on or '-'):>10}"
              f"{(f'{ratio:.0f}x' if ratio else '-'):>8}" + "".join(cells), flush=True)

    ours.close()
    sf.close()

    if ratios:
        ratios.sort()
        print(f"\nnodes-to-find ratio (ours / SF18), n={len(ratios)}")
        print(f"  median {statistics.median(ratios):.1f}x   "
              f"geomean {statistics.geometric_mean(ratios):.1f}x   "
              f"p90 {ratios[int(0.9*(len(ratios)-1))]:.1f}x   max {ratios[-1]:.1f}x")
    dst = args.out or (os.path.splitext(args.corpus)[0] + "_findcost.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
