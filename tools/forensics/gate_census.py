"""Per-gate cost in PLIES, across a whole corpus of losing moves.

WHY: one position blaming one gate is a coincidence generator.  Game 18 was
blamed on RFP for a week on the strength of a three-depth probe with a
hand-typed control; with a control read out of the binary and a full ladder it
turns out to be parent futility, and RFP-off makes that position WORSE.  The
only defence against that is to measure the same thing on many positions and
look at the distribution.

For each row: confirm the right move with a deep cold oracle, then measure the
depth at which OUR search first commits to it under each gate configuration.
`delta` is negative when disabling the gate finds the move EARLIER, i.e. the
gate was hiding it.

Output is a per-gate histogram: how often each gate hides a refutation, and by
how many plies when it does.  A gate that hides refutations on one position and
helps on nine is not a bug -- it is a gate doing its job.

    .venv/bin/python tools/forensics/gate_census.py CORPUS.json
        [--limit N] [--min-cp 100] [--max-cp 2000] [--max-depth 22]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_depth import GATES, engine_defaults  # noqa: E402
from uciprobe import Engine  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-tune-pgo"
SF = REPO + "/engines/bin/stockfish18"


def d_find(eng: Engine, fen: str, target: str, max_depth: int):
    eng.newgame()
    lad = eng.go(fen=fen, depth=max_depth)["ladder"]
    first = None
    for d in sorted(lad):
        if lad[d]["move"] == target:
            if first is None:
                first = d
        else:
            first = None
    return first, (lad[first]["nodes"] if first else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--min-cp", type=int, default=100)
    ap.add_argument("--max-cp", type=int, default=2000,
                    help="drop mate-score blowouts: already lost, nothing to fix")
    ap.add_argument("--max-depth", type=int, default=22)
    ap.add_argument("--verify-nodes", type=int, default=1_000_000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.corpus))
            if args.min_cp <= r["loss"] <= args.max_cp]
    rows = rows[: args.limit]
    print(f"{len(rows)} rows in [{args.min_cp},{args.max_cp}] cp", flush=True)

    dflts = engine_defaults(OURS)
    ours = Engine(OURS, name="archon", hash_mb=64)
    ours.assert_net_live()
    sf = Engine(SF, net=None, hash_mb=256, name="sf18")

    arms = [("control", {})] + [(n, {k: off}) for n, (k, off) in GATES.items()]
    results, per_gate = [], {n: [] for n, _ in GATES.items()}

    print(f"{'game':<6}{'played':>7}{'best':>7}{'ctl':>5}" +
          "".join(f"{n:>7}" for n, _ in GATES.items()))
    for r in rows:
        sf.newgame()
        b = sf.go(fen=r["fen"], nodes=args.verify_nodes)
        if not b["ladder"]:
            continue
        deep = b["ladder"][max(b["ladder"])]
        target = deep["move"]
        if target == r["played"]:
            print(f"g{str(r['game']):<5}{r['played']:>7}{target:>7}   -- deep oracle agrees with us, skip")
            continue

        row = dict(game=r["game"], fen=r["fen"], played=r["played"],
                   target=target, screen_cp=r["loss"])
        base = None
        cells = []
        for name, opts in arms:
            for _n, (k, _off) in GATES.items():
                ours.setopt(k, dflts[k])
            for k, v in opts.items():
                ours.setopt(k, v)
            d, nd = d_find(ours, r["fen"], target, args.max_depth)
            if name == "control":
                base = d
                row["control_d"] = d
                row["control_nodes"] = nd
                cells.append(f"{d if d else 'x':>5}")
            else:
                row[name] = d
                delta = None if (d is None or base is None) else d - base
                row[name + "_delta"] = delta
                if delta is not None:
                    per_gate[name].append(delta)
                cells.append(f"{('%+d' % delta) if delta is not None else ('x' if d is None else 'X')}"
                             .rjust(7))
        results.append(row)
        print(f"g{str(r['game']):<5}{r['played']:>7}{target:>7}" + "".join(cells), flush=True)

    ours.close()
    sf.close()

    print("\n=== PER-GATE COST (negative = disabling finds it EARLIER) ===")
    print(f"{'gate':<8}{'n':>4}{'mean':>8}{'median':>8}{'hides':>7}{'helps':>7}  worst")
    for name in per_gate:
        v = sorted(per_gate[name])
        if not v:
            continue
        mean = sum(v) / len(v)
        med = v[len(v) // 2]
        hides = sum(1 for x in v if x < 0)
        helps = sum(1 for x in v if x > 0)
        print(f"{name:<8}{len(v):>4}{mean:>8.1f}{med:>8}{hides:>7}{helps:>7}  {v[0]}")

    out = args.out or (os.path.splitext(args.corpus)[0] + "_gates.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
