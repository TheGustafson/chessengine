#!/usr/bin/env python3
"""Greedy search-shape optimisation, one failing position at a time.

METHOD (user, 2026-07-28).  The tracer is the fast iteration oracle: run
parameter ablations against ONE loss, keep what improves its search shape, then
carry the winning set to a second loss and re-check BOTH side by side, then a
third.  Sub-problems bottom-up into a global search-shape fix.  Games arbitrate
only at the end — they are far too slow to iterate against.

WHY not the usual argmax statistic: on this corpus an inert +-1 flips 25% of
positions, so "does it play the right move" is mostly noise at n=1.  The shape
metrics below are depth-resolved and monotone-ish, which makes them readable on
a single position:

  collapse_d  the first depth at which the PLAYED (losing) move's forced score
              goes bad and STAYS bad.  Lower is better; SF's value is the
              target.  This is the quantity that decides the blunder — the
              engine plays the losing move exactly while its score is still ok.
  find_d      the first depth at which the free search's bestmove is the oracle
              move, and stays.  Lower is better.
  nodes       cost at the reference depth, so a "fix" that only buys depth by
              spending nodes is visible rather than hidden.

ALWAYS run the null arm.  An inert +-1 that moves collapse_d as much as a real
arm means this position cannot resolve that knob — go to the next position.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET, REPO, to_sf_frame  # noqa: E402

BAD = -150      # "gone bad" threshold, SF printed frame, mover's POV
STAY = -100     # and must stay at least this bad at every deeper ply


def collapse_depth(lad):
    ds = sorted(lad)
    for d in ds:
        if lad[d] <= BAD and all(lad[x] <= STAY for x in ds if x >= d):
            return d
    return None


def find_depth(best_by_depth, oracle):
    ds = sorted(best_by_depth)
    for d in ds:
        if all(best_by_depth[x] == oracle for x in ds if x >= d):
            return d
    return None


def measure(eng, tag, fen, played, oracle, depth):
    eng.newgame()
    g = eng.go(fen=fen, depth=depth, searchmoves=[played])
    lad = {d: to_sf_frame(g["ladder"][d]["score"], tag) for d in g["ladder"]}
    eng.newgame()
    f = eng.go(fen=fen, depth=depth)
    best = {d: f["ladder"][d]["move"] for d in f["ladder"]}
    nodes = f["ladder"][max(f["ladder"])]["nodes"] if f["ladder"] else 0
    return {
        "collapse_d": collapse_depth(lad),
        "find_d": find_depth(best, oracle),
        "final": lad[max(lad)] if lad else None,
        "best": f["best"],
        "nodes": nodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen")
    ap.add_argument("--played")
    ap.add_argument("--oracle")
    ap.add_argument("--positions",
                    help="JSON list of {fen,played,best} — score every arm across ALL "
                         "of them and aggregate. REQUIRED for any verdict: a single "
                         "position cannot resolve a knob. Measured 2026-07-28, "
                         "RED_STATSCORE_NUM swept 445..0 on one position was flat "
                         "everywhere except the exact endpoint 0, which fired at "
                         "~500 cp while three inert nulls spanned 24 cp — a chaotic "
                         "hit, not a knee. Aggregate over positions or do not "
                         "conclude.")
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--engine", default="engines/bin/archon-trace")
    ap.add_argument("--net", default=NET)
    ap.add_argument("--arms", required=True,
                    help='JSON: {"name": {"PARAM": value, ...}, ...}')
    ap.add_argument("--sf", action="store_true", help="also measure Stockfish as the target")
    a = ap.parse_args()
    os.chdir(REPO)
    arms = json.loads(a.arms)

    if a.sf:
        s = Engine("engines/bin/stockfish18", net=None, hash_mb=64, name="sf")
        r = measure(s, "sf", a.fen, a.played, a.oracle, a.depth)
        s.close()
        print(f"{'TARGET SF18':<22} collapse_d {str(r['collapse_d']):>4}  "
              f"find_d {str(r['find_d']):>4}  final {r['final']:+7.0f}  best {r['best']}")

    e = Engine(a.engine, net=a.net, hash_mb=64, name="arm")
    e.assert_net_live()

    # Restore EVERY parameter any arm touches to its engine-reported default
    # before applying the next arm.  Without this, arms silently accumulate:
    # measured 2026-07-28, an arm inherited RED_BASE=1026 from the preceding
    # null and flipped collapse_d from 16 to None -- i.e. the "finding" was an
    # uncontrolled +-1 on an unrelated constant.  Defaults come from the engine's
    # own `uci` output, not from a hand-copied table that can drift.
    defaults = {}
    for line in e.uci_lines if hasattr(e, "uci_lines") else []:
        pass
    touched = sorted({k for opts in arms.values() for k in opts})
    probe = subprocess.run(["nice", "-n", "19", a.engine], input="uci\nquit\n",
                           capture_output=True, text=True, cwd=REPO, timeout=120).stdout
    for line in probe.splitlines():
        if line.startswith("option name "):
            parts = line.split()
            nm = parts[2]
            if "default" in parts:
                defaults[nm] = parts[parts.index("default") + 1]
    missing = [k for k in touched if k not in defaults]
    if missing:
        raise SystemExit(f"no engine default for {missing} — is this a --features tune build?")

    if a.positions:
        pos = json.load(open(a.positions))
    else:
        pos = [{"fen": a.fen, "played": a.played, "best": a.oracle}]

    print(f"{len(pos)} position(s), depth {a.depth}\n")
    print(f"{'arm':<22} {'collapsed':>10} {'found':>7} {'med final':>10} {'med nodes':>12}")
    for name, opts in arms.items():
        for k in touched:
            e.setopt(k, defaults[k])
        for k, v in opts.items():
            e.setopt(k, v)
        rs = [measure(e, "archon", p["fen"], p["played"], p["best"], a.depth) for p in pos]
        coll = sum(1 for r in rs if r["collapse_d"] is not None)
        found = sum(1 for r in rs if r["find_d"] is not None)
        fin = sorted(r["final"] for r in rs if r["final"] is not None)
        nds = sorted(r["nodes"] for r in rs)
        med = lambda v: v[len(v) // 2] if v else float("nan")
        print(f"{name:<22} {coll:>6}/{len(pos):<3} {found:>4}/{len(pos):<2} "
              f"{med(fin):>10.0f} {med(nds):>12,.0f}", flush=True)
    e.close()


if __name__ == "__main__":
    sys.exit(main())
