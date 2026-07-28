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
from math import comb
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


def measure(eng, tag, fen, played, oracle, depth, nodes=None):
    eng.newgame()
    g = eng.go(fen=fen, depth=None if nodes else depth, nodes=nodes,
               searchmoves=[played])
    lad = {d: to_sf_frame(g["ladder"][d]["score"], tag) for d in g["ladder"]}
    eng.newgame()
    f = eng.go(fen=fen, depth=None if nodes else depth, nodes=nodes)
    best = {d: f["ladder"][d]["move"] for d in f["ladder"]}
    nodes_used = f["ladder"][max(f["ladder"])]["nodes"] if f["ladder"] else 0
    # AUC: mean forced score of the PLAYED move over the deep band.  WHY this and
    # not collapse_d: a threshold crossing is chaotic in WHICH depth crosses --
    # an inert +-1 moves collapse_d on 3-7 of 15 positions -- while the mean LEVEL
    # across many depths averages that out.  Lower (more negative) is better: it
    # means the search holds a correct low opinion of the losing move over the
    # whole band, which is what stops it being played.
    top = max(lad) if lad else 0
    band = [d for d in lad if d >= max(8, top - 4)] if nodes else \
           [d for d in lad if d >= max(8, depth - 8)]
    auc = sum(lad[d] for d in band) / len(band) if band else None
    return {
        "collapse_d": collapse_depth(lad),
        "find_d": find_depth(best, oracle),
        "final": lad[max(lad)] if lad else None,
        "auc": auc,
        "best": f["best"],
        "nodes": nodes if nodes else nodes_used,
        "reached": max(lad) if lad else 0,
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
    ap.add_argument("--nodes", type=int,
                    help="EQUAL-NODE mode. Use this for any verdict about reduction or "
                         "pruning knobs: at fixed DEPTH, reducing less trivially "
                         "improves every shape metric because the arm simply searches "
                         "more (measured 2026-07-28: LMR-30%% won AUC by 37 cp while "
                         "spending +158k nodes, and lost -11.6 Elo in real games). "
                         "Equal nodes makes an arm pay for its shape.")
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
        r = measure(s, "sf", a.fen, a.played, a.oracle, a.depth, nodes=a.nodes)
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

    def run_arm(opts):
        for k in touched:
            e.setopt(k, defaults[k])
        for k, v in opts.items():
            e.setopt(k, v)
        return [measure(e, "archon", p["fen"], p["played"], p["best"], a.depth,
                        nodes=a.nodes) for p in pos]

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")

    def sign_p(up, down):
        # two-sided sign test; ties excluded, as usual
        n = up + down
        if n == 0:
            return 1.0
        tail = sum(comb(n, k) for k in range(max(up, down), n + 1))
        return min(1.0, 2 * tail / 2 ** n)

    # PAIRED against base on the SAME position.  WHY: between-position variance
    # dominates -- some losses collapse at d8, some never -- so aggregate counts
    # (11/15 vs 12/15) mix position difficulty into the arm effect and the null
    # band swallows everything.  Pairing cancels the position term.
    first = next(iter(arms))
    base = run_arm(arms[first])
    mode = f"{a.nodes:,} nodes" if a.nodes else f"depth {a.depth}"
    print(f"{len(pos)} positions, {mode}; PAIRED vs '{first}'\n")
    print(f"{'arm':<20} {'AUC better/worse':>18} {'p':>7} {'med d(AUC)':>11} "
          f"{'collapse b/w':>13} {'d depth':>7}")
    print(f"{first:<20} {'(reference)':>22}")
    BIG = 99
    for name, opts in list(arms.items())[1:]:
        rs = run_arm(opts)
        up = down = 0
        for b, r in zip(base, rs):
            bc = b["collapse_d"] if b["collapse_d"] is not None else BIG
            rc = r["collapse_d"] if r["collapse_d"] is not None else BIG
            if rc < bc:
                up += 1
            elif rc > bc:
                down += 1
        dauc = [r["auc"] - b["auc"] for b, r in zip(base, rs)
                if b["auc"] is not None and r["auc"] is not None]
        aup = sum(1 for d in dauc if d < 0)      # more negative AUC = better
        adn = sum(1 for d in dauc if d > 0)
        dnod = [r["nodes"] - b["nodes"] for b, r in zip(base, rs)]
        print(f"{name:<20} {aup:>7} / {adn:<8} {sign_p(aup, adn):>7.3f} "
              f"{med(dauc):>+11.0f} {up:>6} / {down:<4} "
              f"{med([r['reached']-b['reached'] for b,r in zip(base,rs)]):>+6.1f}", flush=True)
    e.close()


if __name__ == "__main__":
    sys.exit(main())
