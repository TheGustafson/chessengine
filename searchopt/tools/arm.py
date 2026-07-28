#!/usr/bin/env python3
"""Test a tunable against a corpus WITHOUT being able to fool yourself.

    python3 searchopt/tools/arm.py --param RED_GIVES_CHECK --value 512

This tool exists because on 2026-07-28 I cleared a SEVEN-sample inert null band
with `RED_GIVES_CHECK=+512`, reported it as the session's one solid result, and
then discovered that `RED_GIVES_CHECK=-512` — reducing checks MORE — cleared the
same band. The metric was rewarding any perturbation that shook the search off
the losing move. A day's headline died to one control that should have run first.

So this runs that control automatically and REFUSES to give a verdict without it.

Three things it does that I failed to do by hand:

1. **ANTI-ARM FIRST.** The mirrored value (default + (default - value)) runs
   before the arm. If it moves the metric as much as the arm does, the metric is
   noise and the tool says so and stops. No amount of null-panel size fixes this,
   because inert +-1 nulls do not move along the arm's axis.

2. **DEFAULT RESTORATION between every arm.** Params are read from the engine's
   own `uci` output, not a hand table, and every touched param is reset before
   each arm. Measured: an arm once inherited RED_BASE=1026 from the preceding
   null and flipped its result — the "finding" was an uncontrolled +-1.

3. **A REPEAT of the base arm** at the end. If base != base, the harness itself
   is nondeterministic and every number in the run is void.

The metric is deliberately pluggable and deliberately suspicious of itself; see
`--metric`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET, REPO  # noqa: E402

NEUTRAL = ["RED_BASE", "RED_FLAT", "RED_DELTA_MULT"]   # inert +-1 null carriers


def engine_defaults(path):
    out = subprocess.run(["nice", "-n", "19", path], input="uci\nquit\n",
                         capture_output=True, text=True, cwd=REPO, timeout=180).stdout
    d = {}
    for ln in out.splitlines():
        if ln.startswith("option name "):
            p = ln.split()
            if "default" in p:
                d[p[2]] = p[p.index("default") + 1]
    return d


def metric_nodes_to_fix(e, pos, budgets):
    """Budget at which the engine stops playing the losing move, log2-summed.

    KNOWN WEAK: any perturbation that shakes the search off the losing move
    improves it, which is exactly why the anti-arm control below is mandatory."""
    tot, fixed = 0.0, 0
    for p in pos:
        got = budgets[-1] * 4
        for n in budgets:
            e.newgame()
            if e.go(fen=p["fen"], nodes=n)["best"] != p["played"]:
                got = n
                break
        tot += math.log2(got)
        fixed += (got <= budgets[1] if len(budgets) > 1 else 0)
    return {"score": tot, "fixed_at_ingame": fixed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--value", required=True, type=int)
    ap.add_argument("--positions", default="searchopt/positions/ours.json")
    ap.add_argument("--budgets", default="110000,250000,500000,1000000")
    ap.add_argument("--engine", default="engines/bin/archon-trace",
                    help="MUST be a --features tune build or every arm is identical")
    ap.add_argument("--nulls", type=int, default=6)
    a = ap.parse_args()
    os.chdir(REPO)
    budgets = [int(x) for x in a.budgets.split(",")]
    pos = json.load(open(a.positions))

    defaults = engine_defaults(a.engine)
    if a.param not in defaults:
        print(f"FAIL: {a.param} is not a UCI option on {a.engine}.")
        print("      Almost certainly a non-tune build — params are inlined constants")
        print("      without `--features tune`, and every arm would return identical results.")
        return 1
    dflt = int(defaults[a.param])
    anti = dflt + (dflt - a.value)

    arms = [("base", {}), ("ANTI", {a.param: anti}), ("arm", {a.param: a.value})]
    for i in range(a.nulls):
        p = NEUTRAL[i % len(NEUTRAL)]
        if p in defaults:
            arms.append((f"null {p}{'+' if i // len(NEUTRAL) else '-'}1",
                         {p: int(defaults[p]) + (1 if i // len(NEUTRAL) else -1)}))
    arms.append(("base(repeat)", {}))

    touched = sorted({k for _, o in arms for k in o})
    e = Engine(a.engine, net=NET, hash_mb=64, name="arm"); e.assert_net_live()
    print(f"{a.param}: default {dflt}   arm {a.value}   ANTI {anti}   "
          f"({len(pos)} positions, budgets {'/'.join(str(b//1000)+'k' for b in budgets)})\n")
    res = {}
    for name, ov in arms:
        for k in touched:
            e.setopt(k, defaults[k])          # restore EVERY touched param first
        for k, v in ov.items():
            e.setopt(k, v)
        r = metric_nodes_to_fix(e, pos, budgets)
        res[name] = r
        print(f"  {name:<16} score {r['score']:8.1f}   fixed@in-game {r['fixed_at_ingame']:>3}",
              flush=True)
    e.close()

    print()
    if abs(res["base"]["score"] - res["base(repeat)"]["score"]) > 1e-9:
        print("VOID: base did not reproduce itself. The harness is nondeterministic;")
        print("      every number above is meaningless. Fix that before interpreting.")
        return 2

    b, arm, anti_s = res["base"]["score"], res["arm"]["score"], res["ANTI"]["score"]
    nulls = [v["score"] for k, v in res.items() if k.startswith("null")]
    d_arm, d_anti = b - arm, b - anti_s          # positive = better than base
    band = (min(nulls) - b, max(nulls) - b) if nulls else (0, 0)

    print(f"  arm  moves the metric by {d_arm:+.1f}")
    print(f"  ANTI moves the metric by {d_anti:+.1f}   <- the control")
    print(f"  inert null band          [{band[0]:+.1f}, {band[1]:+.1f}]  (n={len(nulls)})")
    print()
    if d_anti > 0 and abs(d_anti) >= 0.5 * abs(d_arm):
        print("VERDICT: METRIC INVALID. The OPPOSITE change helps comparably, so this")
        print("         metric rewards perturbation, not quality. Do not report the arm.")
        print("         Find a proof-grade outcome (a forced mate, or a value SF confirms).")
        return 3
    if d_arm <= max(0.0, band[1] - band[0]):
        print("VERDICT: inside the inert null band. No effect distinguishable from noise.")
        return 0
    print("VERDICT: clears the null band AND the anti-arm control.")
    print("         This is a CANDIDATE, not a result — games arbitrate (1200 @ 10+0.1,")
    print("         same binary both sides, differing only by setoption).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
