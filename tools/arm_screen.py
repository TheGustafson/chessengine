#!/usr/bin/env python3
"""Screen tunable configurations on both axes that matter, in one command.

WHY BOTH AXES. A pruning change moves two things in opposition: how big the
tree is, and how good the decisions in it are. Measuring either alone gives
the wrong answer, and we have now made that mistake in both directions:

  * Fixed-DEPTH tactics measures decisions per depth. A search that prunes
    more can only look worse, because the benefit -- reaching further on the
    same budget -- is normalised away. This is why razoring-off looked like a
    pure win and why the RFP correction looked survivable.
  * Node count alone ignores whether the smaller tree still finds the move.
    RFP at 0.55x saves 12.8% of nodes and loses 23 tactical solutions.

So every arm is reported as (nodes at fixed depth, solutions at fixed NODES).
Fixed nodes is the honest tactical normalisation: same budget, does this
configuration use it better? An arm must be non-negative on both to be worth
a game.

WHY DETERMINISTIC INSTRUMENTS. The box usually has a trainer on it, so
wall-clock is unusable. Node counts at fixed depth and solution counts at
fixed nodes are both exactly reproducible under load, which is what makes a
long reason/measure loop possible at all.

    python3 tools/arm_screen.py \
        'combo:NMP_EVAL_BASE=168,RAZOR_BASE=3000' \
        'aggressive:NMP_EVAL_BASE=120,RAZOR_BASE=3000'

The control arm (no overrides) is always measured first and everything is
reported relative to it. `--scale FAMILY=FACTOR` expands a named family to
its member constants scaled by that factor, rounding half AWAY from zero --
truncation silently turned SE_DOUBLE_BASE -3 into -1 once already, a 0.33
ratio where 0.58 was intended.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess

NODES_RE = re.compile(r"^Nodes searched  : (\d+)", re.M)
NET = "engines/canonical/net.cenn"
ENGINE = "target/release/chessengine"
FENS = "tools/nnue/calib/scale-check-800.fens"

# Families of constants that must scale TOGETHER because they share units.
# Only eval-denominated members are listed: a divisor, a ply count or a
# history weight in the same formula is a different class and must not ride
# the same factor (design note: eval-frame-scale-model §4 taxonomy).
FAMILIES: dict[str, dict[str, int]] = {
    "RFP": {"RFP_MULT_BASE": 40, "RFP_MULT_SLOPE": 4, "RFP_MULT_CAP": 80,
            "RFP_MULT_NO_TTHIT": 20},
    "PFUT": {"PFUT_BASE": 40, "PFUT_NO_BESTMOVE": 138, "PFUT_LMRDEPTH_MULT": 117,
             "PFUT_EVAL_ABOVE_ALPHA": 90},
    "CFUT": {"CFUT_BASE": 231, "CFUT_LMRDEPTH_MULT": 232},
    "PROBCUT": {"PROBCUT_BETA_BASE": 214, "PROBCUT_BETA_IMPROVING": 59,
                "PROBCUT_CHECK_BETA": 428},
    "SE": {"SE_BETA_MULT": 60, "SE_BETA_TTPV_MULT": 70, "SE_DOUBLE_BASE": -3,
           "SE_DOUBLE_PV": 201, "SE_DOUBLE_NON_TTCAP": 157, "SE_DOUBLE_DEEP": 41,
           "SE_TRIPLE_BASE": 72, "SE_TRIPLE_PV": 306, "SE_TRIPLE_NON_TTCAP": 188,
           "SE_TRIPLE_TTPV": 84, "SE_TRIPLE_DEEP": 45},
    "RAZOR": {"RAZOR_BASE": 465, "RAZOR_DEPTH_MULT": 300},
    "QSFUT": {"QS_FUTILITY_MARGIN": 335},
    "NMP": {"NMP_EVAL_BASE": 374, "NMP_EVAL_DEPTH": 14, "NMP_EVAL_IMPROVING": 45},
    # The twelve eval-denominated constants that were in NONE of the families
    # above -- so the five refuted frame formulations (global, dual,
    # per-family, per-mechanism, class-4) never moved them. Found 2026-07-27
    # by deriving the constant list from the `Value` type instead of a
    # hand-written table; see tools/frame_audit.py, which now exits non-zero on
    # an unclassified tunable so this hole cannot reopen.
    #
    # The prior that these matter is LOW -- five formulations found nothing --
    # with one exception worth separating out: HINDSIGHT_EVAL_SUM gates a
    # depth+-1, not a margin, and a ply is a much larger lever than a few
    # centipawns of threshold.
    "FRAME12": {"ASPIRATION_BASE_DELTA": 5, "EVALDIFF_CLAMP_HI": 180,
                "EVALDIFF_CLAMP_LO": -183, "EVALDIFF_OFFSET": 62,
                "FL_BELOW_EVAL_MARGIN": 103, "FL_BELOW_PARENT_EVAL_MARGIN": 78,
                "HINDSIGHT_EVAL_SUM": 173, "LMR_DEEPER_MARGIN": 52,
                "LMR_SHALLOWER_MARGIN": 9, "OPTIMISM_DEN_BASE": 45,
                "OPTIMISM_NUM": 75, "RFP_THREAT_MARGIN": 55},
}


def round_away(x: float) -> int:
    """Round half away from zero. Truncation here is a real bug, not a nicety."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def parse_arm(spec: str) -> tuple[str, dict[str, int]]:
    """`name:K=V,K=V,FAMILY*0.64` -> (name, overrides)."""
    name, _, body = spec.partition(":")
    ov: dict[str, int] = {}
    for tok in filter(None, (t.strip() for t in body.split(","))):
        if "*" in tok:
            fam, _, f = tok.partition("*")
            fam = fam.strip()
            if fam not in FAMILIES:
                raise SystemExit(f"unknown family {fam!r}; have {sorted(FAMILIES)}")
            for k, v in FAMILIES[fam].items():
                ov[k] = round_away(v * float(f))
        else:
            k, _, v = tok.partition("=")
            ov[k.strip()] = int(v)
    return name, ov


def bench(depth: int, ov: dict[str, int]) -> int:
    lines = [f"setoption name EvalFile value {NET}"]
    lines += [f"setoption name {k} value {v}" for k, v in ov.items()]
    lines.append(f"bench 16 1 {depth} {FENS} depth")
    lines.append("quit")
    r = subprocess.run([ENGINE], input="\n".join(lines) + "\n", text=True,
                       capture_output=True)
    m = NODES_RE.search(r.stdout + r.stderr)  # bench reports on stderr
    if not m:
        raise RuntimeError(f"no node total for {ov}")
    return int(m.group(1))


def tactics(suite: str, nodes: int, ov: dict[str, int]) -> tuple[int, int]:
    cmd = ["python3", "tools/tactics.py", ENGINE, suite, "--nodes", str(nodes),
           "--option", f"EvalFile={NET}"]
    for k, v in ov.items():
        cmd += ["--option", f"{k}={v}"]
    out = subprocess.run(cmd, text=True, capture_output=True).stdout
    m = re.search(r": (\d+)/(\d+)", out)
    if not m:
        raise RuntimeError(f"no tactics result for {ov}: {out[-300:]}")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name:K=V,FAMILY*FACTOR")
    ap.add_argument("--depths", default="10", help="comma-separated bench depths")
    ap.add_argument("--suite", default="engines/suites/mate_in_3.epd")
    ap.add_argument("--tactic-nodes", type=int, default=50000)
    ap.add_argument("--no-tactics", action="store_true")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    arms = [("control", {})] + [parse_arm(a) for a in args.arms]

    base_nodes = {d: bench(d, {}) for d in depths}
    base_hits = None
    if not args.no_tactics:
        base_hits, total = tactics(args.suite, args.tactic_nodes, {})

    hdr = f"{'arm':<22}" + "".join(f"{'d'+str(d):>12}" for d in depths)
    if base_hits is not None:
        hdr += f"{'tactics':>10}{'delta':>7}"
    print(hdr)
    for name, ov in arms:
        row = f"{name:<22}"
        for d in depths:
            n = bench(d, ov) if ov else base_nodes[d]
            row += f"{n/base_nodes[d]-1:>+11.2%} " if ov else f"{'(base)':>12}"
        if base_hits is not None:
            h = base_hits if not ov else tactics(args.suite, args.tactic_nodes, ov)[0]
            row += f"{h:>7}/{total}" if not ov else f"{h:>10}"
            row += f"{'':>7}" if not ov else f"{h-base_hits:>+7}"
        print(row, flush=True)
    print("\nAn arm must be non-positive on nodes AND non-negative on tactics to\n"
          "be worth a game. Node counts are exact; tactics sigma ~ sqrt(n*p*(1-p)).")


if __name__ == "__main__":
    main()
