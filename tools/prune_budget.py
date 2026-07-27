#!/usr/bin/env python3
"""How much of our tree does each pruning mechanism actually remove?

WHY THIS EXISTS. Three analytic attempts to correct our search margins have
now failed (docs/research/eval-frame-scale-model-2026-07-27.md sections 6.5,
10, 11): the eval frame really is a uniform ~0.64x rescale of Stockfish's,
the eval's ERROR scales with it, so in error-normalised units our margins
really are ~1.6x Stockfish's -- and shrinking them loses Elo anyway. The
surviving explanation is that the constants sit at a JOINT optimum: our tree
is already ~35% smaller than SF16's at equal depth, so we prune harder
somewhere else and the interior margins are compensating.

That story names no mechanism. This tool does. Turn ONE gate off, measure how
much the tree grows, and the growth is that gate's contribution to the tree
we actually search. The result is a budget: which mechanisms shape our tree,
in what proportion. SPSA leverage follows the budget -- tuning a gate that
removes 1% of the tree cannot pay, however mis-set it is.

METHOD, and its two traps.

  Node counts are deterministic, so one bench per arm is exact and costs
  seconds. But node count is NOT monotone in pruning aggressiveness -- TT and
  re-search feedback mean a bigger tree can follow from MORE pruning. So
  "growth when disabled" is a contribution measure, not a proof of direction,
  and a NEGATIVE growth is a real result worth reporting rather than a bug.

  The net MUST be activated with `setoption name EvalFile`. Until 2026-07-27
  the in-session `bench` parsed its optional net argument and silently ignored
  it, so a day of measurements ran on PSQT while looking entirely plausible.
  Even with that fixed, this is the form that cannot regress.

Needs a `--features tune` build (every tunable becomes a settable option):

    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --features tune
    python3 tools/prune_budget.py --engine target/release/chessengine
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

NODES_RE = re.compile(r"^Nodes searched  : (\d+)", re.M)

# (label, {param: disabling value}, note)
#
# Each disabling value is chosen from the gate's own algebra, not guessed:
#   RFP        `depth < RFP_MAX_DEPTH` -> 0 removes the gate outright
#   razoring   prune if `eval < alpha - BASE - MULT*d^2` -> huge BASE never fires
#   NMP        fires when `eval >= beta - DEPTH*d - IMPROVING + BASE`
#              -> huge BASE raises the bar out of reach
#   ProbCut    `depth >= PROBCUT_MIN_DEPTH` -> 99 removes it
#   IIR        `depth >= IIR_MIN_DEPTH` -> 99 removes it
#   LMP        `moveCount >= (LMP_BASE + d^2)/(2-improving)` -> huge never reached
#   capture &  futility prunes when `staticEval + BASE + ... <= alpha`
#   parent fut -> huge BASE lifts the bound above alpha always
#   quiet SEE  prunes if `!see_ge(m, -QSEE_MULT*lmrDepth^2)` -> huge MULT makes
#              the threshold so permissive that see_ge always holds
#   conthist   prunes if `history < -QHIST_PRUNE_MULT*depth` -> huge never true
BIG = 10_000_000
ABLATIONS: list[tuple[str, dict[str, int], str]] = [
    ("razoring", {"RAZOR_BASE": BIG}, "N9"),
    ("RFP", {"RFP_MAX_DEPTH": 0}, "N10"),
    ("NMP", {"NMP_EVAL_BASE": BIG}, "N11"),
    ("IIR", {"IIR_MIN_DEPTH": 99}, "N12"),
    ("ProbCut", {"PROBCUT_MIN_DEPTH": 99}, "N13"),
    ("LMP", {"LMP_BASE": BIG}, "step 14 quiet count"),
    ("capture futility", {"CFUT_BASE": BIG}, "step 14 capture arm"),
    ("parent futility", {"PFUT_BASE": BIG}, "step 14 quiet arm"),
    ("quiet SEE", {"QSEE_MULT": BIG}, "step 14 quiet arm"),
    ("conthist pruning", {"QHIST_PRUNE_MULT": BIG}, "step 14 quiet arm"),
    ("LMR log term", {"RED_TABLE_NUM": 0}, "approx: additive terms remain"),
]


def bench(engine: str, net: str, depth: int, fens: str,
          overrides: dict[str, int]) -> int:
    lines = [f"setoption name EvalFile value {net}"]
    lines += [f"setoption name {k} value {v}" for k, v in overrides.items()]
    lines.append(f"bench 16 1 {depth} {fens} depth")
    lines.append("quit")
    r = subprocess.run([engine], input="\n".join(lines) + "\n", text=True,
                       capture_output=True)
    # bench reports on stderr; stdout stays a clean UCI stream.
    m = NODES_RE.search(r.stdout + r.stderr)
    if not m:
        raise RuntimeError(f"no node total (overrides={overrides})")
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="target/release/chessengine")
    ap.add_argument("--net", default="data/nnue/nets/screens/d47v-a4b2-cal400.cenn")
    ap.add_argument("--fens", default="tools/nnue/calib/wl50.fens")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--only", nargs="*", help="restrict to these labels")
    args = ap.parse_args()

    if not Path(args.net).exists():
        raise SystemExit(f"net not found: {args.net}")

    base = bench(args.engine, args.net, args.depth, args.fens, {})
    print(f"baseline: {base:,} nodes   (depth {args.depth}, {args.fens})\n")
    print(f"{'mechanism':<20} {'nodes w/o it':>14} {'growth':>10} {'share':>8}  note")

    rows = []
    for label, ov, note in ABLATIONS:
        if args.only and label not in args.only:
            continue
        try:
            n = bench(args.engine, args.net, args.depth, args.fens, ov)
        except RuntimeError as e:
            print(f"{label:<20} {'FAILED':>14}   {e}")
            continue
        growth = n / base - 1.0
        rows.append((label, n, growth, note))
        print(f"{label:<20} {n:>14,} {growth:>+9.1%} {'':>8}  {note}")

    total = sum(max(0.0, g) for _, _, g, _ in rows)
    if total > 0:
        print(f"\n{'mechanism':<20} {'share of total tree reduction':>32}")
        for label, _, g, _ in sorted(rows, key=lambda r: -r[2]):
            share = max(0.0, g) / total
            bar = "#" * round(share * 40)
            print(f"{label:<20} {share:>7.1%}  {bar}")
    print(
        "\nRead with care. Node count is not monotone in pruning aggressiveness\n"
        "(TT and re-search feedback), so a negative growth is a real result and\n"
        "the shares are a contribution ranking, not an additive decomposition --\n"
        "the mechanisms interact and will not sum to the whole tree."
    )


if __name__ == "__main__":
    main()
