#!/usr/bin/env python3
"""Ablate ONE eval-frame constant at a time and measure its isolated effect.

WHY. The FRAME_S sweep moves every wired constant together, so it can tell us
whether the eval-frame model is right in aggregate but never which constants
carry the effect — or which were misclassified. For the contentious ones we
need to probe each directly.

THE TRICK: no extra wiring is required. Every search constant is already an
individually settable tunable (`setoption name RAZOR_BASE value N`) in a
`--features tune` build. So an ablation is just: leave FRAME_S at 1024
(identity, engine unchanged) and set the ONE constant to its transformed
value. Perfect isolation, zero code risk, and it works on constants nobody has
wired yet.

THIS IS ALSO A CLASS TEST, which is its real value. Each class predicts a
DIRECTION for the node count:

  class 1/2 at s<1  -> margins shrink -> more pruning -> FEWER nodes
  class 3   at s<1  -> divisor grows  -> smaller eval adjustment -> depends on
                       sign of the adjustment; report, do not assume
  class 5           -> no change at all; if nodes move, it is NOT class 5

A constant whose measured direction contradicts its assigned class is
misclassified, and wiring it into the global sweep would push that part of the
correction backwards. Node counts are deterministic, so this is exact and
costs seconds — unlike games, which cannot resolve a single constant at all.

    python3 tools/frame_ablate.py --s 0.58 RAZOR_BASE RFP_MULT_CAP ...
    python3 tools/frame_ablate.py --s 0.58 --expand SE_CORR_DIV
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

NODES_RE = re.compile(r"^Nodes searched  : (\d+)", re.M)
OPTION_RE = re.compile(r"option name (\S+) type string default (-?\d+)")


def defaults(engine: str) -> dict[str, int]:
    out = subprocess.run([engine], input="uci\nquit\n", text=True,
                         capture_output=True).stdout
    return {m.group(1): int(m.group(2)) for m in OPTION_RE.finditer(out)}


def bench(engine: str, net: str, depth: int, fens: str,
          overrides: dict[str, int]) -> int:
    """Run one bench with `overrides` applied.

    The net MUST be activated by `setoption name EvalFile`, not by bench's
    6th argument. Until 2026-07-27 the in-session `bench` parsed that argument
    and then silently ignored it, so every measurement driven over a UCI pipe
    ran on PSQT while looking entirely plausible — a PSQT tree is a perfectly
    reasonable-looking tree. The engine bug is fixed, but setting EvalFile
    explicitly is the form that cannot regress, and it works on old binaries
    too.
    """
    lines = [f"setoption name EvalFile value {net}"]
    lines += [f"setoption name {k} value {v}" for k, v in overrides.items()]
    # No net on the bench line — EvalFile above is what activates it.
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
    ap.add_argument("params", nargs="+")
    ap.add_argument("--s", type=float, default=0.58,
                    help="frame factor; class 1/2 multiply by it")
    ap.add_argument("--expand", action="store_true",
                    help="treat the named params as class 3 (divide by s)")
    ap.add_argument("--engine", default="engines/bin/archon-d47v-tune-pgo")
    ap.add_argument("--net", default="data/nnue/nets/screens/d47v-a4b2-cal400.cenn")
    ap.add_argument("--fens", default="tools/nnue/calib/wl50.fens")
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()

    if not Path(args.engine).exists():
        raise SystemExit(f"{args.engine} missing — build --features sota,arch-d47v,tune")
    dflt = defaults(args.engine)
    missing = [p for p in args.params if p not in dflt]
    if missing:
        raise SystemExit(f"not tunables in this binary: {missing}")

    base = bench(args.engine, args.net, args.depth, args.fens, {})
    kind = "class 3 (expand, /s)" if args.expand else "class 1/2 (scale, *s)"
    print(f"baseline {base:,} nodes   s={args.s}   treating as {kind}")
    print(f"\n{'parameter':26s} {'default':>9} {'ablated':>9} {'nodes':>11} "
          f"{'delta':>9}  direction")

    for p in args.params:
        d = dflt[p]
        new = round(d / args.s) if args.expand else round(d * args.s)
        if new == d:
            print(f"{p:26s} {d:9d} {new:9d} {'—':>11} {'—':>9}  no-op at this s")
            continue
        n = bench(args.engine, args.net, args.depth, args.fens, {p: new})
        delta = (n / base - 1) * 100
        # A class-1 margin made smaller should prune MORE, i.e. fewer nodes.
        if abs(delta) < 0.01:
            verdict = "INERT — mis-scoped, gated off, or class 5"
        elif delta < 0:
            verdict = "shrinks tree"
        else:
            verdict = "GROWS tree — see the caveat below"
        print(f"{p:26s} {d:9d} {new:9d} {n:11,} {delta:+8.2f}%  {verdict}")

    print(
        "\nHOW MUCH THIS PROVES — measured 2026-07-27, and it is less than it looks.\n"
        "\n"
        "INERT is solid and actionable: a constant that cannot move the tree at all\n"
        "is mis-scoped, gated off, or class 5, and wiring it adds risk for nothing.\n"
        "\n"
        "But 'GROWS tree' does NOT establish a misclassification, because node count\n"
        "is not monotone in pruning aggressiveness. Pruning harder returns a cutoff\n"
        "score earlier, which changes what lands in the TT, which changes move\n"
        "ordering and re-searches everywhere above — a second-order feedback that can\n"
        "easily outweigh the first-order saving. QS_FUTILITY_MARGIN is the proof:\n"
        "it is unambiguously eval-denominated (`futility_base = static_eval + C`),\n"
        "yet shrinking it by 0.58 GREW the tree 3.8%. That cannot be a units error.\n"
        "\n"
        "So use this tool for what it can decide — which constants are reachable at\n"
        "all, and their effect SIZE — and settle direction from the use-site units\n"
        "plus games. A large |delta| means the constant matters and deserves care;\n"
        "a positive delta is a flag to investigate, not a verdict."
    )


if __name__ == "__main__":
    main()
