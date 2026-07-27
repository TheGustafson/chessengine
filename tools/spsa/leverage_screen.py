#!/usr/bin/env python3
"""Per-parameter LEVERAGE screen for an SPSA campaign.

The question this answers: when SPSA nudges a parameter by one `c_end`, how
much does the search tree actually move? Node counts at fixed depth are a
pure function of search behaviour, so this is deterministic and completely
immune to box load — unlike Elo, it can be measured on a busy machine.

Two things come out of it.

**Dead dimensions.** A parameter whose +-c_end nudge leaves the node count
bit-identical cannot affect the search at all, so it contributes nothing to
the SPSA gradient but still soaks up perturbation budget: with n parameters
the cross-term noise on each one grows like sqrt(n-1), so a dead dimension
actively degrades every other parameter's signal. Drop them.

**A regime warning.** This began as a calibration tool — the usual
`c_end = (max - min) / 20` is a rule about the hand-set RANGE and says
nothing about effect size, so it seemed obvious to rescale every c_end to a
common leverage. Measuring it killed that idea. On c2-reduction the leverage
RANKING barely survives a change of depth (Spearman +0.21 for d6 vs d8,
+0.20 d6 vs d10, +0.12 d8 vs d10): `LMR_DEEPER_MARGIN` is last at d6 and
4th at d10, `RED_MOVECOUNT` is 12th at d6 and 1st at d8. Partly that is
genuine — reduction parameters are depth-scaled by construction — and partly
the response surface is chaotic. Either way a single-depth screen cannot
calibrate anything, so the script now probes several depths and reports the
rank correlation instead of proposing new c_end values.

What it still tells you reliably is the DEAD list, and — more usefully — how
far apart the tuning and deployment regimes are. That gap is real here:
fn20k is about depth 12, while 10+0.1 at ~1.2M nps and ~265 ms/move reaches
roughly depth 18.

A caveat worth stating plainly: tree movement is NOT Elo. A parameter can
reshape the tree and be Elo-neutral. Zero leverage proves irrelevance; high
leverage only proves the search noticed. Never quote this as a result.

    python3 tools/spsa/leverage_screen.py tools/spsa/campaigns/c2-reduction.json \
        --engine engines/bin/archon-d47v-tune-pgo \
        --net data/nnue/nets/screens/d47v-a4b2-cal400.cenn --depth 6 8 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
from pathlib import Path

NODES_RE = re.compile(r"^Nodes searched  : (\d+)", re.M)
OPTION_RE = re.compile(r"option name (\S+) type string default (-?\d+)")


def engine_defaults(engine: Path) -> dict[str, int]:
    """Read theta-start from the binary itself rather than trusting a doc."""
    out = subprocess.run([str(engine)], input="uci\nquit\n", text=True,
                         capture_output=True).stdout
    return {m.group(1): int(m.group(2)) for m in OPTION_RE.finditer(out)}


def bench_nodes(engine: Path, net: str | None, depth: int,
                overrides: dict[str, int]) -> int:
    lines = []
    if net:
        lines.append(f"setoption name EvalFile value {net}")
    lines += [f"setoption name {k} value {v}" for k, v in overrides.items()]
    lines.append(f"bench 16 1 {depth} default depth")
    lines.append("quit")
    r = subprocess.run([str(engine)], input="\n".join(lines) + "\n", text=True,
                       capture_output=True)
    # bench reports on stderr; UCI keeps stdout clean.
    m = NODES_RE.search(r.stdout + r.stderr)
    if not m:
        raise RuntimeError(f"no node total from {engine} (overrides={overrides})")
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--engine", type=Path,
                    default=Path("engines/bin/archon-d47v-tune-pgo"))
    ap.add_argument("--net", default="data/nnue/nets/screens/d47v-a4b2-cal400.cenn",
                    help="empty string to screen on PSQT instead")
    ap.add_argument("--depth", type=int, nargs="+", default=[6, 8, 10],
                    help="one or more depths; several is STRONGLY preferred — "
                         "leverage is depth-unstable and a single depth will "
                         "mislead you (measured 2026-07-27: rank correlation "
                         "between d6/d8/d10 is only ~0.2)")
    args = ap.parse_args()

    params = json.loads(args.campaign.read_text())["params"]
    defaults = engine_defaults(args.engine)
    missing = [p["name"] for p in params if p["name"] not in defaults]
    if missing:
        raise SystemExit(f"engine exposes no option for: {missing} "
                         f"(is this a --features tune build?)")

    lev: dict[int, dict[str, float]] = {}
    for depth in args.depth:
        base = bench_nodes(args.engine, args.net, depth, {})
        print(f"depth {depth}: baseline {base:,} nodes "
              f"({'NNUE' if args.net else 'PSQT'})")
        per: dict[str, float] = {}
        for p in params:
            name = p["name"]
            c_end = p.get("c_end", (p["max"] - p["min"]) / 20.0)
            theta = defaults[name]
            # Clamp exactly as the tuner does, so this measures the
            # perturbation the campaign can really apply.
            lo = max(p["min"], round(theta - c_end))
            hi = min(p["max"], round(theta + c_end))
            n_lo = bench_nodes(args.engine, args.net, depth, {name: lo})
            n_hi = bench_nodes(args.engine, args.net, depth, {name: hi})
            per[name] = max(abs(n_lo - base), abs(n_hi - base)) / base
        lev[depth] = per

    order = sorted(lev[args.depth[0]], key=lambda n: -lev[args.depth[0]][n])
    head = "".join(f"{'d' + str(d):>9}" for d in args.depth)
    print(f"\n{'parameter':26s}{head}")
    for name in order:
        cells = "".join(f"{lev[d][name] * 100:8.2f}%" for d in args.depth)
        print(f"{name:26s}{cells}")

    # DEAD means dead at EVERY depth probed. A parameter inert at shallow
    # depth is routinely alive deeper (LMR_DEEPER_MARGIN is the type case).
    dead = [n for n in order if all(lev[d][n] == 0.0 for d in args.depth)]
    print(f"\nDEAD at every depth (safe to drop): {dead if dead else 'none'}")

    # The stability check is the point of probing several depths. If the
    # ranking does not hold, no single-depth screen can calibrate anything —
    # measured 2026-07-27 on c2-reduction, Spearman ~0.2 across d6/d8/d10,
    # so this is the expected outcome rather than a surprise.
    def spearman(a: dict[str, float], b: dict[str, float]) -> float:
        keys = list(a)
        n = len(keys)
        ra = {k: i for i, k in enumerate(sorted(keys, key=lambda x: -a[x]))}
        rb = {k: i for i, k in enumerate(sorted(keys, key=lambda x: -b[x]))}
        return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in keys) / (n * (n * n - 1))

    if len(args.depth) > 1:
        print("\nrank stability across depths (Spearman):")
        worst = 1.0
        for d1, d2 in itertools.combinations(args.depth, 2):
            rho = spearman(lev[d1], lev[d2])
            worst = min(worst, rho)
            print(f"  d{d1} vs d{d2}: {rho:+.3f}")
        if worst < 0.5:
            print(
                "\n  UNSTABLE. Leverage does not transfer across depth, so do NOT\n"
                "  rescale c_end from one depth's numbers — you would be fitting\n"
                "  noise plus genuine depth-scaling. The usable conclusion is the\n"
                "  DEAD list above; everything else here is a regime warning, not\n"
                "  a calibration. It also means the tuning regime matters: tuning\n"
                "  at a node count far below the deployment depth optimises\n"
                "  parameters whose sensitivity structure is not the one that will\n"
                "  be played."
            )


if __name__ == "__main__":
    main()
