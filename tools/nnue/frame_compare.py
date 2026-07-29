#!/usr/bin/env python3
"""Paired eval-frame comparison across nets — does a candidate share a FRAME
with the incumbent, or only a GAIN?

Why this exists (2026-07-29). Every cp-denominated search margin (razoring
`465 + 300 d^2`, futility, delta, aspiration) is co-adapted to the incumbent
net's output SHAPE. `network_scale` is a single scalar, so it can only correct a
GAIN difference. If a candidate's eval differs in CURVATURE, calibration aligns
the median and leaves the tails mismatched, and a genuinely better net can lose
a match on margin mistuning alone — measured at **-12.0 +/- 18.2 Elo** on the
Reckless frame-null, with zero net-quality change.

The tell is LS-vs-median disagreement on the paired ratio:

    ARCHON vs SF        LS 0.5126 / median 0.5096   0.6% apart  -> scalar FINE
    ARCHON vs Reckless  LS 0.7316 / median 0.554    1.32x apart -> SHAPE MISMATCH

Usage (all three pairs, which is the point):

    .venv/bin/python tools/nnue/frame_compare.py --count 400 \
      --net D48X=archon:data/nnue/nets/screens/d48x-1280-sb520-ema.cenn \
      --net D48Y=archon:data/nnue/nets/screens/d48y-1280-sb300-ema.cenn \
      --net SF=sfnet:/home/nick/sf-eca43a97/src/nn-0ee0657fb25e.nnue

`archon:` uses `engines/canonical/archon nnue-eval <cenn>` (stm-POV).
`sfnet:`  uses the firewalled sf-swap binary's `sf-eval` (stm-POV, SF units).
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHON = str(ROOT / "engines/canonical/archon")
SFSWAP = str(ROOT / "engines/bin/archon-sfswapbin")


def positions(count: int, seed: int, pgn_path: str) -> list[str]:
    """Sample positions across the whole game, not just openings — the tails of
    the eval distribution are exactly where a shape mismatch shows up."""
    rng = random.Random(seed)
    out: list[str] = []
    with open(pgn_path, errors="ignore") as fh:
        while len(out) < count:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            b = g.board()
            moves = list(g.mainline_moves())
            if len(moves) < 20:
                continue
            want = rng.sample(range(8, len(moves)), min(3, len(moves) - 8))
            for i, m in enumerate(moves):
                b.push(m)
                if i in want and not b.is_check() and not b.is_game_over():
                    out.append(b.fen())
    return out[:count]


def eval_archon(net: str, fens: list[str]) -> list[float]:
    r = subprocess.run([ARCHON, "nnue-eval", net], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, timeout=3600)
    return [float(m.group(1)) for ln in r.stdout.splitlines()
            if (m := re.match(r"(-?\d+)(?:\t|$)", ln.strip()))]


def eval_sfnet(net: str, fens: list[str]) -> list[float]:
    r = subprocess.run([SFSWAP, "sf-eval", net], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, timeout=3600)
    return [float(ln) for ln in r.stdout.splitlines()
            if ln.strip().lstrip("-").isdigit()]


KINDS = {"archon": eval_archon, "sfnet": eval_sfnet}


def compare(a_name: str, a: np.ndarray, b_name: str, b: np.ndarray) -> None:
    m = (np.abs(a) > 30) & (np.abs(b) > 30) & (np.abs(a) < 3000) & (np.abs(b) < 3000)
    if m.sum() < 30:
        print(f"  {a_name} vs {b_name}: too few usable pairs ({m.sum()})")
        return
    x, y = b[m], a[m]                      # a ~ k * b
    ls = float(x @ y / (x @ x))
    rat = y / x
    med = float(np.median(rat))
    q1, q3 = np.percentile(rat, [25, 75])
    disagree = max(ls, med) / max(min(ls, med), 1e-9)
    verdict = ("SCALAR FINE" if disagree < 1.05 else
               "borderline" if disagree < 1.15 else "SHAPE MISMATCH")
    print(f"  {a_name:>6} vs {b_name:<6} n={int(m.sum()):4d}  LS {ls:.4f}  "
          f"median {med:.4f}  IQR [{q1:.3f},{q3:.3f}]  "
          f"pearson {np.corrcoef(y, x)[0, 1]:.4f}  "
          f"LS/med {disagree:.3f}x  -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", action="append", required=True,
                    metavar="NAME=KIND:PATH", help="repeatable; kind in " + "/".join(KINDS))
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--pgn", default=str(ROOT / "engines/runs/corpus-d48x/games.pgn"))
    a = ap.parse_args()

    fens = positions(a.count, a.seed, a.pgn)
    print(f"positions: {len(fens)}  (whole-game sample, seed {a.seed})", flush=True)

    series: dict[str, np.ndarray] = {}
    for spec in a.net:
        name, _, rest = spec.partition("=")
        kind, _, path = rest.partition(":")
        if kind not in KINDS:
            sys.exit(f"unknown kind {kind!r}; have {list(KINDS)}")
        if not os.path.exists(path):
            print(f"  !! {name}: {path} missing — SKIPPED", file=sys.stderr)
            continue
        v = KINDS[kind](path, fens)
        if len(v) != len(fens):
            print(f"  !! {name}: got {len(v)}/{len(fens)} — SKIPPED", file=sys.stderr)
            continue
        series[name] = np.array(v, dtype=float)
        print(f"  {name}: {len(v)} evals", flush=True)

    names = list(series)
    print("\npaired frame relations (a ~ k * b):")
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            compare(n1, series[n1], n2, series[n2])
    print("\nreference: ARCHON vs SF was LS 0.5126 / median 0.5096 (0.6% apart, scalar fine);")
    print("           ARCHON vs Reckless was LS 0.7316 / median 0.554 (1.32x, shape mismatch).")
    print("If D48Y-vs-SF disagrees LESS than D48X-vs-SF, the new target moved us")
    print("TOWARD Stockfish's shape and our SF-derived margins should fit it better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
