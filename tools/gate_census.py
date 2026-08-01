#!/usr/bin/env python3
"""Aggregate the per-gate SSTATS/SSTATS3 census over a bench run.

Search-simplification study, 2026-08-01. Runs `bench` on a `--features
search-stats` build with ARCHON_SEARCH_STATS=1 (argv bench — NEVER piped
stdin, which reads as a STOP and returns all zeros; searchopt/LEDGER.md:1242),
sums every counter across positions, and prints each gate's fire count per
1000 main-search node entries (`main_entry`, the denominator the census
defines) and per 1000 qsearch entries for the qs_* family.

    python3 tools/gate_census.py --engine target/release/chessengine \
        --net /abs/path/net.cenn --depth 12 --fens tools/nnue/calib/wl50.fens
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess

PAIR_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="target/release/chessengine")
    ap.add_argument("--net", required=True)
    ap.add_argument("--fens", default="tools/nnue/calib/wl50.fens")
    ap.add_argument("--depth", type=int, default=12)
    args = ap.parse_args()

    env = dict(os.environ, ARCHON_SEARCH_STATS="1")
    r = subprocess.run(
        [args.engine, "bench", "16", "1", str(args.depth), args.fens, "depth", args.net],
        capture_output=True,
        text=True,
        env=env,
    )
    sums: dict[str, float] = {}
    for line in (r.stdout + r.stderr).splitlines():
        if not line.startswith("SSTATS"):
            continue
        for k, v in PAIR_RE.findall(line.split(" ", 1)[1]):
            # Skip the per-position derived ratios; recompute from sums.
            if k in ("qshare", "fhf_rate", "avg_moves_at_cut", "lmr_avg_plies", "nmp_rate", "depth"):
                continue
            sums[k] = sums.get(k, 0.0) + float(v)

    main_n = sums.get("main_entry", 0.0) or 1.0
    qs_n = sums.get("qs_entry", 0.0) or 1.0
    nodes = sums.get("nodes", 0.0) or 1.0
    print(f"nodes={nodes:.0f} main_entries={main_n:.0f} qs_entries={qs_n:.0f} "
          f"(depth {args.depth}, {args.fens})")
    print(f"{'gate':<16}{'count':>12}{'/1k main':>12}{'/1k qs':>10}")
    for k in sorted(sums, key=lambda k: -sums[k]):
        if k in ("nodes", "main_entry", "qs_entry"):
            continue
        per_main = 1000.0 * sums[k] / main_n
        per_qs = 1000.0 * sums[k] / qs_n if k.startswith("qs_") else float("nan")
        qs_col = f"{per_qs:>10.2f}" if k.startswith("qs_") else f"{'':>10}"
        print(f"{k:<16}{sums[k]:>12.0f}{per_main:>12.2f}{qs_col}")


if __name__ == "__main__":
    main()
