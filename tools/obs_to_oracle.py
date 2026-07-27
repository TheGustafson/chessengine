#!/usr/bin/env python3
"""Turn a `.ceobs` root-analysis shard into a tiered oracle cache.

Why this beats building an oracle live (`move_quality.py --build-oracle`):

  * EVERY legal move is scored, not a MultiPV top-K. That removes censoring
    entirely — the live oracle floors any move outside its top-K and therefore
    UNDERSTATES loss exactly where a config is behaving worst.
  * 10,000 positions instead of a few hundred, so the standard error on a mean
    cp loss falls by ~6x and small deltas become readable.
  * Three node TIERS (4k/16k/64k) of the same positions from the same oracle.
    That is the diagnostic axis: an arm that improves against the 4k labels
    while worsening against 64k is trading deep mistakes for shallow ones,
    which a single-tier screen reports as a wash.

The shard is `arm_mode: searchmoves` — Stockfish searched the root restricted
to each move in turn — so `score` is root-relative from the side to move, and
the best arm's score is the position's value. Loss for a played move is
therefore `max_over_arms(score) - score(played)`, in SF18 centipawns.

    ./target/release/obs-tool export data/obs/big-v1-sf18.ceobs \\
        | python3 tools/obs_to_oracle.py --out tools/nnue/calib/oracle-obs.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/nnue/calib/oracle-obs.json")
    ap.add_argument("--max-positions", type=int, default=0,
                    help="cap positions per tier (0 = all)")
    ap.add_argument("--min-moves", type=int, default=2,
                    help="skip positions with fewer scored arms than this — "
                    "a position with one legal move measures nothing")
    args = ap.parse_args()

    tiers: dict[str, dict[str, dict[str, int]]] = {}
    rows = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        # `complete: false` means the arm's search was cut short, so its score
        # is not comparable with its siblings. Dropping it is not optional:
        # a truncated arm can look like the best move.
        if not d.get("complete", True):
            continue
        t = str(d["tier"])
        tiers.setdefault(t, {}).setdefault(d["fen"], {})[d["mv"]] = d["score"]
        rows += 1
        if rows % 200_000 == 0:
            print(f"  {rows} rows", file=sys.stderr, flush=True)

    out = {}
    for t, table in tiers.items():
        keep = {f: mv for f, mv in table.items() if len(mv) >= args.min_moves}
        if args.max_positions:
            keep = dict(list(keep.items())[: args.max_positions])
        out[t] = keep
        print(f"tier {t}: {len(keep)} positions, "
              f"{sum(len(m) for m in keep.values())} scored moves",
              file=sys.stderr)

    json.dump({"source": "ceobs", "tiers": out}, open(args.out, "w"))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
