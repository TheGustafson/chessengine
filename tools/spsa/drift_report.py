#!/usr/bin/env python3
"""Which SPSA parameters actually carry signal, and which are random-walking?

A finished tune hands you one vector and no error bars, which invites the
worst habit in tuning: reading every moved number as a discovery. Most of
them are noise. This separates the two using the trajectory the run already
wrote to history.jsonl.

The discriminator is DIRECTEDNESS = |net displacement| / total path length.
A parameter dragged steadily one way by a real gradient has directedness
near 1. A parameter shoved around by z-noise random-walks, and its path
length grows like sqrt(steps) while its displacement does not — so it lands
near 1/sqrt(steps), which for a few thousand iterations is a few percent.

Read it as: high directedness AND meaningful displacement = worth carrying
into the verify. Low directedness = the tuner learned nothing about this
one; keep the default rather than banking noise as a "win".

    python3 tools/spsa/drift_report.py [run-dir]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        runs = sorted((ROOT / "tools/spsa/out").glob("s2-stage*-*"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        runs = [r for r in runs if not r.name.endswith("aborted")]
        if not runs:
            print("no run directories found")
            return 1
        run = runs[0]
    hist = run / "history.jsonl"
    if not hist.exists():
        print(f"{run}: no history.jsonl yet")
        return 1

    rows = [json.loads(ln) for ln in hist.read_text().splitlines() if ln.strip()]
    if len(rows) < 2:
        print(f"{run}: only {len(rows)} iterations")
        return 1
    names = list(rows[0]["theta"])
    start = {n: rows[0]["theta"][n] for n in names}
    end = {n: rows[-1]["theta"][n] for n in names}

    path = {n: 0.0 for n in names}
    for a, b in zip(rows, rows[1:]):
        for n in names:
            path[n] += abs(b["theta"][n] - a["theta"][n])

    k = len(rows)
    games = sum(r["games"] for r in rows)
    # z should straddle zero; a persistent offset means the two sides are not
    # symmetric (colour imbalance, a broken option) rather than a gradient.
    zbar = sum(r["z"] for r in rows) / k
    zsd = math.sqrt(sum((r["z"] - zbar) ** 2 for r in rows) / max(1, k - 1))
    zerr = zsd / math.sqrt(k)

    print(f"run       {run.name}")
    print(f"iterations {k}   games {games}")
    print(f"mean z    {zbar:+.4f} +- {zerr:.4f}"
          f"   {'<-- SUSPICIOUS: should straddle 0' if abs(zbar) > 3 * zerr else '(centred, as it should be)'}")
    # Per-parameter step rms, so a displacement can be quoted in sigma:
    # a pure random walk of k steps of rms s has |displacement| ~ s*sqrt(k),
    # so net/(s*sqrt(k)) is how many sigma the drift is from "went nowhere".
    step_rms = {}
    for n in names:
        sq = sum((b["theta"][n] - a["theta"][n]) ** 2
                 for a, b in zip(rows, rows[1:]))
        step_rms[n] = math.sqrt(sq / max(1, k - 1))

    print(f"{'parameter':<26}{'start':>9}{'end':>9}{'moved':>9}"
          f"{'path':>9}{'directed':>10}{'sigma':>8}")
    order = sorted(names, key=lambda n: -abs(end[n] - start[n]) /
                   max(1e-9, step_rms[n] * math.sqrt(k)))
    for n in order:
        moved = end[n] - start[n]
        d = abs(moved) / path[n] if path[n] > 0 else 0.0
        sig = abs(moved) / (step_rms[n] * math.sqrt(k)) if step_rms[n] else 0.0
        flag = "  <-- signal" if sig > 2.0 else ""
        print(f"{n:<26}{start[n]:>9.0f}{end[n]:>9.0f}{moved:>+9.0f}"
              f"{path[n]:>9.0f}{d:>10.2f}{sig:>8.1f}{flag}")
    print(f"\nrandom-walk directedness for {k} steps is about "
          f"{1 / math.sqrt(k):.2f}; treat anything near that as noise.")
    print("sigma is the displacement against its own random-walk scale — "
          ">2 is worth carrying,\nbelow that keep the default rather than "
          "banking noise as a win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
