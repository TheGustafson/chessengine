#!/usr/bin/env python3
"""tools/perf70/screen70.py — screen ordering arms with the PAIRED metric.

Replaces `screen70.sh`, whose set-total metric failed its own null (see
`paired70.py`'s header). One `arm70` build, many arms, two position sets:
each arm is compared to the SAME baseline run position by position.

  tools/perf70/screen70.py 'lva=8' 'evcap=1' ...
  NULLS=6 tools/perf70/screen70.py            # just the null panel

Env: BIN, NET, SETS ("fens:depth,fens:depth"), NULLS (extra jitter seeds).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired70 import collect, compare, fmt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

BIN = os.environ.get("BIN", "target/arm/release/chessengine")
NET = os.environ.get("NET", "engines/canonical/net.cenn")
SETS = os.environ.get(
    "SETS", "tools/perf70/uho1000.fens:10,tools/nnue/calib/wl50.fens:13"
).split(",")
NULLS = int(os.environ.get("NULLS", "4"))
# The baseline arm. Defaults to the shipped configuration; set BASE to an
# arm spec to compare a LANDED change against its own predecessor inside one
# binary (e.g. BASE="qsdef=0,evmvv=1").
BASE = os.environ.get("BASE", "")


def main():
    arms = sys.argv[1:]
    for spec in SETS:
        fens, depth = spec.rsplit(":", 1)
        name = os.path.basename(fens).split(".")[0] + "@d" + depth
        print(f"\n=== {name}  ({BIN}) ===")
        base = collect(BIN, depth, fens, NET, env={"ARCHON_A70": BASE})
        print(f"{'(baseline)':<30} n={len(base):<5} sum={sum(base)}")
        # The null panel FIRST, always: an inert +/-1 jitter can only re-break
        # exact ties, so whatever it moves is this instrument's chaos floor.
        for seed in range(1, NULLS + 1):
            a = collect(BIN, depth, fens, NET, env={"ARCHON_A70": (f"{BASE},jitter=1,jseed={seed}" if BASE else f"jitter=1,jseed={seed}")})
            print(fmt(compare(base, a, f"NULL jitter=1 seed={seed}")))
        for arm in arms:
            a = collect(BIN, depth, fens, NET, env={"ARCHON_A70": (f"{BASE},{arm}" if BASE and arm else (arm or BASE))})
            print(fmt(compare(base, a, arm)))


if __name__ == "__main__":
    main()
