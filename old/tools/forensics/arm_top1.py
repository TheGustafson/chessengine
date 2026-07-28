"""Screen tunable arms at a REALISTIC node budget, judged by top-1 agreement.

The companion to `deep_top1.py`.  Everything else in this repo screens arms at
12,500 nodes against a 64k oracle; this runs them at the budget our games
actually use, against an oracle deep enough to outrank them.

Reports agreement per phase bucket as well as overall, because the 2026-07-28
measurements showed the verdict is phase-dependent AND budget-dependent: at
12.5k we trail Stockfish in every bucket, at 200k we match it in the middlegame
and trail only at 8-12 pieces.  An arm that helps one bucket and hurts another
is invisible in an aggregate.

A NULL arm (a parameter set to its own default) is always prepended.  On this
engine a +-1 change to an unrelated constant moves the argmax on 13-20% of hard
positions, so an arm that does not clear the null band means nothing -- and the
null also catches a typo'd option name, which UCI silently ignores.

    .venv/bin/python tools/forensics/arm_top1.py --oracle deep-top1.json
        [--nodes 200000] [--limit 3000] NAME:K=V [NAME:K=V ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET  # noqa: E402

REPO = "/home/nick/projects/chessengine"
TUNE = REPO + "/engines/bin/archon-d47v-tune-pgo"


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def engine_defaults():
    out = subprocess.run([TUNE, "tunables"], capture_output=True, text=True,
                         cwd=REPO).stdout
    d = {}
    for line in out.splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            d[k.strip()] = v.strip()
    if not d:
        raise SystemExit("no `tunables` output -- not a --features tune build?")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--oracle", default="tools/nnue/calib/deep-top1.json")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hash", type=int, default=16)
    args = ap.parse_args()

    o = json.load(open(args.oracle))
    rows = list(o["best"].items())
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} positions, oracle @ {o['nodes']} nodes, "
          f"arms @ {args.nodes} nodes  (ratio {o['nodes']/args.nodes:.0f}x)")
    if o["nodes"] < 4 * args.nodes:
        print("  WARNING: the oracle is less than 4x the arm budget. An arm that "
              "outruns its judge measures agreement with a weaker search.")

    dflts = engine_defaults()
    specs = []
    # the null goes first and uses a real parameter set to its own value
    anchor = "RED_BASE" if "RED_BASE" in dflts else sorted(dflts)[0]
    specs.append(("NULL", {anchor: dflts[anchor]}))
    for a in args.arms:
        label, _, body = a.partition(":")
        opts = {}
        for kv in body.split(","):
            k, _, v = kv.partition("=")
            k = k.strip()
            if k not in dflts:
                raise SystemExit(f"--arm {label}: '{k}' is not a tunable")
            opts[k] = v.strip()
        specs.append((label, opts))

    e = Engine(TUNE, net=NET, hash_mb=args.hash, name="archon")
    e.assert_net_live()
    touched = sorted({k for _l, o2 in specs for k in o2})

    base = None
    hdr = f"{'arm':<20}{'top1':>8}{'delta':>8}{'+-':>6}"
    for b in ("21+", "13-20", "8-12", "<=7"):
        hdr += f"{b:>8}"
    print(hdr)
    for label, opts in specs:
        for k in touched:
            e.setopt(k, dflts[k])
        for k, v in opts.items():
            e.setopt(k, v)
        hit = collections.Counter()
        tot = collections.Counter()
        agree = 0
        for fen, (best, _cp) in rows:
            e.newgame()
            got = e.go(fen=fen, nodes=args.nodes)["best"]
            b = bucket(npieces(fen))
            tot[b] += 1
            if got == best:
                agree += 1
                hit[b] += 1
        n = len(rows)
        rate = agree / n
        if base is None:
            base = rate
            delta = "(base)"
            sem = ""
        else:
            # paired positions, so this over-states the error; still the honest floor
            sem = f"{100*math.sqrt(rate*(1-rate)/n):.2f}"
            delta = f"{100*(rate-base):+.2f}"
        line = f"{label:<20}{100*rate:>7.2f}%{delta:>8}{sem:>6}"
        for b in ("21+", "13-20", "8-12", "<=7"):
            line += f"{(100*hit[b]/tot[b] if tot[b] else 0):>7.1f}%"
        print(line, flush=True)
    e.close()
    print("\nThe NULL row is a parameter set to its own default: it MUST read "
          "+0.00.\nAnything it does not clear is noise, not a finding.")


if __name__ == "__main__":
    main()
