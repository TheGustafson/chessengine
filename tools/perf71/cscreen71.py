#!/usr/bin/env python3
"""tools/perf71/cscreen71.py — census screen for perf study 71 (the TT-less
move-ordering deficit).

Same instrument and same discipline as `tools/perf70/cscreen70.py` — the null
panel runs FIRST and every number is a delta against the same baseline run —
but it reports the statistics study 71 is aimed at, which study 70's screen
does not print:

  fhf                 first-move beta-cutoff rate, per node kind (pp)
  ttno%               share of that kind's CUTOFFS that arrived TT-less
  fhf_no / fhf_yes    the split those cutoffs divide into
  mless%              share of TT-less NODE ENTRIES that had an entry but no
                      move (the population a store-policy change can move)

A store-policy arm moves `ttno%` by construction; the question the census is
being asked is whether the OVERALL `fhf` moves with it, because that is the
only one of these that a pruning side effect cannot manufacture.

  tools/perf71/cscreen71.py 'flmove=2' 'flmove=1' ...
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

BIN = os.environ.get("BIN", "target/both/release/chessengine")
NET = os.environ.get("NET", "engines/canonical/net.cenn")
FENS = os.environ.get("FENS", "tools/perf70/uho1000.fens")
DEPTH = os.environ.get("DEPTH", "10")
NULLS = int(os.environ.get("NULLS", "3"))
BASE = os.environ.get("BASE", "")
KIND = os.environ.get("KIND", "NONPV")


def run(spec):
    env = dict(os.environ, ARCHON_A70=spec)
    p = subprocess.run(
        [BIN, "bench", "16", "1", DEPTH, FENS, "depth", NET],
        capture_output=True,
        text=True,
        env=env,
    )
    out = {}
    for line in p.stderr.splitlines():
        if line.startswith("Nodes searched  :"):
            out["nodes"] = int(line.split(":")[1])
        m = re.match(r"C70 tot kind=(\w+) fh=(\d+) fh1=(\d+) fhf=([\d.]+)", line)
        if m:
            k = m.group(1)
            out[f"fh_{k}"] = int(m.group(2))
            out[f"fhf_{k}"] = float(m.group(4))
        m = re.match(
            r"C70 ttsplit kind=(\w+) no_tt_fh=(\d+) no_tt_fhf=([\d.]+)\s+"
            r"has_tt_fh=(\d+) has_tt_fhf=([\d.]+)",
            line,
        )
        if m:
            k = m.group(1)
            no, yes = int(m.group(2)), int(m.group(4))
            out[f"ttno_{k}"] = no / max(no + yes, 1)
            out[f"fhfno_{k}"] = float(m.group(3))
            out[f"fhfyes_{k}"] = float(m.group(5))
        m = re.match(r"C70 ttwhy kind=(\w+) n=(\d+) miss=(\d+)\S* moveless=(\d+)", line)
        if m:
            k = m.group(1)
            n = int(m.group(2))
            out[f"ttless_n_{k}"] = n
            out[f"mless_{k}"] = int(m.group(4)) / max(n, 1)
        m = re.match(r"C70 lmr kind=(\w+) try=(\d+) research=(\d+) rate=([\d.]+)", line)
        if m:
            out[f"lmrrate_{m.group(1)}"] = float(m.group(4))
    if "nodes" not in out:
        raise SystemExit(f"cscreen71: no output\n{p.stderr[-2000:]}")
    return out


def fmt(spec, r, b=None):
    def d(key, w=7):
        v = r.get(key, 0.0)
        if b is None:
            return f"{100.0 * v:{w}.3f}"
        return f"{100.0 * (v - b.get(key, 0.0)):+{w}.3f}"

    nodes = r["nodes"]
    npct = "" if b is None else f" ({100.0 * (nodes / b['nodes'] - 1):+6.3f}%)"
    k = KIND
    return (
        f"{spec:<28} fhf={d(f'fhf_{k}')} fhf_QS={d('fhf_QS')} "
        f"ttno%={d(f'ttno_{k}')} fhf_no={d(f'fhfno_{k}')} fhf_yes={d(f'fhfyes_{k}')} "
        f"mless%={d(f'mless_{k}')} lmrres={d('lmrrate_NONPV')} nodes={nodes}{npct}"
    )


def main():
    base = run(BASE)
    print(f"# {FENS} @ d{DEPTH}  kind={KIND}  BASE={BASE!r}")
    print("# row 1 absolute %, the rest deltas in percentage points")
    print(fmt("(baseline)", base))
    for seed in range(1, NULLS + 1):
        spec = f"{BASE},jitter=1,jseed={seed}" if BASE else f"jitter=1,jseed={seed}"
        print(fmt(f"NULL jitter=1 seed={seed}", run(spec), base))
    for spec in sys.argv[1:]:
        full = f"{BASE},{spec}" if BASE and spec else (spec or BASE)
        print(fmt(spec, run(full), base))


if __name__ == "__main__":
    main()
