#!/usr/bin/env python3
"""tools/perf70/cscreen70.py — screen ordering arms on the CENSUS statistics.

The node-count metric is chaotic: an inert +/-1 tie-break jitter moves a
50-position d13 total by up to +18%, and even the 1000-position paired
geometric mean has a +/-1.5% null spread. The census statistics are averages
over 10^5-10^6 EVENTS inside one deterministic run, so their sampling noise is
two orders of magnitude smaller:

  fhf(NonPV)      ~ 220k cutoffs  -> binomial sd ~0.08 pp
  LMR research    ~ 760k scouts   -> binomial sd ~0.02 pp

They measure ORDERING QUALITY directly rather than its downstream node
consequence, so they are the screening instrument; nodes and then games
arbitrate. Same discipline as everywhere else: the null panel runs first.

  tools/perf70/cscreen70.py 'lva=8' 'evcap=1' ...
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
NULLS = int(os.environ.get("NULLS", "4"))


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
        m = re.match(r"C70 tot kind=(\w+) fh=(\d+) fh1=(\d+) fhf=([\d.]+) mean_idx=([\d.]+)", line)
        if m:
            k = m.group(1)
            out[f"fh_{k}"] = int(m.group(2))
            out[f"fhf_{k}"] = float(m.group(4))
            out[f"midx_{k}"] = float(m.group(5))
        m = re.match(
            r"C70 lmr kind=(\w+) try=(\d+) research=(\d+) rate=([\d.]+)\s+"
            r"scout_nodes=(\d+) research_nodes=(\d+)",
            line,
        )
        if m:
            out[f"lmrtry_{m.group(1)}"] = int(m.group(2))
            out[f"lmrrate_{m.group(1)}"] = float(m.group(4))
            out[f"lmrnodes_{m.group(1)}"] = int(m.group(6))
    if "nodes" not in out:
        raise SystemExit(f"cscreen70: no output\n{p.stderr[-2000:]}")
    return out


def fmt(spec, r, b=None):
    def d(key, scale=100.0, w=6):
        v = r.get(key, 0.0)
        if b is None:
            return f"{scale * v:{w}.3f}"
        return f"{scale * (v - b[key]):+{w}.3f}"

    nodes = r["nodes"]
    npct = "" if b is None else f" ({100.0 * (nodes / b['nodes'] - 1):+6.3f}%)"
    return (
        f"{spec:<30} fhf_NONPV={d('fhf_NONPV')} fhf_CHK={d('fhf_NONPVCHK')} "
        f"fhf_PV={d('fhf_PV')} fhf_QS={d('fhf_QS')} "
        f"lmrres={d('lmrrate_NONPV')} nodes={nodes}{npct}"
    )


def main():
    base = run("")
    print(f"# {FENS} @ d{DEPTH}   (first row absolute %, the rest deltas in pp)")
    print(fmt("(baseline)", base))
    for seed in range(1, NULLS + 1):
        print(fmt(f"NULL jitter=1 seed={seed}", run(f"jitter=1,jseed={seed}"), base))
    for spec in sys.argv[1:]:
        print(fmt(spec, run(spec), base))


if __name__ == "__main__":
    main()
