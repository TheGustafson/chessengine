#!/usr/bin/env python3
"""Paired NPS: identical positions, identical NODE budget, alternating order.

Why this exists (2026-07-29). A match PGN's nps looked like a great instrument
and is not. Two Stockfish binaries that a paired bench shows are the same speed
within 0.7% came out **1.237x apart** on match-aggregated nps -- larger than the
effect it was being used to measure. Match nps confounds engine speed with:
  * each engine choosing its OWN time per position (time manager, not speed),
  * each engine reaching a DIFFERENT position mix (endgames are cheap per node),
  * whichever side happened to get the long thinks.

This tool removes all three. Every engine searches the SAME fens to the SAME
node count, engines alternate per position so drift and contention hit both
equally, and the statistic is total_nodes / total_wall_time.

NPS still is not strength -- a node is not a fixed unit of work across engines.
It is a throughput number and nothing more.

Usage:
    python3 tools/paired_nps.py --nodes 400000 \
        ARCHON=engines/canonical/archon:engines/canonical/net.cenn \
        SF=engines/bin/stockfish-eca43a97
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFO = re.compile(r"\bnodes\s+(\d+)")


def run_engine(path: str, net: str | None, fens: list[str], nodes: int):
    """Drive one engine over all fens; return (total_nodes, total_seconds)."""
    cmd = [path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def wait_for(token):
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(token):
                return line

    send("uci")
    wait_for("uciok")
    if net:
        send(f"setoption name EvalFile value {net}")
    send("setoption name Hash value 64")
    send("setoption name Threads value 1")
    send("isready")
    wait_for("readyok")

    total_nodes = 0
    t0 = time.perf_counter()
    for fen in fens:
        send(f"position fen {fen}")
        send(f"go nodes {nodes}")
        last = 0
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine died mid-search")
            if line.startswith("info"):
                m = INFO.search(line)
                if m:
                    last = int(m.group(1))
            elif line.startswith("bestmove"):
                # NODE COUNT BUG, found 2026-07-30 (perf study 28, verified by me).
                # ARCHON emits `info` ONLY at COMPLETED depths, so the last info
                # line reports the node count at the last finished iteration --
                # NOT the total searched when `go nodes` trips mid-depth.
                # Measured at budget 1,000,000 from startpos:
                #     ARCHON  938,175  -> under-reports 6.2%
                #     SF    1,000,317  -> accurate
                # So a reported-node ratio understated ARCHON by 6.6% and every
                # ARCHON/SF figure taken before this fix is biased AGAINST us.
                #
                # Both engines honour the budget to within ~0.03%, so the budget
                # itself is the accurate common currency. Fall back to the
                # reported count only if the engine stopped early (mate/TB), which
                # is the one case where the budget would overstate.
                total_nodes += nodes if last >= nodes * 0.90 else last
                break
    elapsed = time.perf_counter() - t0
    send("quit")
    p.wait(timeout=10)
    return total_nodes, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="+", metavar="NAME=PATH[:NET]")
    ap.add_argument("--nodes", type=int, default=400000)
    ap.add_argument("--fens", default=str(ROOT / "tools/nnue/calib/wl50.fens"))
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    fens = [ln.strip() for ln in open(a.fens) if ln.strip()][: a.count]
    specs = []
    for e in a.engines:
        name, _, rest = e.partition("=")
        path, _, net = rest.partition(":")
        specs.append((name, str(ROOT / path), str(ROOT / net) if net else None))

    print(f"{len(fens)} positions x {a.nodes:,} nodes, {a.rounds} rounds, "
          f"engines alternate each round\n")

    acc: dict[str, list[float]] = {n: [] for n, _, _ in specs}
    for r in range(a.rounds):
        # Alternate order each round so drift/contention is not always paid by
        # the same engine.
        order = specs if r % 2 == 0 else list(reversed(specs))
        for name, path, net in order:
            n, secs = run_engine(path, net, fens, a.nodes)
            nps = n / secs
            acc[name].append(nps)
            print(f"  round {r+1}  {name:<12} {nps/1e6:7.3f} Mnps  "
                  f"({n:,} nodes / {secs:.2f}s)")

    print()
    means = {k: sum(v) / len(v) for k, v in acc.items()}
    for k in sorted(means, key=lambda x: -means[x]):
        v = acc[k]
        print(f"{k:<12} mean {means[k]/1e6:.3f} Mnps   "
              f"min {min(v)/1e6:.3f}  max {max(v)/1e6:.3f}  "
              f"spread {(max(v)-min(v))/means[k]*100:.1f}%")
    names = list(means)
    if len(names) > 1:
        print("\nratios (mean):")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                print(f"  {x} / {y} = {means[x]/means[y]:.4f}x")
        print("\nCompare each ratio against the per-engine SPREAD above. A ratio "
              "smaller than the spread is not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
