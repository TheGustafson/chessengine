#!/usr/bin/env python3
"""perf study 9 -- the SMP scaling curve, driven through UCI.

WHY NOT `bench`: ARCHON's `bench` PARSES its threads argument and never reads
it (src/bench.rs -- `params.threads` has no consumer), so a bench-driven curve
silently measures Threads=1 at every point.  That is exactly what made this
study's first SMP panel worthless.  Only `setoption name Threads` reaches the
pool (src/uci.rs `Threads` arm -> ThreadPool::resize).

  python3 smpuci9.py <rounds> <depth> <outfile> <label=binary[,net]> ...

Records, per (arm, position, threads), the final `info depth <D>` line's
nodes/nps/time.  Two derived statistics:
  * NPS speedup   = nps(T) / nps(1)  -- raw aggregate throughput
  * time-to-depth = ms(1) / ms(T)    -- the EFFECTIVE speedup; a shared TT
    makes helpers search overlapping trees, so nodes grow with T and NPS
    scaling flatters the result.
"""

import re
import subprocess
import sys
import time

FENS = {
    "start": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwi": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
}


def run(binary, net, threads, fen, depth, timeout=600):
    """One `go depth` through a live UCI session; returns (nodes, nps, ms)."""
    p = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    send = lambda s: (p.stdin.write(s + "\n"), p.stdin.flush())
    send("uci")
    if net:
        send(f"setoption name EvalFile value {net}")
    send("setoption name Threads value %d" % threads)
    send("setoption name Hash value 1024")
    send("isready")
    send(f"position fen {fen}")
    send(f"go depth {depth}")

    last, deadline = None, time.time() + timeout
    while time.time() < deadline:
        line = p.stdout.readline()
        if not line:
            break
        if line.startswith(f"info depth {depth} ") and " nodes " in line:
            last = line
        if line.startswith("bestmove"):
            break
    send("quit")
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    if not last:
        return None
    g = lambda k: int(m.group(1)) if (m := re.search(rf" {k} (\d+)", last)) else 0
    return g("nodes"), g("nps"), g("time")


def main():
    rounds, depth, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    arms = []
    for spec in sys.argv[4:]:
        label, _, rest = spec.partition("=")
        binary, _, net = rest.partition(",")
        arms.append((label, binary, net))

    with open(out, "w") as f:
        f.write(f"# smpuci9 rounds={rounds} depth={depth}\n")
        for r in range(1, rounds + 1):
            for t in (1, 2, 4, 8, 16):
                for label, binary, net in arms:
                    for pos, fen in FENS.items():
                        res = run(binary, net, t, fen, depth)
                        if res is None:
                            f.write(f"round={r} arm={label} pos={pos} threads={t} FAILED\n")
                        else:
                            n, nps, ms = res
                            f.write(
                                f"round={r} arm={label} pos={pos} threads={t} "
                                f"nodes={n} nps={nps} ms={ms}\n"
                            )
                        f.flush()
    print("wrote", out)


if __name__ == "__main__":
    main()
