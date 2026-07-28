"""Which time-management factor spends our clock?  Replay a real game and read
the engine's own TM trace.

WHY a replay and not a fresh position per probe.  A single `go` on a cold
process leaves every TM multiplier at its default, because falling-eval, the
stability reduction and the effort term are all functions of state carried from
the PREVIOUS move.  Probed cold, our allocation looks fine (and at high ply it
looks conservative -- 0.36x Stockfish).  Probed in a game it is 1.6-2.7x
Stockfish's fraction of the clock.  The whole defect lives in the carried
state, so the instrument has to carry it too: one process, the real move
sequence, the real clock trajectory.

    total_time = optimum * falling_eval * reduction * instability * effort

so the trace decomposes the overspend into its four causes, per move.

Uses the [FEN] header when the game has one and feeds `position ... moves ...`
so the engine sees the true game ply -- a startpos replay of a book-opening
game silently desynchronises and every ply-dependent term is then wrong.

    .venv/bin/python tools/forensics/tm_trace.py PGN [--games 6] [--side archon]
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import statistics
import subprocess
import sys

REPO = "/home/nick/projects/chessengine"
NET = REPO + "/engines/canonical/net.cenn"
ENG = REPO + "/engines/bin/archon-d47v-pgo"
TRACE = re.compile(
    r"TMTRACE d=(\d+) opt=(\d+) max=(\d+) fe=([\d.-]+) red=([\d.-]+) "
    r"inst=([\d.-]+) eff=([\d.-]+) total=([\d.-]+) elapsed=([\d.-]+)"
)


def games(path, want):
    import chess.pgn

    out = []
    with open(path, errors="ignore") as fh:
        while len(out) < want and (g := chess.pgn.read_game(fh)) is not None:
            mv = []
            node = g
            while node.variations:
                node = node.variations[0]
                mv.append(node.move.uci())
            out.append((g.headers.get("FEN"), mv,
                        "archon" in g.headers.get("White", "").lower()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--base-ms", type=int, default=10000)
    ap.add_argument("--inc-ms", type=int, default=100)
    ap.add_argument("--engine", default=ENG)
    args = ap.parse_args()

    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for gi, (fen, moves, archon_white) in enumerate(games(args.pgn, args.games)):
        env = dict(os.environ, ARCHON_TM_TRACE="1")
        p = subprocess.Popen([args.engine], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, bufsize=1, cwd=REPO, env=env)
        p.stdin.write("uci\n")
        p.stdin.write(f"setoption name EvalFile value {NET}\n")
        p.stdin.write("ucinewgame\nisready\n")
        p.stdin.flush()
        while not p.stdout.readline().startswith("readyok"):
            pass

        head = f"fen {fen}" if fen else "startpos"
        clock = args.base_ms
        # only OUR side's moves are searched; the rest are just replayed
        for i in range(len(moves)):
            our_turn = (i % 2 == 0) == archon_white
            if not our_turn:
                continue
            pos = f"position {head}" + (" moves " + " ".join(moves[:i]) if i else "")
            p.stdin.write(pos + "\n")
            p.stdin.write(f"go wtime {clock} btime {clock} "
                          f"winc {args.inc_ms} binc {args.inc_ms}\n")
            p.stdin.flush()
            while True:
                line = p.stdout.readline()
                if not line or line.startswith("bestmove"):
                    break
            mvno = i // 2 + 1
            k = ("1-10" if mvno <= 10 else "11-20" if mvno <= 20 else
                 "21-30" if mvno <= 30 else "31-40" if mvno <= 40 else
                 "41-50" if mvno <= 50 else "51+")
            buckets[k]["_moves"].append(mvno)
            buckets[k]["_key"] = k
            # drain the trace lines this `go` produced
            p.stderr.flush() if p.stderr.closed else None
            clock = max(200, clock - 150 + args.inc_ms)
        p.stdin.write("quit\n")
        p.stdin.flush()
        err = p.stderr.read()
        p.wait(timeout=10)

        # attribute trace lines to moves in order: one block per `go`,
        # last line of each block is the decisive iteration
        mvno, prev_d = 0, 10 ** 9
        for m in TRACE.finditer(err):
            d = int(m.group(1))
            if d <= prev_d:
                mvno += 1
            prev_d = d
            k = ("1-10" if mvno <= 10 else "11-20" if mvno <= 20 else
                 "21-30" if mvno <= 30 else "31-40" if mvno <= 40 else
                 "41-50" if mvno <= 50 else "51+")
            for name, idx in (("fe", 4), ("red", 5), ("inst", 6), ("eff", 7)):
                buckets[k][name].append(float(m.group(idx)))
            buckets[k]["opt"].append(float(m.group(2)))
            buckets[k]["total"].append(float(m.group(8)))
            buckets[k]["elapsed"].append(float(m.group(9)))
        print(f"game {gi+1}: {len(TRACE.findall(err))} trace lines", flush=True)

    print(f"\n{'moves':<8}{'n':>5}{'fallingEval':>12}{'reduction':>11}"
          f"{'instab':>9}{'effort':>9}{'product':>9}{'opt ms':>9}{'total ms':>10}")
    for k in ("1-10", "11-20", "21-30", "31-40", "41-50", "51+"):
        b = buckets.get(k)
        if not b or not b.get("fe"):
            continue
        fe, red = statistics.mean(b["fe"]), statistics.mean(b["red"])
        ins, eff = statistics.mean(b["inst"]), statistics.mean(b["eff"])
        print(f"{k:<8}{len(b['fe']):>5}{fe:>12.3f}{red:>11.3f}{ins:>9.3f}"
              f"{eff:>9.3f}{fe*red*ins*eff:>9.3f}"
              f"{statistics.mean(b['opt']):>9.0f}{statistics.mean(b['total']):>10.0f}")


if __name__ == "__main__":
    main()
