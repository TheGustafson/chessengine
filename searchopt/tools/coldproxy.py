#!/usr/bin/env python3
"""A UCI pass-through that makes an engine play every move from COLD state.

WHY. Every instrument in this folder is a cold per-position probe: `ucinewgame`,
one FEN, one `go`. Every GAME is warm -- the TT, the butterfly/continuation/
capture/pawn histories, the five correction-history tables and the aspiration
seed all carry across the 80 moves of a game. So a defect that lives ONLY in
how state carries between moves is invisible to every measurement on record,
and would show up exactly as the ledger describes the residual: at par on every
cold statistic, ~5 cp/move of distributed drift in games, budget-independent.

This proxy turns that blind spot into a game-arbitrated number. Run

    engine-cold  vs  engine-warm      (self-play, equal nodes)

for BOTH engines. The quantity of interest is not either match on its own but
the RATIO of the two gains: how much does each engine get out of its carried
state? Each side of each match is the same binary with the same eval, so eval
cancels exactly and there is no frame to convert -- the comparison is between
two within-engine deltas.

Mechanics: hold each `position` line, and on the following `go` emit
`ucinewgame` + `isready` (swallowing exactly that one `readyok`), then the held
`position`, then the `go`. Ordering matters -- `ucinewgame` before `position`,
because a reset after the position could discard it.

    engines/bin/fastchess -engine cmd="searchopt/tools/coldproxy.py <engine>" ...
"""

from __future__ import annotations

import subprocess
import sys
import threading

# How many injected `readyok` lines are still owed to us rather than to the GUI.
# Guarded because the pump thread decrements it while the main thread increments.
_owed = 0
_lock = threading.Lock()


def pump(proc):
    """engine stdout -> our stdout, swallowing our own injected readyok lines."""
    global _owed
    for line in proc.stdout:
        if line.startswith("readyok"):
            with _lock:
                if _owed > 0:
                    _owed -= 1
                    continue
        sys.stdout.write(line)
        sys.stdout.flush()


def main() -> int:
    global _owed
    argv = sys.argv[1:]
    # --warm passes everything through untouched. Both arms of the match then
    # run through the SAME wrapper, so the only difference between them is the
    # injected reset -- not the process tree, the pipe buffering or the extra
    # `position` round-trip. Controls of exactly this kind are what killed five
    # instruments in README S2; cheaper to build one in than to argue later.
    warm = "--warm" in argv
    # --tt-only clears ONLY the transposition table (`setoption name Clear
    # Hash`, which both engines expose) and leaves the histories, the five
    # correction tables and the aspiration seed alone. Running warm / tt-only /
    # fully-cold as three arms decomposes the carried-state gain into its TT
    # half and its everything-else half, per engine, with no code change to
    # either engine.
    tt_only = "--tt-only" in argv
    argv = [x for x in argv if x not in ("--warm", "--tt-only")]
    if not argv:
        print("usage: coldproxy.py [--warm] <engine> [args...]", file=sys.stderr)
        return 2
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    threading.Thread(target=pump, args=(proc,), daemon=True).start()

    def send(s):
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    held = None
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.startswith("position"):
            held = line          # hold it: the reset has to come first
            continue
        if line.startswith("go"):
            if tt_only:
                send("setoption name Clear Hash")
            elif not warm:
                with _lock:
                    _owed += 1
                send("ucinewgame")
                send("isready")
            if held is not None:
                send(held)
            send(line)
            continue
        send(line)
        if line.startswith("quit"):
            break
    proc.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
