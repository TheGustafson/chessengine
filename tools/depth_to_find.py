#!/usr/bin/env python3
"""How deep must each engine search before it finds the right move — and how
deep did it actually get in the game?

THE QUESTION THIS SETTLES. When we lose a game to a specific move, there are
two completely different explanations and they have opposite fixes:

  SEARCH SHAPE   our search needs far more depth than Stockfish's to surface
                 the same move. Ordering, pruning or extensions are wrong, and
                 no amount of extra time fixes it.
  DEPTH REACHED  both engines need about the same depth, and we simply did not
                 get there in the time we allocated. Speed and time management
                 are the levers; the search itself is fine.

Aggregates cannot separate these. cp loss, loss taxonomies and depth-by-phase
all average over the moment that decided the game. This measures the moment
directly: for each losing position it runs BOTH engines on the identical FEN,
walks their iteration ladders, and records the first depth from which each
engine's choice settles on the oracle's move and never leaves it. Then it pulls
what actually happened in the game — depth reached, seconds spent, clock
remaining — out of the PGN.

The headline is the pair of differences:

    ours_needed - theirs_needed     > 0 means our SEARCH is the problem
    ours_needed - ours_reached      > 0 means DEPTH IN TIME is the problem

Worked example that motivated the tool (game 22, move 32): ARCHON settles on
the right move at depth 21, SF18 at depth 19 — two plies apart, and from there
their principal variations are identical, so nothing was pruned away that
Stockfish saw. But ARCHON reached depth 13 in the game, having spent 0.078 s
with 1.12 s on its clock while SF18 had 2.59 s. The search was fine; the depth
was not there.

    python3 tools/depth_to_find.py engines/runs/rematch-v1/time/games.pgn \
        --oracle tools/nnue/calib/oracle-loss.json --fens data/losscorpus/lossmine-0727.fens
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
try:
    import chess
    import chess.pgn
except ModuleNotFoundError:  # pragma: no cover
    _v = REPO / ".venv" / "bin" / "python3"
    if _v.exists() and Path(sys.executable).resolve() != _v.resolve():
        os.execv(str(_v), [str(_v), *sys.argv])
    raise

ANNOT = re.compile(r"([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s")
INFO = re.compile(r"^info depth (\d+) .*? pv (\S+)", re.M)
NET = "engines/canonical/net.cenn"


def ladder(binary: str, fen: str, nodes: int, net: str | None) -> dict[int, str]:
    """depth -> best move, from one search's info lines.

    One search yields the whole ladder, which is why this is affordable at all;
    re-searching per depth would multiply the cost by the depth count. stdin is
    held open past the search because our engine treats EOF as a stop and would
    otherwise emit a bestmove with no info lines.
    """
    script = "uci\n"
    if net:
        script += f"setoption name EvalFile value {net}\n"
    script += f"isready\nposition fen {fen}\ngo nodes {nodes}\n"
    out = subprocess.run(
        ["bash", "-c", f'{{ printf %s "$S"; sleep 1; }} | timeout 180 {binary}'],
        capture_output=True, text=True, env={**os.environ, "S": script}).stdout
    best: dict[int, str] = {}
    for d, mv in INFO.findall(out):
        best.setdefault(int(d), mv)          # first pv line at each depth
        best[int(d)] = mv                    # ...but last wins (re-searches)
    return best


def settles_at(lad: dict[int, str], target: str) -> int | None:
    """First depth from which the ladder is `target` and never changes again."""
    if not lad:
        return None
    ds = sorted(lad)
    if lad[ds[-1]] != target:
        return None
    out = ds[-1]
    for d in reversed(ds):
        if lad[d] != target:
            break
        out = d
    return out


def game_facts(pgn: str, us: str, base: float, inc: float) -> dict[str, dict]:
    """fen -> what happened on that move in the game."""
    facts = {}
    with open(pgn, errors="ignore") as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            us_w = us.lower() in g.headers.get("White", "").lower()
            clock = base
            node = g
            while node.variations:
                node = node.variations[0]
                m = ANNOT.search(node.comment or "")
                if not m:
                    continue
                b = node.parent.board()
                if (b.turn == chess.WHITE) != us_w:
                    continue
                clock = clock - float(m.group(3)) + inc
                facts[b.fen()] = {"depth": int(m.group(2)),
                                  "time": float(m.group(3)),
                                  "clock": round(clock, 2),
                                  "played": node.move.uci(),
                                  "game": g.headers.get("Round", "?")}
    return facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--oracle", default="", help="optional; target defaults to "
                    "SF18's own converged move, which needs no labelled corpus")
    ap.add_argument("--fens", default="", help="restrict to these positions")
    ap.add_argument("--engine", default="engines/bin/archon-d47v-pgo")
    ap.add_argument("--net", default=NET)
    ap.add_argument("--opponent", default="engines/bin/stockfish18")
    ap.add_argument("--us", default="archon")
    ap.add_argument("--nodes", type=int, default=6_000_000)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.chdir(REPO)

    base, _, inc = args.tc.partition("+")
    facts = game_facts(args.pgn, args.us, float(base), float(inc or 0))
    deep = {}
    if args.oracle:
        tiers = json.load(open(args.oracle))["tiers"]
        deep = tiers[sorted(tiers, key=int)[-1]]
    if args.fens:
        want = [l.strip() for l in open(args.fens) if l.strip()]
        fens = [f for f in want if f in facts][: args.limit]
    else:
        fens = [f for f in deep if f in facts][: args.limit]
    print(f"{len(fens)} positions with both an oracle verdict and a game record"
          f"  ({args.nodes:,} nodes/side)\n")
    print(f"{'game':>5}{'played':>7}{'best':>7}{'ourD':>6}{'sfD':>5}"
          f"{'gameD':>7}{'sec':>7}{'clock':>7}   verdict")

    rows = []
    for i, fen in enumerate(fens, 1):
        g = facts[fen]
        # Target = the oracle's move when we have one, otherwise SF18's own
        # converged choice. The latter is mildly biased toward Stockfish, but
        # the SHAPE question -- how many plies each engine needs to reach the
        # SAME conclusion -- is well posed either way, and it lets this run on
        # any position from any PGN with no labelled corpus.
        sf_lad = ladder(args.opponent, fen, args.nodes, None)
        if fen in deep:
            target = max(deep[fen], key=deep[fen].get)
        elif sf_lad:
            target = sf_lad[max(sf_lad)]
        else:
            continue
        ours = settles_at(ladder(args.engine, fen, args.nodes, args.net), target)
        theirs = settles_at(sf_lad, target)
        if ours is None and theirs is None:
            verdict = "neither converges"
        elif ours is None:
            verdict = "WE NEVER FIND IT"
        elif theirs is None:
            verdict = "we find it, they do not"
        elif ours - theirs >= 3:
            verdict = f"search shape (+{ours - theirs} plies)"
        elif ours > g["depth"]:
            verdict = f"depth short by {ours - g['depth']}"
        else:
            verdict = "found it in game"
        rows.append({"fen": fen, "game": g["game"], "played": g["played"],
                     "best": target, "ours": ours, "theirs": theirs,
                     "game_depth": g["depth"], "time": g["time"],
                     "clock": g["clock"], "verdict": verdict})
        print(f"{g['game']:>5}{g['played']:>7}{target:>7}"
              f"{(ours if ours else '--'):>6}{(theirs if theirs else '--'):>5}"
              f"{g['depth']:>7}{g['time']:>7.3f}{g['clock']:>7.2f}   {verdict}",
              flush=True)

    both = [r for r in rows if r["ours"] and r["theirs"]]
    if both:
        shape = [r["ours"] - r["theirs"] for r in both]
        short = [r["ours"] - r["game_depth"] for r in both]
        print(f"\n{'':<28}{'median':>9}{'mean':>8}{'>0':>7}")
        print(f"{'ours - theirs (SEARCH SHAPE)':<28}{st.median(shape):>9.1f}"
              f"{st.fmean(shape):>8.1f}{sum(1 for x in shape if x > 0):>4}/{len(shape)}")
        print(f"{'ours - reached (DEPTH IN TIME)':<28}{st.median(short):>9.1f}"
              f"{st.fmean(short):>8.1f}{sum(1 for x in short if x > 0):>4}/{len(short)}")
        print("\nIf 'ours - theirs' is near zero while 'ours - reached' is large and")
        print("positive, our search finds what Stockfish finds and the loss is depth")
        print("we did not buy. If 'ours - theirs' is large, the search itself is the")
        print("problem and more time will not fix it.")
    never = sum(1 for r in rows if r["verdict"] == "WE NEVER FIND IT")
    if never:
        print(f"\n{never} positions where we never converge on the oracle move but "
              f"Stockfish does —\nthose are genuine blind spots and deserve "
              f"individual inspection.")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
