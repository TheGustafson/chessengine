"""Mechanism histogram over a whole verified error corpus.

WHY this shape.  Everything else tonight that tried to attribute a single
position to a single mechanism was overturned -- the argmax of one search is
chaotic enough that a +-1 change to an unrelated constant flips it, so a
per-position verdict is not evidence.  What survives is a DISTRIBUTION over
many verified errors.

First, decidability: ALREADY_LOST / MINOR_WP drop rows where the result was not
in play, measured in WIN PROBABILITY (mate scores make centipawns useless here).

Then a 2x2, NOT a decision tree.  An earlier version short-circuited to "EVAL"
whenever our static preferred the move we played, and that conflates two
independent things -- the search frequently overcomes a wrong static given a few
more plies, and those rows are depth-fixable, not eval-fixable.  So both
questions are always asked:

                          search FINDS it by dgame+6   search NEVER finds it
  static points right         DEPTH                      SEARCH_REJECTS
  static points wrong         DEPTH_vs_EVAL              EVAL

  DEPTH            everything already agrees; we just needed the plies (or the
                   clock).  Time management and search efficiency convert these.
  DEPTH_vs_EVAL    the search had to overcome its own eval and managed it.  More
                   depth converts these too, but a better eval converts them
                   cheaper.
  SEARCH_REJECTS   the eval already knew and the search threw it away.  THE only
                   bucket that needs a search MECHANISM change -- and the one
                   that no amount of eval or clock work will touch.
  EVAL             eval wrong and search cannot rescue it.  Needs a better net.

Statics are batched (one process, one pass), so the cost is two searches per
row.  Deterministic throughout: fixed depth, safe beside anything.

    .venv/bin/python tools/forensics/classify.py CORPUS.json [--limit 0]
        [--extra-depth 6] [--workers 4]
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
OURS = REPO + "/engines/bin/archon-d47v-pgo"


def wp(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / 170.0))


def npieces(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def bucket(n: int) -> str:
    return "<=7" if n <= 7 else "8-12" if n <= 12 else "13-20" if n <= 20 else "21+"


def child_fens(rows):
    """(fen after played, fen after best) for every row, plus a validity mask."""
    import chess

    out = []
    for r in rows:
        try:
            b = chess.Board(r["fen"])
            b.push(chess.Move.from_uci(r["played"]))
            after_played = b.fen()
            b = chess.Board(r["fen"])
            b.push(chess.Move.from_uci(r["best"]))
            out.append((after_played, b.fen()))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def batch_static(fens):
    out = subprocess.run([OURS, "nnue-eval", NET], input="\n".join(fens) + "\n",
                         capture_output=True, text=True, cwd=REPO).stdout.split()
    return [int(x) for x in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra-depth", type=int, default=6)
    ap.add_argument("--min-wp", type=float, default=0.08)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = json.load(open(args.corpus))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} verified errors", flush=True)

    kids = child_fens(rows)
    # one batched pass for every child position that parsed
    flat, index = [], []
    for i, k in enumerate(kids):
        if k:
            index.append(i)
            flat += [k[0], k[1]]
    stat = batch_static(flat) if flat else []
    static = {}
    for j, i in enumerate(index):
        # child is the OPPONENT to move, so negate to get our point of view
        static[i] = (-stat[2 * j], -stat[2 * j + 1])

    eng = Engine(OURS, net=NET, hash_mb=64, name="archon")
    eng.assert_net_live()

    hist = collections.Counter()
    by_phase = collections.defaultdict(collections.Counter)
    out = []
    print(f"{'game':<6}{'played':>7}{'best':>7}{'wp':>6}{'pieces':>7}"
          f"{'stat p/b':>12}{'dfind':>6}{'dgame':>6}  verdict")
    for i, r in enumerate(rows):
        ph = bucket(npieces(r["fen"]))
        v = None
        if wp(r["best_cp"]) < 0.10:
            v = "ALREADY_LOST"
        elif r["wp_loss"] < args.min_wp:
            v = "MINOR_WP"
        sp = sb = None
        dfind = None
        dgame = int(r.get("game_depth") or 12)
        if v is None:
            if i not in static:
                v = "PARSE_FAIL"
            else:
                sp, sb = static[i]
                # ALWAYS run the depth test.  Short-circuiting to EVAL whenever
                # the static prefers the played move conflates two independent
                # things: the search often overcomes a wrong static given plies,
                # and those rows are depth-fixable, not eval-fixable.  The two
                # dimensions are reported as a 2x2 instead.
                dmax = min(30, dgame + args.extra_depth)
                eng.newgame()
                lad = eng.go(fen=r["fen"], depth=dmax)["ladder"]
                first = None
                for d in sorted(lad):
                    if lad[d]["move"] == r["best"]:
                        if first is None:
                            first = d
                    else:
                        first = None
                dfind = first
                static_right = sb > sp
                if first is not None:
                    v = "DEPTH" if static_right else "DEPTH_vs_EVAL"
                else:
                    v = "SEARCH_REJECTS" if static_right else "EVAL"
        hist[v] += 1
        by_phase[ph][v] += 1
        out.append(dict(r, verdict=v, static_played=sp, static_best=sb,
                        d_find=dfind, phase=ph))
        if v in ("EVAL", "DEPTH", "DEPTH_vs_EVAL", "SEARCH_REJECTS"):
            print(f"g{str(r['game']):<5}{r['played']:>7}{r['best']:>7}"
                  f"{r['wp_loss']:>6.2f}{ph:>7}"
                  f"{f'{sp}/{sb}' if sp is not None else '-':>12}"
                  f"{str(dfind or '-'):>6}{dgame:>6}  {v}", flush=True)
    eng.close()

    print("\n=== MECHANISM HISTOGRAM ===")
    tot = sum(hist.values())
    for k, n in hist.most_common():
        print(f"{n:>5}  {100*n/tot:5.1f}%  {k}")
    REAL = ("DEPTH", "DEPTH_vs_EVAL", "SEARCH_REJECTS", "EVAL")
    real = {k: v for k, v in hist.items() if k in REAL}
    rt = sum(real.values())
    if rt:
        print(f"\n=== of the {rt} DECIDABLE errors ===")
        for k, n in sorted(real.items(), key=lambda kv: -kv[1]):
            print(f"{n:>5}  {100*n/rt:5.1f}%  {k}")
        print(f"\n{'phase':<8}" + "".join(f"{k:>16}" for k in REAL))
        for ph in ("21+", "13-20", "8-12", "<=7"):
            c = by_phase[ph]
            if not any(c[k] for k in REAL):
                continue
            print(f"{ph:<8}" + "".join(f"{c[k]:>16}" for k in REAL))

    dst = args.out or (os.path.splitext(args.corpus)[0] + "_classified.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
