#!/usr/bin/env python3
"""Walk a loss to the node that discards the right move. This is METHOD STEP 2.

    python3 searchopt/tools/walk.py --index 0            # from positions/ours.json
    python3 searchopt/tools/walk.py --fen "<FEN>" --played e2e4 --oracle d2d4

Descends the disagreement automatically: at each node it asks both engines the
same question at the same budget, finds the depth at which each stops preferring
the losing move, then follows the oracle's line one ply and repeats — until it
reaches a node where the two engines AGREE. That node is the bottom; the ply
ABOVE it is where the error is manufactured.

Output is one line per ply. No row dumps: `searchdbg.py` is for reading the
decisions AT a node once this has told you which node to look at.

WHY it is built this way
------------------------
* **Both engines, same budget, per ply.** The corpus is full of positions where
  SF blunders too; those are not our defect and the walk shows it immediately
  instead of after an hour of tracing.
* **Node budgets, never fixed depth.** At fixed depth every "search more" change
  wins trivially. Everything here is `go nodes`.
* **`searchmoves` for the forced scores.** A free search reports the score of
  ITS pick; to compare two candidate moves you must force each. Free-search
  argmax alone is chaotic — an inert +-1 flips ~25% of these positions.
* **Frame.** Cross-engine numbers go through `to_sf_frame`; ours are ~2.07x SF's
  printed cp and a raw comparison is meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET, REPO, to_sf_frame  # noqa: E402

import chess


def ladder(eng, tag, fen, mv, budgets):
    """Forced score for one move across a node ladder, in SF's printed frame."""
    out = {}
    for n in budgets:
        eng.newgame()
        g = eng.go(fen=fen, nodes=n, searchmoves=[mv])
        if g["ladder"]:
            out[n] = to_sf_frame(g["ladder"][max(g["ladder"])]["score"], tag)
    return out


def free_switch(eng, fen, losing, budgets):
    """Smallest budget at which the engine stops choosing `losing` and never
    goes back. `None` = still playing it at the largest budget."""
    picks = {}
    for n in budgets:
        eng.newgame()
        picks[n] = eng.go(fen=fen, nodes=n)["best"]
    for n in budgets:
        if all(picks[m] != losing for m in budgets if m >= n):
            return n, picks
    return None, picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, help="row of --set to walk")
    ap.add_argument("--set", default="searchopt/positions/ours.json")
    ap.add_argument("--fen"); ap.add_argument("--played"); ap.add_argument("--oracle")
    ap.add_argument("--budgets", default="50000,110000,250000,500000,1000000")
    ap.add_argument("--max-ply", type=int, default=8)
    ap.add_argument("--engine", default="engines/canonical/archon")
    ap.add_argument("--sf", default="engines/bin/stockfish18")
    a = ap.parse_args()
    os.chdir(REPO)
    budgets = [int(x) for x in a.budgets.split(",")]
    INGAME = 110_000

    if a.index is not None:
        row = json.load(open(a.set))[a.index]
        fen, played, oracle = row["fen"], row["played"], row.get("best") or row.get("oracle")
        wp = row.get("wp") or row.get("wp_loss")
    else:
        fen, played, oracle, wp = a.fen, a.played, a.oracle, None
    if not (fen and played and oracle):
        ap.error("need --index, or --fen/--played/--oracle")

    ar = Engine(a.engine, net=NET, hash_mb=64, name="archon"); ar.assert_net_live()
    sf = Engine(a.sf, net=None, hash_mb=64, name="sf18")

    print(f"WALK  played {played}  oracle {oracle}" + (f"  wp_loss {wp:.2f}" if wp else ""))
    print(f"      {fen}")
    print(f"      budgets {'/'.join(str(b//1000)+'k' for b in budgets)}   in-game = 110k\n")

    # --- ply 0: whose defect is this? -------------------------------------
    a_sw, a_picks = free_switch(ar, fen, played, budgets)
    s_sw, s_picks = free_switch(sf, fen, played, budgets)
    fmt = lambda n: (f"{n//1000}k" if n else "never")
    print(f"  budget to stop playing {played}:   ARCHON {fmt(a_sw):>6}   SF18 {fmt(s_sw):>6}")
    if s_sw is not None and s_sw > INGAME:
        print(f"  >> NOT OUR DEFECT — SF also plays it at the in-game budget. Skip this position.")
    elif a_sw is not None and a_sw <= INGAME:
        print(f"  >> we already avoid it at the in-game budget; the GAME lost it "
              f"(warm context or instability), not a cold search deficit.")
    else:
        ratio = (a_sw / s_sw) if (a_sw and s_sw) else float("inf")
        print(f"  >> OURS. We need {'>=' if not a_sw else ''}{ratio:.1f}x SF's nodes here.")

    # forced comparison of the two candidates at ply 0
    print(f"\n  {'budget':>8} | {'A '+played:>10} {'A '+oracle:>10} {'A picks':>8}"
          f" | {'S '+played:>10} {'S '+oracle:>10} {'S picks':>8}")
    ap_l = ladder(ar, "archon", fen, played, budgets)
    ao_l = ladder(ar, "archon", fen, oracle, budgets)
    sp_l = ladder(sf, "sf", fen, played, budgets)
    so_l = ladder(sf, "sf", fen, oracle, budgets)
    for n in budgets:
        g = lambda d: f"{d[n]:+.0f}" if n in d else "-"
        print(f"  {n//1000:>7}k | {g(ap_l):>10} {g(ao_l):>10} {a_picks[n]:>8}"
              f" | {g(sp_l):>10} {g(so_l):>10} {s_picks[n]:>8}")

    # --- descend the oracle line until the engines agree -------------------
    print(f"\n  descending SF's line (stop when both engines agree at the in-game budget):")
    b = chess.Board(fen)
    b.push_uci(played)          # walk INTO the losing move — that subtree is where
                                # the refutation we fail to see lives
    sf.newgame()
    pv = sf.go(fen=b.fen(), nodes=1_000_000)["ladder"]
    line = pv[max(pv)]["pv"] if pv else []
    print(f"    refutation line after {played}: {' '.join(line[:8])}")
    last_div = None
    for i, u in enumerate(line[:a.max_ply]):
        try:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                break
        except Exception:
            break
        ar.newgame(); amv = ar.go(fen=b.fen(), nodes=INGAME)
        sf.newgame(); smv = sf.go(fen=b.fen(), nodes=INGAME)
        a_best, s_best = amv["best"], smv["best"]
        a_sc = to_sf_frame(amv["ladder"][max(amv["ladder"])]["score"], "archon") if amv["ladder"] else 0
        s_sc = to_sf_frame(smv["ladder"][max(smv["ladder"])]["score"], "sf") if smv["ladder"] else 0
        agree = (a_best == s_best)
        mark = "agree" if agree else "DIVERGE"
        print(f"    ply {i:<2} {b.fen().split()[0][:26]:<26} "
              f"A {a_best:>6} {a_sc:+6.0f} | S {s_best:>6} {s_sc:+6.0f}  {mark}"
              f"{'  gap '+format(a_sc-s_sc,'+.0f') if not agree else ''}")
        if not agree:
            last_div = (b.fen(), s_best, a_best, i)     # the node that manufactures it
        if agree and last_div:
            f_, sb, ab, di = last_div
            print(f"\n  >> BOTTOM RUNG at ply {i} (both engines agree). The error is"
                  f" manufactured at ply {di}, where SF plays {sb} and we play {ab}.")
            print(f"     Read the decisions there:")
            print(f"       python3 searchopt/tools/searchdbg.py \\\n"
                  f"         --fen \"{f_}\" --depth 14 --moves {sb},{ab}")
            break
        b.push(m)
    ar.close(); sf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
