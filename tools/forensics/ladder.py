"""The forensic ladder: drive every harvested error to a MECHANISM.

One traced position is an anecdote. This runs the same six-rung ladder over a
whole gift corpus so the output is a *mechanism histogram* -- which machinery
in our engine failed, how often, and therefore what to change.

The rungs, and WHY each one exists:

  R0 VERIFY      Re-ask the oracle at 10x the harvest budget.  WHY: a 100k-node
                 screen has false positives, and game 18 flipped its verdict
                 between 120k and 256k.  An unverified error is not evidence.
  R1 DECIDABLE   Was the game still winnable/drawable before the move?  WHY: a
                 200 cp "error" in a -1500 position converts nothing.  Measured
                 in WIN PROBABILITY, never centipawns (playbook 2.2).
  R2 DEPTH       At what depth does OUR engine first prefer (and hold) the
                 right move?  Compare with the depth we actually reached in the
                 game.  WHY: separates "could never see it" from "ran out of
                 clock", which are different fixes (search shape vs TM).
  R3 FORCED      Give our engine each move as the ONLY root move.  WHY: this
                 strips away root ordering and root pruning, so a wrong ranking
                 here is an EVAL/tactics defect, while a correct ranking proves
                 the eval knew and the SEARCH threw the knowledge away.
  R4 GATES       Re-run at the game depth with the eval-based prunings off.
                 WHY: if that flips the move, a gate hid the refutation -- and
                 R5 says which one.
  R5 BISECT      One gate at a time, only for rows R4 flipped.

Everything is fixed-depth or fixed-node, hence deterministic and load-immune:
safe to run at nice 19 beside a match (playbook 1.3).

Usage:
    .venv/bin/python tools/forensics/ladder.py CORPUS.json [--limit N]
        [--out OUT.json] [--verify-nodes 1000000] [--max-depth 22]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-tune-pgo"
SF = REPO + "/engines/bin/stockfish18"
OFF = 30000

# The eval-based prunings, each with the knob that disables it and its default.
# RFP is switched by its DEPTH CAP, not by inflating its margins: the margin
# form has an `improving`/`opp_worsening` path that survives a big base (that
# bug made an early game-18 probe read "RFP is not the cause" -- playbook 8).
GATES = {
    "razor": ("RAZOR_BASE", 465, OFF),
    "rfp": ("RFP_MAX_DEPTH", 17, 0),
    "pfut": ("PFUT_BASE", 40, OFF),
    "cfut": ("CFUT_BASE", 231, OFF),
    "qsfut": ("QS_FUTILITY_MARGIN", 335, OFF),
}


def wp(cp: float) -> float:
    """Win probability from centipawns. The only additive unit (playbook 2.2)."""
    return 1.0 / (1.0 + math.exp(-cp / 170.0))


def sf_score(eng: Engine, fen: str, nodes: int, move: str | None) -> int | None:
    """Side-to-move score from Stockfish at a fixed node budget."""
    r = eng.go(fen=fen, nodes=nodes, searchmoves=[move] if move else None)
    if not r["ladder"]:
        return None
    return r["ladder"][max(r["ladder"])]["score"]


def our_score(eng: Engine, fen: str, depth: int, move: str | None) -> int | None:
    r = eng.go(fen=fen, depth=depth, searchmoves=[move] if move else None)
    if not r["ladder"]:
        return None
    return r["ladder"][max(r["ladder"])]["score"]


def defaults(eng: Engine) -> None:
    for _, (knob, dflt, _off) in GATES.items():
        eng.setopt(knob, dflt)


def climb(row: dict, ours: Engine, sf: Engine, args) -> dict:
    fen, played = row["fen"], row["played"]
    out = dict(row)

    # ---- R0 VERIFY -------------------------------------------------------
    sf.newgame()
    best_deep = sf.go(fen=fen, nodes=args.verify_nodes)
    if not best_deep["ladder"]:
        out["verdict"] = "ORACLE_FAIL"
        return out
    top = best_deep["ladder"][max(best_deep["ladder"])]
    bs, bmove = top["score"], top["move"]
    sf.newgame()
    ps = sf_score(sf, fen, args.verify_nodes, played)
    if ps is None:
        out["verdict"] = "ORACLE_FAIL"
        return out
    out.update(deep_best=bmove, deep_best_cp=bs, deep_played_cp=ps,
               deep_loss=bs - ps, deep_wp_loss=round(wp(bs) - wp(ps), 3))
    if bs - ps < 100:
        out["verdict"] = "FALSE_POSITIVE"          # the screen oracle was wrong
        return out

    # ---- R1 DECIDABLE ----------------------------------------------------
    # WHY a band and not a threshold on cp: mate scores (+-20000) dominate loss
    # corpora, and a "2900 cp blunder" in an already-lost game is not a defect.
    if wp(bs) < 0.10:
        out["verdict"] = "ALREADY_LOST"
        return out
    if out["deep_wp_loss"] < 0.10:
        out["verdict"] = "MINOR_WP"
        return out

    # ---- R2 DEPTH --------------------------------------------------------
    ours.newgame()
    defaults(ours)
    lad = ours.go(fen=fen, depth=args.max_depth)["ladder"]
    d_find = None
    for d in sorted(lad):
        if lad[d]["move"] == bmove:
            if d_find is None:
                d_find = d
        else:
            d_find = None                          # must HOLD it, not touch it
    out["our_ladder"] = {d: lad[d]["move"] for d in sorted(lad)}
    out["d_find"] = d_find
    out["d_game"] = row.get("game_depth")

    # ---- R3 FORCED -------------------------------------------------------
    # Root ordering and root pruning are gone: each move is the only root move.
    fd = min(args.max_depth, 18)
    ours.newgame()
    f_played = our_score(ours, fen, fd, played)
    ours.newgame()
    f_best = our_score(ours, fen, fd, bmove)
    out["forced_played"], out["forced_best"], out["forced_depth"] = f_played, f_best, fd
    if f_played is None or f_best is None:
        out["verdict"] = "PROBE_FAIL"
        return out
    knows = f_best > f_played

    if d_find is not None and out["d_game"] and d_find <= out["d_game"]:
        out["verdict"] = "SHOULD_HAVE_FOUND"       # depth was there; something else lost it
        return out

    if not knows:
        # Our eval ranks them wrongly even with the search handed the answer.
        out["verdict"] = "EVAL"
        return out

    # ---- R4 GATES --------------------------------------------------------
    gd = max(8, min(int(out["d_game"] or 12), args.max_depth))
    allon = {k: off for _, (k, _d, off) in GATES.items()}
    ours.newgame()
    defaults(ours)
    for k, v in allon.items():
        ours.setopt(k, v)
    g = ours.go(fen=fen, depth=gd)
    ours.newgame()
    defaults(ours)
    out["gates_off_move"], out["gate_depth"] = g["best"], gd
    if g["best"] != bmove:
        out["verdict"] = "DEPTH" if d_find else "SEARCH_BLIND"
        return out

    # ---- R5 BISECT -------------------------------------------------------
    culprits = []
    for name, (knob, dflt, off) in GATES.items():
        ours.newgame()
        defaults(ours)
        ours.setopt(knob, off)
        if ours.go(fen=fen, depth=gd)["best"] == bmove:
            culprits.append(name)
    ours.newgame()
    defaults(ours)
    out["culprits"] = culprits
    out["verdict"] = "PRUNING"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--verify-nodes", type=int, default=1_000_000)
    ap.add_argument("--max-depth", type=int, default=22)
    args = ap.parse_args()

    rows = json.load(open(args.corpus))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} rows -> ladder", flush=True)

    ours = Engine(OURS, name="archon")
    ours.assert_net_live()
    sf = Engine(SF, net=None, hash_mb=256, name="sf18")

    done, hist = [], {}
    hdr = f"{'game':<6}{'played':>7}{'best':>7}{'cp':>7}{'wp':>6}{'dfind':>6}{'dgame':>6}  verdict"
    print(hdr, flush=True)
    for r in rows:
        try:
            res = climb(r, ours, sf, args)
        except Exception as e:                      # noqa: BLE001
            res = dict(r, verdict=f"ERROR:{e}")
        done.append(res)
        hist[res["verdict"]] = hist.get(res["verdict"], 0) + 1
        extra = ("+" + ",".join(res["culprits"])) if res.get("culprits") else ""
        print(f"g{str(res.get('game','?')):<5}{res['played']:>7}"
              f"{str(res.get('deep_best','?')):>7}{str(res.get('deep_loss','?')):>7}"
              f"{str(res.get('deep_wp_loss','?')):>6}{str(res.get('d_find','-')):>6}"
              f"{str(res.get('d_game','-')):>6}  {res['verdict']} {extra}", flush=True)

    ours.close()
    sf.close()
    out = args.out or (os.path.splitext(args.corpus)[0] + "_ladder.json")
    json.dump(done, open(out, "w"), indent=1)
    print("\n=== MECHANISM HISTOGRAM ===")
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"{v:>4}  {k}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
