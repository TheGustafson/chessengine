#!/usr/bin/env python3
"""Is OUR move worse than SF's, according to a NEUTRAL third engine?

WHY THIS TOOL EXISTS. Every instrument in this folder that compared us to SF18
either (a) compared one engine's number against the other's in a different
frame, (b) compared an engine against ITS OWN deeper search -- which measures
convergence, and which a SYSTEMATICALLY biased search passes trivially -- or
(c) died to an inert null because it read one position. The residual ~68 Elo at
equal nodes, equal depth and at-par static eval is unexplained, and it is
unexplained partly because no admissible instrument has ever asked the direct
question: *at the same budget, is the move we play worse than the move SF
plays?*

THE DESIGN, and what each piece is defending against:

* **A neutral judge.** Reckless (a top-2 Rust engine) has never seen either
  engine's eval. Judging with SF would be circular -- SF's deep search agrees
  with SF's shallow search far more often than with ours, for reasons that have
  nothing to do with either move being better. We run all THREE judges anyway
  (ours, SF's, Reckless's): the two biased ones should BRACKET the neutral one,
  and if they do not, the method is broken and says so.
* **Forced `searchmoves`, one move per probe.** A fail-soft return is a BOUND
  (trap 6 in README S3); a single-root-move full-window search is a value.
* **One frame per judge, never across.** Each judge scores both moves itself,
  and only the DIFFERENCE inside that judge is ever used. No cp is ever carried
  between engines, so R2 cannot be violated by construction.
* **A discrete primary statistic.** "How often does the judge prefer SF's move"
  is a count, immune to frame, bounds, scale and chaos (README S2 rule 3). The
  cp mean is reported second, never first.
* **`--null` runs the whole pipeline with SF replaced by an inert perturbation
  of ARCHON itself.** Same tool, same judges, same positions. If the null does
  not land on 50%, the instrument cannot answer the question -- stop and change
  instrument (rule 1). This is not a footnote; run it BEFORE reading the arm.

Usage:
    python3 searchopt/tools/judge.py --n 400                  # the arm
    python3 searchopt/tools/judge.py --n 400 --null           # the control
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uciprobe as U  # noqa: E402

REPO = U.REPO
BOOK = REPO + "/engines/books/d48x-plateau.epd"


def best_move(eng, fen, nodes):
    r = eng.go(fen=fen, nodes=nodes)
    return r["best"]


def forced_score(eng, fen, move, nodes):
    """The judge's own value for `move`, in the judge's own units, mover POV.

    One root move + full window => the printed score is a value, not a bound.
    Returns None if the engine emitted no usable info line (illegal move, mate
    already on the board, ...) so the caller can drop the position rather than
    silently score it 0.
    """
    r = eng.go(fen=fen, nodes=nodes, searchmoves=[move])
    if not r["ladder"]:
        return None
    return r["ladder"][max(r["ladder"])]["score"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default=BOOK)
    ap.add_argument("--n", type=int, default=400, help="positions to screen")
    ap.add_argument("--play-nodes", type=int, default=110_000)
    ap.add_argument("--judge-nodes", type=int, default=2_000_000)
    ap.add_argument("--max-pairs", type=int, default=150,
                    help="stop after this many disagreements (cost control)")
    ap.add_argument("--null", action="store_true",
                    help="replace SF with an inert +-1 perturbation of ARCHON")
    ap.add_argument("--null-param", default="RED_TABLE_NUM")
    ap.add_argument("--null-delta", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    fens = [l.strip() for l in open(a.book) if l.strip()][: a.n]

    ours = U.Engine(REPO + "/engines/canonical/archon", hash_mb=64, name="archon")
    if a.null:
        # The control arm is ARCHON itself with one constant nudged by +-1 --
        # inert by construction, but enough to move the argmax on ~25% of
        # positions (walk.py's docstring). Needs the `tune` build; without it
        # setoption is a no-op and the two arms are byte-identical.
        other = U.Engine(REPO + "/engines/bin/archon-trace", hash_mb=64, name="archon-null")
        default = None
        for line in other.uci_lines:
            f = line.split()
            if len(f) >= 6 and f[:2] == ["option", "name"] and f[2] == a.null_param:
                default = int(f[f.index("default") + 1])
        if default is None:
            raise SystemExit(
                f"{a.null_param} is not a UCI option -- the binary is missing "
                f"--features tune, and every arm would be byte-identical "
                f"(README S3, trap 'features tune missing')")
        other.setopt(a.null_param, default + a.null_delta)
        other_label = f"null({a.null_param} {default}{a.null_delta:+d})"
    else:
        other = U.Engine(REPO + "/engines/bin/stockfish18", net=None, hash_mb=64, name="sf18")
        other_label = "sf18"

    judges = {
        "reckless": U.Engine(REPO + "/engines/bin/reckless", net=None, hash_mb=256,
                             name="reckless"),
        "sf18": U.Engine(REPO + "/engines/bin/stockfish18", net=None, hash_mb=256,
                         name="sf18-judge"),
        "archon": U.Engine(REPO + "/engines/canonical/archon", hash_mb=256,
                           name="archon-judge"),
    }

    rows = []
    screened = 0
    t0 = time.time()
    for fen in fens:
        if len(rows) >= a.max_pairs:
            break
        screened += 1
        ours.newgame()
        other.newgame()
        m_us = best_move(ours, fen, a.play_nodes)
        m_them = best_move(other, fen, a.play_nodes)
        if m_us == m_them or m_us in (None, "(none)") or m_them in (None, "(none)"):
            continue
        row = {"fen": fen, "ours": m_us, "theirs": m_them, "judge": {}}
        ok = True
        for name, j in judges.items():
            j.newgame()
            su = forced_score(j, fen, m_us, a.judge_nodes)
            st = forced_score(j, fen, m_them, a.judge_nodes)
            if su is None or st is None:
                ok = False
                break
            row["judge"][name] = {"ours": su, "theirs": st, "delta": st - su}
        if not ok:
            continue
        rows.append(row)
        d = row["judge"]["reckless"]["delta"]
        print(f"[{len(rows):>3}/{screened:>3}] {m_us} vs {m_them}  "
              f"reckless delta {d:+6d}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nscreened {screened} positions, {len(rows)} disagreements "
          f"({100*len(rows)/max(screened,1):.0f}%)")
    print(f"arm = archon vs {other_label};  play {a.play_nodes} nodes, "
          f"judge {a.judge_nodes} nodes forced\n")

    print(f"{'judge':<10}{'prefers theirs':>16}{'prefers ours':>14}{'tie':>6}"
          f"{'% theirs':>10}{'mean delta':>12}{'median':>9}")
    for name in judges:
        ds = [r["judge"][name]["delta"] for r in rows]
        if not ds:
            continue
        t = sum(1 for x in ds if x > 0)
        o = sum(1 for x in ds if x < 0)
        z = sum(1 for x in ds if x == 0)
        dec = t + o
        pct = 100 * t / dec if dec else float("nan")
        print(f"{name:<10}{t:>16}{o:>14}{z:>6}{pct:>9.1f}%"
              f"{statistics.mean(ds):>12.1f}{statistics.median(ds):>9.1f}")

    print("\nPrimary statistic is the COUNT, not the mean. A judge is only "
          "informative\nif the two biased judges bracket the neutral one.")

    if a.out:
        json.dump({"rows": rows, "screened": screened, "arm": other_label,
                   "play_nodes": a.play_nodes, "judge_nodes": a.judge_nodes},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")

    for e in list(judges.values()) + [ours, other]:
        e.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
