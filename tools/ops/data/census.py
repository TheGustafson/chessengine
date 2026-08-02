#!/usr/bin/env python3
"""tools/ops/data/census.py — what IS this corpus a sample of?

PURPOSE
  Answers the diversity half of the brief: piece-count distribution, game
  phase, label distribution, decisiveness, and how much of the corpus carries
  a MEANINGFUL WDL column rather than a nominal one.

  A corpus can pass every D46 check and still be a bad sample. The receipt is
  the previous oracle position set, which was 100% >=21-piece positions and
  therefore blind to 82% of our damage; nothing in the contract catches that,
  because it is a question about the distribution, not about any single row.

"WDL IS MEANINGFUL" — made concrete instead of asserted
  A result column is informative about a position only if the game was still
  live there. Three ways it can be uninformative, all measured here:

    * DEAD-DRAWN     the oracle calls the position level AND the game drew.
                     The label and the result agree, but neither discriminates.
    * ALREADY-DECIDED the oracle already calls it >= 600 cp. The outcome was
                     nearly determined; the WDL bit adds little the eval did
                     not already say, and these positions dominate the tail of
                     long won games.
    * CONTRADICTED   the oracle calls it decisive and the game went the other
                     way (the study_label_quality statistic, repeated here so a
                     census alone is enough to judge a corpus).

  The live fraction -- neither dead-drawn nor already-decided -- is the share of
  rows where the result column is actually doing work.

USAGE
  .venv/bin/python tools/ops/data/census.py labelled.tsv [more.tsv ...] \
      [--json out.json] [--label-frame sf-printed-cp]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

import chess

PHASE = [(0, 12, "endgame  <=12"), (13, 20, "middlegame 13-20"),
         (21, 32, "opening  >=21")]


def bucket(pc):
    for lo, hi, name in PHASE:
        if lo <= pc <= hi:
            return name
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--decided-cp", type=int, default=600)
    ap.add_argument("--level-cp", type=int, default=50)
    ap.add_argument("--threshold", type=int, default=150)
    args = ap.parse_args()

    n = 0
    pieces = collections.Counter()
    phases = collections.Counter()
    plies = collections.Counter()
    results = collections.Counter()
    terms = collections.Counter()
    cpband = collections.Counter()
    games = set()
    fens = set()
    dead = decided = contra = decisive = live = 0
    stm = collections.Counter()

    for path in args.tsv:
        with open(path) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ix = {c: i for i, c in enumerate(hdr)}
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < len(hdr):
                    continue
                if "drop_reason" in ix and f[ix["drop_reason"]].strip():
                    continue
                cp = f[ix["label_cp"]]
                if cp == "":
                    continue
                n += 1
                fen = f[ix["fen"]]
                cp = int(cp)
                res = f[ix["result"]]
                fens.add(" ".join(fen.split()[:4]))
                if "game_id" in ix:
                    games.add(f[ix["game_id"]])
                b = chess.Board(fen)
                pc = len(b.piece_map())
                pieces[pc] += 1
                phases[bucket(pc)] += 1
                plies[min(int(f[ix["ply"]] or 0) // 20 * 20, 120)] += 1
                results[res] += 1
                if "termination" in ix:
                    terms[f[ix["termination"]]] += 1
                stm["w" if b.turn == chess.WHITE else "b"] += 1
                a = abs(cp)
                cpband[("0-50" if a < 50 else "50-150" if a < 150 else
                        "150-400" if a < 400 else "400-900" if a < 900 else
                        "900+")] += 1
                rf = {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0}.get(res)
                is_dead = a <= args.level_cp and rf == 0.5
                is_dec = a >= args.decided_cp
                if is_dead:
                    dead += 1
                if is_dec:
                    decided += 1
                if not is_dead and not is_dec:
                    live += 1
                if a >= args.threshold and rf is not None:
                    decisive += 1
                    if (cp >= args.threshold and rf == 0.0) or \
                       (cp <= -args.threshold and rf == 1.0):
                        contra += 1

    if n == 0:
        sys.stderr.write("no rows\n")
        return 2
    pct = lambda k: round(100.0 * k / n, 2)
    rep = dict(
        rows=n, unique_positions=len(fens), source_games=len(games),
        dup_pct=round(100.0 * (n - len(fens)) / n, 2),
        rows_per_game=round(n / max(len(games), 1), 2),
        phase={k: [v, pct(v)] for k, v in phases.most_common()},
        piece_count_mean=round(sum(k * v for k, v in pieces.items()) / n, 2),
        ply_buckets={f"{k}-{k+19}": [v, pct(v)] for k, v in sorted(plies.items())},
        results={k: [v, pct(v)] for k, v in results.most_common()},
        terminations={k: [v, pct(v)] for k, v in terms.most_common()},
        abs_label_bands={k: [v, pct(v)] for k, v in cpband.most_common()},
        stm_balance={k: [v, pct(v)] for k, v in stm.most_common()},
        wdl_meaning=dict(
            dead_drawn=[dead, pct(dead)],
            already_decided=[decided, pct(decided)],
            live=[live, pct(live)],
            decisive_labels=decisive,
            contradicted=contra,
            contradiction_pct=round(100.0 * contra / max(decisive, 1), 2)),
    )
    if args.json:
        open(args.json, "w").write(json.dumps(rep, indent=2))

    print(f"\n=== corpus census: {n} rows, {len(fens)} unique "
          f"({rep['dup_pct']}% dup), {len(games)} games ===")
    print(f"\nphase (mean {rep['piece_count_mean']} pieces):")
    for k, (v, p) in rep["phase"].items():
        print(f"  {k:<20} {v:>8}  {p:>5.1f}%")
    print("\ngame ply:")
    for k, (v, p) in rep["ply_buckets"].items():
        print(f"  {k:<20} {v:>8}  {p:>5.1f}%")
    print("\n|label| band:")
    for k in ("0-50", "50-150", "150-400", "400-900", "900+"):
        if k in rep["abs_label_bands"]:
            v, p = rep["abs_label_bands"][k]
            print(f"  {k:<20} {v:>8}  {p:>5.1f}%")
    print("\nresult (White-relative; UHO reference is ~W42/D47/L11):")
    for k, (v, p) in rep["results"].items():
        print(f"  {k:<20} {v:>8}  {p:>5.1f}%")
    if terms:
        print("\ntermination:")
        for k, (v, p) in rep["terminations"].items():
            print(f"  {k:<20} {v:>8}  {p:>5.1f}%")
    w = rep["wdl_meaning"]
    print("\nis the WDL column doing work?")
    print(f"  dead-drawn (|cp|<={args.level_cp} and drawn)   "
          f"{w['dead_drawn'][0]:>8}  {w['dead_drawn'][1]:>5.1f}%")
    print(f"  already-decided (|cp|>={args.decided_cp})       "
          f"{w['already_decided'][0]:>8}  {w['already_decided'][1]:>5.1f}%")
    print(f"  LIVE (neither)                        "
          f"{w['live'][0]:>8}  {w['live'][1]:>5.1f}%")
    print(f"  contradicted / decisive               "
          f"{w['contradicted']}/{w['decisive_labels']}  "
          f"{w['contradiction_pct']:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
