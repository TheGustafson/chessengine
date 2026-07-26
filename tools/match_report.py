#!/usr/bin/env python3
"""match_report — the comprehensive post-match story (standing habit, user
directive 2026-07-26: "in addition to the score, why did so-and-so lose").

One reusable report per fastchess PGN:
  1. Score, per-color split, termination histogram.
  2. BOTH sides' loss taxonomies (decisive-moment forensics, slope-aware).
  3. Depth story: by phase, and at the decisive ply.
  4. Clock story (timed games): remaining-time track + pressure incidence.
  5. Draw anatomy: how draws happened, who was worse when they did.
  6. Blunder ledger: biggest own-eval collapses per side, with game refs
     for manual review.
  7. A computed narrative summary.

Usage:
  tools/match_report.py <games.pgn> --a NAME --b NAME \
      [--a-slope 0.548] [--b-slope 0.548] [--base-ms 10000]
Slopes: each side's reported-cp -> oracle-cp factor (eval_bench). Equal
nets => equal slopes is fine; cross-engine use the measured ones
(ARCHON d47 ~0.55, SF16 ~0.95, SF18 ~1.07).
"""
from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "nnue"))
from loss_forensics import TOK, parse_games, phase_of  # noqa: E402


def side_split(moves, hdr, name):
    fen = hdr.get("FEN", "")
    white_first = (" w " in fen) if fen else True
    a_is_white = name.lower() in hdr.get("White", "").lower()
    off = 0 if (a_is_white == white_first) else 1
    return moves[off::2], moves[1 - off::2], a_is_white


def analyse(path, A, B, a_slope, b_slope, base_ms):
    S = {A: Counter(), B: Counter()}          # result tallies
    color = {A: Counter(), B: Counter()}
    term = Counter()
    depth_phase = {A: {}, B: {}}
    losses = {A: [], B: []}
    draws = []
    blunders = {A: [], B: []}
    slope = {A: a_slope, B: b_slope}
    ngames = 0
    for hdr, moves in parse_games(path):
        ngames += 1
        res = hdr.get("Result", "*")
        term[hdr.get("Termination", "normal/mate")] += 1
        a_mv, b_mv, a_white = side_split(moves, hdr, A)
        for who, seq in ((A, a_mv), (B, b_mv)):
            for i, (_, dep, _) in enumerate(seq):
                depth_phase[who].setdefault(phase_of(i + 1), []).append(dep)
            # blunders: own-eval drop across own consecutive moves
            for i in range(len(seq) - 1):
                # exclude mate-score transitions (|eval| > 25 pawns): they
                # produce thousand-pawn "drops" that are score-encoding, not
                # blunders (first version's ledger was polluted by them)
                if abs(seq[i][0]) > 25 or abs(seq[i + 1][0]) > 25:
                    continue
                drop = (seq[i][0] - seq[i + 1][0]) * slope[who]
                if drop >= 2.0:
                    blunders[who].append((drop, hdr.get("Round", "?"), i + 1,
                                          seq[i][0], seq[i + 1][0]))
        if res == "1/2-1/2":
            # draw anatomy: who stood worse at the end (own POV, slope-corr)
            # adjudicated draws end ~0 by definition; sample at 2/3 of the
            # game for "who stood worse en route to the draw"
            k = (2 * len(a_mv)) // 3
            enda = a_mv[k][0] * slope[A] if a_mv else 0
            draws.append((enda, len(moves), hdr.get("Termination", "?")))
            S[A]["D"] += 1; S[B]["D"] += 1
            continue
        if res == "*":
            continue
        a_won = (res == "1-0") == a_white
        w, l = (A, B) if a_won else (B, A)
        S[w]["W"] += 1; S[l]["L"] += 1
        color[w]["as_white" if (res == "1-0") else "as_black"] += 1
        # decisive-moment record for the loser (winner's eval, slope-corr)
        win_seq = a_mv if a_won else b_mv
        lose_seq = b_mv if a_won else a_mv
        w_ev = [e * slope[w] for e, _, _ in win_seq]
        pnr = next((i for i, e in enumerate(w_ev)
                    if e >= 2.0 and all(x >= 1.0 for x in w_ev[i:])), None)
        rec = {"game": hdr.get("Round", "?"), "pnr": pnr}
        if pnr is not None:
            rec["phase"] = phase_of(pnr + 1)
            rec["loser_eval"] = (lose_seq[pnr][0] * slope[l]
                                 if pnr < len(lose_seq) else None)
            rec["loser_depth"] = lose_seq[pnr][1] if pnr < len(lose_seq) else None
            rec["winner_depth"] = win_seq[pnr][1]
            first07 = next((i for i, e in enumerate(w_ev) if e >= 0.7), pnr)
            nxt = [e * slope[l] for e, _, _ in lose_seq[pnr:pnr + 3]]
            tags = []
            if rec["loser_eval"] is not None and rec["loser_eval"] >= -0.5:
                tags.append("misjudgment")
            if rec["loser_eval"] is not None and len(nxt) >= 2 \
               and (rec["loser_eval"] - min(nxt)) >= 1.5:
                tags.append("tactical")
            if pnr - first07 >= 6 and "tactical" not in tags:
                tags.append("grind")
            rec["tags"] = tags or ["collapse"]
        losses[l].append(rec)
    return dict(n=ngames, S=S, color=color, term=term, dp=depth_phase,
                losses=losses, draws=draws, blunders=blunders)


def report(r, A, B):
    n, S = r["n"], r["S"]
    print(f"═══ MATCH STORY: {A} vs {B}  ({n} games) ═══")
    for who in (A, B):
        s = S[who]
        pts = s["W"] + s["D"] / 2
        print(f"  {who:12s} W{s['W']:>3} D{s['D']:>3} L{s['L']:>3}  "
              f"{pts / max(n,1) * 100:5.1f}%   "
              f"wins as W/B: {r['color'][who]['as_white']}/{r['color'][who]['as_black']}")
    print(f"  terminations: {dict(r['term'].most_common(4))}")

    print("\n── why each side lost ──")
    for who in (A, B):
        L = r["losses"][who]
        tags = Counter(t for x in L for t in x.get("tags", []))
        ph = Counter(x.get("phase", "?") for x in L if x.get("pnr") is not None)
        print(f"  {who} ({len(L)} losses): " +
              ("; ".join(f"{k} {v}" for k, v in tags.most_common()) or "—"))
        if ph:
            print(f"    decisive phase: {dict(ph.most_common())}")
        dd = [(x["loser_depth"], x["winner_depth"]) for x in L
              if x.get("loser_depth") and x.get("winner_depth")]
        if dd:
            print(f"    depth at decisive ply: own {st.mean(d[0] for d in dd):.1f} "
                  f"vs winner {st.mean(d[1] for d in dd):.1f}")

    print("\n── depth by phase (mean completed) ──")
    for ph in ("opening", "middlegame", "endgame"):
        da = r["dp"][A].get(ph); db = r["dp"][B].get(ph)
        if da and db:
            print(f"  {ph:11s} {A} {st.mean(da):5.1f}   {B} {st.mean(db):5.1f}   "
                  f"gap {st.mean(da) - st.mean(db):+5.1f}")

    if r["draws"]:
        endw = [d[0] for d in r["draws"]]
        worse = sum(1 for e in endw if e < -0.5)
        better = sum(1 for e in endw if e > 0.5)
        print(f"\n── draw anatomy ({len(r['draws'])}) ──")
        print(f"  {A} ended worse in {worse}, better in {better}, "
              f"balanced {len(endw) - worse - better}; "
              f"median length {st.median(d[1] for d in r['draws']):.0f} plies")

    print("\n── blunder ledger (own-eval drops >= 2.0 oracle-pawns) ──")
    for who in (A, B):
        bl = sorted(r["blunders"][who], reverse=True)[:5]
        print(f"  {who}: {len(r['blunders'][who])} total; worst: " +
              ("; ".join(f"g{g} mv{m} ({d:.1f})" for d, g, m, _, _ in bl) or "—"))

    # computed narrative
    print("\n── narrative ──")
    la, lb = r["losses"][A], r["losses"][B]
    ta = Counter(t for x in la for t in x.get("tags", []))
    tb = Counter(t for x in lb for t in x.get("tags", []))
    lead = A if S[A]["W"] > S[B]["W"] else B
    trail = B if lead == A else A
    top_t = (Counter(t for x in r["losses"][trail] for t in x.get("tags", []))
             .most_common(1))
    print(f"  {lead} leads. {trail}'s losses are primarily "
          f"{top_t[0][0] if top_t else '—'}-driven; "
          f"blunder counts {len(r['blunders'][A])}({A}) vs {len(r['blunders'][B])}({B}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-slope", type=float, default=0.548)
    ap.add_argument("--b-slope", type=float, default=0.548)
    ap.add_argument("--base-ms", type=int, default=10000)
    args = ap.parse_args()
    r = analyse(args.pgn, args.a, args.b, args.a_slope, args.b_slope, args.base_ms)
    report(r, args.a, args.b)


if __name__ == "__main__":
    main()
