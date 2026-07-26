#!/usr/bin/env python3
"""loss_forensics — WHY do we lose? Per-game decisive-moment analysis of a
fastchess match PGN, using both engines' own eval/depth/time trails.

For every decisive game the script locates the POINT OF NO RETURN (pnr): the
first ply where the WINNER's reported eval reaches +2.00 (its POV) and never
falls back under +1.00. Around that ply it captures both engines' evals
(disagreement = misjudgment fingerprint), our depth vs theirs, both clocks,
game phase, and our own eval's subsequent collapse speed (tactical-collapse
fingerprint). Aggregates classify each loss:

  misjudgment  we said fine (>= -0.5 our POV) at pnr while the winner said
               won — our eval disagreed with reality where it decides games
  tactical     our eval collapsed >= 1.5 pawns within our next 2 moves after
               pnr — we saw the refutation only after committing
  grind        winner's eval drifted up slowly (>= 12 plies from +0.7 to
               +2.0) with no single collapse — outplayed move by move
  timepressure our clock at pnr < 20% of base — quality decayed with clock

(One loss can carry several tags; `grind` is the residual when nothing acute
fired.) Run on an equal-TIME and an equal-NODE leg of the same pairing and
diff the taxonomies: tags that vanish at equal nodes are bought by SPEED,
tags that persist are net/search quality.

Usage:
  loss_forensics.py GAMES.pgn --us archon --base-ms 10000 [--json out.json]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import statistics as st
import sys

TOK = re.compile(r'([^\s{}]+)\s+\{([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s\}')


def parse_games(path: str):
    raw = open(path, errors="ignore").read()
    for chunk in re.split(r'\n\n(?=\[Event )', raw):
        if "{" not in chunk:
            continue
        hdr = dict(re.findall(r'\[(\w+) "([^"]*)"\]', chunk))
        body = chunk[chunk.rindex('"]') + 2:]
        moves = []
        for m in TOK.finditer(body):
            _, ev, dep, tm = m.groups()
            ev = float(ev.replace("M", "")) * (100.0 if "M" in ev else 1.0)
            moves.append((ev, int(dep), float(tm)))
        if moves:
            yield hdr, moves


def phase_of(fullmove: int) -> str:
    return "opening" if fullmove <= 12 else ("middlegame" if fullmove <= 32 else "endgame")


def analyse(path: str, us: str, base_ms: int, wslope: float = 1.0):
    losses, wins, draws = [], [], []
    depth_by_phase = {}          # (who, phase) -> [depths]
    clock_track = {}             # (who, move#) -> [clock ms]
    for hdr, moves in parse_games(path):
        white, res = hdr.get("White", ""), hdr.get("Result", "*")
        our_white = us.lower() in white.lower()
        # PGN move list starts with the side the FEN puts on move; fastchess
        # books can start either way. Ply parity: PlyCount/FEN not needed —
        # infer from FEN header stm if present.
        fen = hdr.get("FEN", "")
        white_first = (" w " in fen) if fen else True
        our_off = 0 if (our_white == white_first) else 1

        ours = moves[our_off::2]
        theirs = moves[1 - our_off::2]

        # clock reconstruction (base + inc, per side)
        inc = 100.0
        clk, our_clk = base_ms, []
        for _, _, t in ours:
            our_clk.append(clk); clk = clk - t * 1000 + inc
        clk, their_clk = base_ms, []
        for _, _, t in theirs:
            their_clk.append(clk); clk = clk - t * 1000 + inc

        for who, seq in (("us", ours), ("them", theirs)):
            for i, (_, dep, _) in enumerate(seq):
                fm = (2 * i + 1) // 2 + 1
                depth_by_phase.setdefault((who, phase_of(fm)), []).append(dep)
        for who, cl in (("us", our_clk), ("them", their_clk)):
            for i, c in enumerate(cl):
                if (i + 1) in (10, 20, 30, 40):
                    clock_track.setdefault((who, i + 1), []).append(c)

        we_won = (res == "1-0") == our_white if res in ("1-0", "0-1") else None
        if res == "1/2-1/2":
            draws.append(hdr); continue
        if we_won is None:
            continue
        if we_won:
            wins.append(hdr); continue

        # ---- LOSS: find the winner's point of no return ----
        w_ev = [e * wslope for e, _, _ in theirs]
        pnr = None
        for i, e in enumerate(w_ev):
            if e >= 2.0 and all(x >= 1.0 for x in w_ev[i:]):
                pnr = i; break
        if pnr is None:
            losses.append({"game": hdr.get("Round", "?"), "tags": ["unclassified"],
                           "term": hdr.get("Termination", "")})
            continue
        fm_pnr = pnr + 1
        our_at = ours[pnr][0] if pnr < len(ours) else None
        our_depth_at = ours[pnr][1] if pnr < len(ours) else None
        their_depth_at = theirs[pnr][1]
        our_clock_at = our_clk[pnr] if pnr < len(our_clk) else None

        tags = []
        if our_at is not None and our_at >= -0.5:
            tags.append("misjudgment")
        nxt = [e for e, _, _ in ours[pnr:pnr + 3]]
        if our_at is not None and len(nxt) >= 2 and (our_at - min(nxt)) >= 1.5:
            tags.append("tactical")
        if our_clock_at is not None and our_clock_at < 0.2 * base_ms:
            tags.append("timepressure")
        # grind: winner needed >= 12 plies to go +0.7 -> +2.0
        first07 = next((i for i, e in enumerate(w_ev) if e >= 0.7), pnr)
        if pnr - first07 >= 6 and "tactical" not in tags:
            tags.append("grind")
        if not tags:
            tags.append("collapse")

        losses.append({
            "game": hdr.get("Round", "?"),
            "pnr_move": fm_pnr,
            "phase": phase_of(fm_pnr),
            "our_eval_at_pnr": our_at,
            "winner_eval_at_pnr": w_ev[pnr],
            "our_depth_at_pnr": our_depth_at,
            "their_depth_at_pnr": their_depth_at,
            "our_clock_at_pnr_s": round(our_clock_at / 1000, 2) if our_clock_at else None,
            "their_clock_at_pnr_s": round(their_clk[pnr] / 1000, 2) if pnr < len(their_clk) else None,
            "tags": tags,
            "term": hdr.get("Termination", ""),
        })
    return losses, wins, draws, depth_by_phase, clock_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--us", default="archon")
    ap.add_argument("--base-ms", type=int, default=10000)
    ap.add_argument("--winner-slope", type=float, default=1.0,
                    help="eval_bench slope of the WINNER engine's reported cp "
                         "onto oracle cp; pnr thresholds are divided by it so "
                         "+2.00 means +2.00 in ORACLE units regardless of the "
                         "winner's internal scale (ARCHON ~0.518, SF16 ~0.937)")
    ap.add_argument("--json")
    a = ap.parse_args()

    losses, wins, draws, dbp, ct = analyse(a.pgn, a.us, a.base_ms, a.winner_slope)
    n = len(losses) + len(wins) + len(draws)
    print(f"{a.pgn}\n  games {n}: W {len(wins)}  D {len(draws)}  L {len(losses)}\n")

    print("DEPTH BY PHASE (mean completed depth per move)")
    for ph in ("opening", "middlegame", "endgame"):
        u = dbp.get(("us", ph), []); t = dbp.get(("them", ph), [])
        if u and t:
            print(f"  {ph:11s} us {st.mean(u):5.1f}  them {st.mean(t):5.1f}  "
                  f"gap {st.mean(u) - st.mean(t):+5.1f}")

    print("\nCLOCK TRACK (mean remaining, s)")
    for mv in (10, 20, 30, 40):
        u = ct.get(("us", mv), []); t = ct.get(("them", mv), [])
        if u and t:
            print(f"  move {mv:2d}: us {st.mean(u)/1000:5.2f}  them {st.mean(t)/1000:5.2f}")

    print("\nLOSS TAXONOMY")
    from collections import Counter
    c = Counter(tag for L in losses for tag in L["tags"])
    for tag, k in c.most_common():
        print(f"  {tag:13s} {k:3d} / {len(losses)}")
    byph = Counter(L.get("phase", "?") for L in losses)
    print("  by phase: " + ", ".join(f"{p} {k}" for p, k in byph.most_common()))

    mj = [L for L in losses if "misjudgment" in L["tags"] and L.get("our_eval_at_pnr") is not None]
    if mj:
        print(f"\n  misjudgment detail: our mean eval at pnr "
              f"{st.mean(x['our_eval_at_pnr'] for x in mj):+.2f} "
              f"while winner said >= +2.00 (n={len(mj)})")
    dd = [(L["our_depth_at_pnr"], L["their_depth_at_pnr"]) for L in losses
          if L.get("our_depth_at_pnr") and L.get("their_depth_at_pnr")]
    if dd:
        print(f"  depth at pnr: us {st.mean(x[0] for x in dd):.1f} vs "
              f"them {st.mean(x[1] for x in dd):.1f} (n={len(dd)})")
    tp = [L for L in losses if L.get("our_clock_at_pnr_s") is not None]
    if tp:
        print(f"  our clock at pnr: median {st.median(x['our_clock_at_pnr_s'] for x in tp):.2f}s "
              f"(base {a.base_ms/1000:.0f}s)")

    if a.json:
        json.dump({"losses": losses, "n": n}, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
