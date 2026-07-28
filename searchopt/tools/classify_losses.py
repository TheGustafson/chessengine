#!/usr/bin/env python3
"""Split a mined loss corpus into the categories that decide whether a loss is OURS.

Established on the D47V corpus (searchopt/LEDGER.md) and the reason this pass
exists at all: **only about a third of our "blunders" are our defect.** SF18
plays the losing move on a quarter of our own loss positions at our own budget,
and another large slice are near-ties for BOTH engines. Working on those is
wasted effort, so the corpus has to be split before anyone walks a position.

At the IN-GAME budget (default 110k nodes):

  OURS      SF avoids the blunder, we play it        -> the actionable defect
  BOTH      SF plays it too                          -> position hardness, skip
  NEITHER   we avoid it from a cold search           -> in-game context or
                                                        argmax instability

Also records the **top1-top2 separation** for both engines via MultiPV 2, because
the character of the OURS class is the load-bearing finding: on those positions
WE are the confident one (~55 cp separation vs SF's ~10), i.e. we commit to a
losing move rather than failing to discriminate. Any fix that sharpens eval
makes that worse.

Every score here stays inside one engine (ours vs ours, SF vs SF), so no
cross-engine frame conversion is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine, NET, REPO  # noqa: E402

INFO = re.compile(r"^info depth (\d+).*?multipv (\d+).*?score (cp|mate) (-?\d+)")
BOUND = re.compile(r"\b(lowerbound|upperbound)\b")


def top2_gap(e, fen, nodes):
    """top1 - top2 at the final depth, in the engine's OWN units."""
    e.send("setoption name MultiPV value 2"); e.send("isready"); e.wait_for("readyok")
    e.newgame(); e.send(f"position fen {fen}"); e.send(f"go nodes {nodes}")
    best = {}
    while True:
        ln = e.p.stdout.readline()
        if not ln or ln.startswith("bestmove"):
            break
        if BOUND.search(ln):
            continue
        m = INFO.match(ln)
        if m:
            d, mp, kind, sc = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            v = sc if kind == "cp" else (30000 - abs(sc)) * (1 if sc > 0 else -1)
            best.setdefault(d, {})[mp] = v
    e.send("setoption name MultiPV value 1"); e.send("isready"); e.wait_for("readyok")
    ok = [d for d in best if 1 in best[d] and 2 in best[d]]
    return (best[max(ok)][1] - best[max(ok)][2]) if ok else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", default="searchopt/positions/losses.json")
    ap.add_argument("--budget", type=int, default=110_000,
                    help="the in-game node budget; the whole split is defined AT it")
    ap.add_argument("--wp-min", type=float, default=0.10)
    ap.add_argument("--outdir", default="searchopt/positions")
    ap.add_argument("--engine", default="engines/canonical/archon")
    ap.add_argument("--oracle", default="engines/bin/stockfish18")
    a = ap.parse_args()
    os.chdir(REPO)

    raw = json.load(open(a.losses))
    if isinstance(raw, dict):                       # provenance-stamped form
        prov = raw.get("_provenance", {})
        cur = os.path.basename(os.path.realpath("engines/canonical/net.cenn"))
        if prov.get("net") and prov["net"] != cur:
            print(f"!! STALE CORPUS: mined with {prov['net']}, canonical is now {cur}.")
            print(f"   Positions mined from another net say nothing about this one. Re-mine.")
        raw = raw["rows"]
    rows = [r for r in raw if r["wp_loss"] >= a.wp_min]
    print(f"{len(rows)} moves with wp_loss >= {a.wp_min}")

    ar = Engine(a.engine, net=NET, hash_mb=64, name="archon"); ar.assert_net_live()
    sf = Engine(a.oracle, net=None, hash_mb=64, name="sf18")
    out = []
    for i, r in enumerate(rows, 1):
        ar.newgame(); we = ar.go(fen=r["fen"], nodes=a.budget)["best"]
        sf.newgame(); th = sf.go(fen=r["fen"], nodes=a.budget)["best"]
        we_bl, sf_bl = (we == r["played"]), (th == r["played"])
        cls = "OURS" if (we_bl and not sf_bl) else ("BOTH" if we_bl else "NEITHER")
        out.append({**r, "class": cls, "our_move_at_budget": we, "sf_move_at_budget": th,
                    "our_gap": top2_gap(ar, r["fen"], a.budget),
                    "sf_gap": top2_gap(sf, r["fen"], a.budget)})
        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}]", flush=True)
    ar.close(); sf.close()

    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.outdir, "classified.json"), "w"))
    for cls in ("OURS", "BOTH", "NEITHER"):
        sub = [r for r in out if r["class"] == cls]
        json.dump([{k: r[k] for k in ("fen", "played", "best", "wp_loss")} for r in sub],
                  open(os.path.join(a.outdir, f"{cls.lower()}.json"), "w"))
        gaps = [r["our_gap"] for r in sub if r["our_gap"] is not None]
        sgaps = [r["sf_gap"] for r in sub if r["sf_gap"] is not None]
        med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
        print(f"  {cls:<8} {len(sub):>4}  ({100*len(sub)/max(1,len(out)):4.0f}%)   "
              f"median top1-top2: ours {med(gaps):7.1f}  SF {med(sgaps):7.1f}")
    print(f"\n-> {a.outdir}/{{classified,ours,both,neither}}.json")
    print("Work the OURS set. BOTH is position hardness; NEITHER is context/instability.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
