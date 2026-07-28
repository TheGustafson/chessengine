"""The gift corpus, calibrated -- every error we made in a lost game.

WHY this supersedes `harvest.py`: that one asked Stockfish at 100k nodes, and a
budget sweep on the single most-studied blunder we own (game 18, Be2) reads

      50k  -15      256k  +212      4M  +278
     100k  +24      500k  +257     16M  +325
     120k  +82        1M  +258

-- i.e. at the harvest budget our canonical blunder scores +24 cp and does not
clear the 100 cp bar.  The refutation is a quiet deflection that only resolves
past depth ~16, so a 100k-node oracle is BLIND to exactly the class of error we
care most about.  That is the same failure as the 2-ply own-eval detector
before it (~9% recall), one level up, and for the same root cause: both had
their sensitivity set by a COST budget instead of by a POSITIVE CONTROL.

So this script refuses to run unless it first re-detects game 18.  Any future
detector in this directory should do the same: calibrate against a known
positive, or you are measuring your own budget rather than the engine.

Two-stage for cost: ask for the best move first (one search), and only pay for
the forced-move probe when what we played is not it.

    .venv/bin/python tools/forensics/harvest2.py [--nodes 1000000] [--workers 5]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine  # noqa: E402

REPO = "/home/nick/projects/chessengine"
SF = REPO + "/engines/bin/stockfish18"
ANNOT = re.compile(r"([-+]?\d+(?:\.\d+)?|[+-]?M\d+)/(\d+)\s+([\d.]+)s")

# The positive control: game 18 move 55.  Be2 loses ~260 cp to Rc5 and the
# refutation needs depth ~17.  If a budget cannot see this, it cannot see the
# class, and the run is aborted rather than reported as a clean sweep.
CONTROL_FEN = "8/k2bN3/Pn2p3/R4p1B/7P/1rp3P1/5K2/8 w - - 0 55"
CONTROL_MOVE = "h5e2"
CONTROL_MIN_LOSS = 150


def wp(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / 170.0))


def collect() -> list[dict]:
    """Every ARCHON move from a game ARCHON lost, both legs."""
    import chess
    import chess.pgn

    out = []
    for leg in ("time", "nodes"):
        p = f"{REPO}/engines/runs/rematch-v1/{leg}/games.pgn"
        if not os.path.exists(p):
            continue
        with open(p, errors="ignore") as fh:
            while (g := chess.pgn.read_game(fh)) is not None:
                aw = "archon" in g.headers.get("White", "").lower()
                r = g.headers.get("Result", "*")
                if not ((r == "1-0" and not aw) or (r == "0-1" and aw)):
                    continue
                node = g
                while node.variations:
                    node = node.variations[0]
                    m = ANNOT.search(node.comment or "")
                    if not m:
                        continue
                    b = node.parent.board()
                    if (b.turn == chess.WHITE) == aw and not b.is_check():
                        out.append(dict(leg=leg, game=g.headers.get("Round", "?"),
                                        fen=b.fen(), played=node.move.uci(),
                                        game_depth=int(m.group(2))))
    return out


def score_one(sf: Engine, fen: str, played: str, nodes: int) -> dict | None:
    """Stage 1: best move.  Stage 2 (only on a mismatch): what we played."""
    sf.newgame()
    b = sf.go(fen=fen, nodes=nodes)
    if not b["ladder"]:
        return None
    bl = b["ladder"][max(b["ladder"])]
    if bl["move"] == played:
        return dict(loss=0, best=played, best_cp=bl["score"], played_cp=bl["score"],
                    best_depth=bl["depth"], played_depth=bl["depth"])
    sf.newgame()
    f = sf.go(fen=fen, nodes=nodes, searchmoves=[played])
    if not f["ladder"]:
        return None
    fl = f["ladder"][max(f["ladder"])]
    return dict(loss=bl["score"] - fl["score"], best=bl["move"], best_cp=bl["score"],
                played_cp=fl["score"], best_depth=bl["depth"], played_depth=fl["depth"])


def worker(chunk_and_nodes):
    chunk, nodes = chunk_and_nodes
    sf = Engine(SF, net=None, hash_mb=256, name="sf18")
    res = []
    for row in chunk:
        try:
            s = score_one(sf, row["fen"], row["played"], nodes)
        except Exception:  # noqa: BLE001
            s = None
        if s is None:
            continue
        r = dict(row, **s)
        r["wp_loss"] = round(wp(s["best_cp"]) - wp(s["played_cp"]), 3)
        res.append(r)
    sf.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--min-loss", type=int, default=100)
    ap.add_argument("--out", default=os.environ.get("S", "/tmp") + "/gift_corpus_deep.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # --- the positive control, before anything else --------------------------
    sf = Engine(SF, net=None, hash_mb=256, name="sf18-control")
    c = score_one(sf, CONTROL_FEN, CONTROL_MOVE, args.nodes)
    sf.close()
    if c is None or c["loss"] < CONTROL_MIN_LOSS:
        raise SystemExit(
            f"CONTROL FAILED: game 18 Be2 scores {c and c['loss']} cp at "
            f"{args.nodes} nodes, need >= {CONTROL_MIN_LOSS}. Raise --nodes; a "
            f"budget that cannot see the control cannot see the class."
        )
    print(f"control OK: game-18 Be2 = {c['loss']} cp at {args.nodes} nodes "
          f"(best {c['best']} d{c['best_depth']})", flush=True)

    rows = collect()
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} ARCHON moves from lost games -> {args.workers} workers "
          f"@ {args.nodes} nodes", flush=True)

    n = args.workers
    chunks = [(rows[i::n], args.nodes) for i in range(n)]
    with mp.Pool(n) as pool:
        got = pool.map(worker, chunks)
    scored = [r for part in got for r in part]

    hits = [r for r in scored if r["loss"] >= args.min_loss]
    hits.sort(key=lambda h: -h["wp_loss"])
    json.dump(hits, open(args.out, "w"), indent=1)
    print(f"\n{len(scored)} scored, {len(hits)} errors >= {args.min_loss} cp "
          f"({100*len(hits)/max(1,len(scored)):.1f}%) -> {args.out}")
    print(f"{'game':<7}{'played':>8}{'best':>8}{'cp':>7}{'wp':>7}{'d':>4}")
    for h in hits[:25]:
        print(f"g{h['game']:<6}{h['played']:>8}{str(h['best']):>8}"
              f"{h['loss']:>7}{h['wp_loss']:>7.2f}{h['game_depth']:>4}")


if __name__ == "__main__":
    main()
