#!/usr/bin/env python3
"""Validate a bootstrap NNUE text shard against the design/15 §5 contract.

Re-parses every `<FEN> | <score> | <result>` line emitted by gen_sf_data.py
and asserts the curation invariants independently of the generator:

  * FEN parses and is a legal position (python-chess);
  * side-to-move is NOT in check (§3.5.2);
  * `|score| < SCORE_CAP` and the line is not a mate sentinel (§3.5.4);
  * `result` is exactly one of {0.0, 0.5, 1.0} (White-relative, §3.2.2);
  * within a run every position key is UNIQUE (dedup filter worked, §3.5.5);
  * score is a plain integer (White-relative cp on disk).

It CANNOT re-derive the quiet-bestmove filter (§3.5.3) or the exact
White-relative sign without a search; pass --sf PATH to additionally, on a
random --resample subset, run a fresh SF search and confirm (a) the stored
bestmove-would-be is quiet and (b) the sign of the White-relative score
matches SF's `pov(WHITE)` -- the §3.2.4 sign-flip tripwire and §5
perspective round-trip. Without --sf those two checks are skipped (reported).

Reports filter-independent health: the piece-count and game-phase histograms
(§3.5.7 balance monitor) and the result distribution.

Usage:
    python3 tools/nnue/validate_sample.py data/nnue/raw/shard_1_0.txt
    python3 tools/nnue/validate_sample.py data/nnue/raw/*.txt \
        --sf engines/stockfish-datagen --resample 40 --nodes 5000

Exit 0 iff every line passes every applicable assertion.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import chess.engine
import chess.polyglot

SCORE_CAP = 2500
MATE_SENTINEL = 100_000


def phase_of(board: chess.Board) -> int:
    w = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    ph = sum(wt * (len(board.pieces(pt, chess.WHITE))
                   + len(board.pieces(pt, chess.BLACK)))
             for pt, wt in w.items())
    return min(24, ph)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="validate NNUE text shard(s), design/15 §5")
    ap.add_argument("shards", nargs="+", help="text shard file(s)")
    ap.add_argument("--score-cap", type=int, default=SCORE_CAP)
    ap.add_argument("--sf", default="", help="SF for the perspective round-trip subset")
    ap.add_argument("--resample", type=int, default=0, help="lines to re-search with --sf")
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    lines: list[tuple[str, str, int, str]] = []  # (shard, fen, score, result)
    for path in a.shards:
        for ln, raw in enumerate(Path(path).read_text().splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            lines.append((f"{path}:{ln}", raw, 0, ""))

    errors: list[str] = []
    seen: dict[int, str] = {}
    pc_hist: dict[int, int] = {}
    ph_hist: dict[int, int] = {}
    res_dist = {"0.0": 0, "0.5": 0, "1.0": 0}
    parsed = []  # (loc, board, score, result_str)

    for loc, raw, _, _ in lines:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) != 3:
            errors.append(f"{loc}: expected 3 '|'-fields, got {len(parts)}: {raw!r}")
            continue
        fen, score_s, result_s = parts
        # FEN legal?
        try:
            board = chess.Board(fen)
        except ValueError as e:
            errors.append(f"{loc}: bad FEN: {e}")
            continue
        if not board.is_valid():
            errors.append(f"{loc}: illegal position: {fen}")
            continue
        # score integer, in-band, not mate sentinel
        try:
            score = int(score_s)
        except ValueError:
            errors.append(f"{loc}: score not an integer: {score_s!r}")
            continue
        if abs(score) >= a.score_cap:
            errors.append(f"{loc}: |score|={abs(score)} >= cap {a.score_cap}")
        if abs(score) >= MATE_SENTINEL - 10_000:
            errors.append(f"{loc}: looks like a mate sentinel: {score}")
        # result one of the three
        if result_s not in res_dist:
            errors.append(f"{loc}: result not in {{0.0,0.5,1.0}}: {result_s!r}")
        else:
            res_dist[result_s] += 1
        # not in check
        if board.is_check():
            errors.append(f"{loc}: side to move in check (should be filtered)")
        # dedup
        key = chess.polyglot.zobrist_hash(board)
        if key in seen:
            errors.append(f"{loc}: duplicate position (also {seen[key]})")
        else:
            seen[key] = loc
        pc = chess.popcount(board.occupied)
        pc_hist[pc] = pc_hist.get(pc, 0) + 1
        ph = phase_of(board)
        ph_hist[ph] = ph_hist.get(ph, 0) + 1
        parsed.append((loc, board, score, result_s))

    # Optional SF round-trip: quiet-bestmove + White-relative sign (§3.2.4/§5).
    rt_checked = 0
    if a.sf and a.resample and parsed:
        rng = random.Random(a.seed)
        subset = rng.sample(parsed, min(a.resample, len(parsed)))
        eng = chess.engine.SimpleEngine.popen_uci(a.sf, timeout=30)
        eng.configure({"Threads": 1, "Hash": 16})
        try:
            for loc, board, score, _ in subset:
                info = eng.analyse(board, chess.engine.Limit(nodes=a.nodes),
                                   game=object())
                sc = info["score"]
                sw = sc.white().score(mate_score=MATE_SENTINEL)
                bm = (info.get("pv") or [None])[0]
                # (a) quiet bestmove
                if bm is not None and (board.is_capture(bm) or bm.promotion):
                    errors.append(f"{loc}: SF bestmove {bm} is noisy (not quiet)")
                # (b) sign of White-relative score agrees with SF pov(WHITE).
                # Node counts differ from generation time, so compare SIGN /
                # rough magnitude, not the exact cp (the §3.2.4 tripwire is the
                # SIGN being right, i.e. we didn't forget to flip for Black).
                if score != 0 and sw != 0 and (score > 0) != (sw > 0):
                    errors.append(
                        f"{loc}: White-relative SIGN mismatch: disk={score} "
                        f"fresh-SF-white={sw} (§3.2.4 sign-flip trap)")
                rt_checked += 1
        finally:
            try:
                eng.quit()
            except Exception:
                eng.close()

    # --- report ---
    n = len(parsed)
    print(f"lines parsed        : {n}")
    print(f"unique positions    : {len(seen)}  (dup errors: "
          f"{sum(1 for e in errors if 'duplicate' in e)})")
    print(f"result distribution : W={res_dist['1.0']} D={res_dist['0.5']} "
          f"L={res_dist['0.0']}")
    if a.sf and a.resample:
        print(f"SF round-trip checks: {rt_checked} (quiet-bestmove + White sign)")
    else:
        print("SF round-trip checks: SKIPPED (pass --sf + --resample to enable)")

    def show_hist(name, h, lo, hi):
        print(f"{name}:")
        mx = max(h.values()) if h else 1
        for k in range(lo, hi + 1):
            c = h.get(k, 0)
            if c:
                bar = "#" * max(1, round(40 * c / mx))
                print(f"  {k:>2}: {c:>7} {bar}")

    show_hist("piece-count histogram (2..32)", pc_hist, 2, 32)
    show_hist("game-phase histogram (0..24)", ph_hist, 0, 24)

    if errors:
        print(f"\nFAIL: {len(errors)} error(s):", file=sys.stderr)
        for e in errors[:50]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more", file=sys.stderr)
        return 1
    print("\nOK: all lines pass the design/15 §5 line-format + dedup contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
