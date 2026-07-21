#!/usr/bin/env python3
"""SF relabel at a chosen node budget; order-preserving; identity-safe pairing.

Label-value pilot (2026-07-18): relabel a fixed FEN set at a parametrized node
budget so a cheap (5k) vs deep (100k) A/B controls everything except search
depth. Input is `FEN | cp | result` OR `FEN` (one per line); the cp column, if
present, is DISCARDED (we relabel fresh). Output: `FEN | cp | result`
(cp = White-relative centipawns). Order is preserved so cheap/deep files stay
line-aligned for the calibration corroborator.

DATA-GRADE RULES (2026-07-21, docs/research/data-recipe-contract.md):
* WITHOUT --curate the output is a PILOT INSTRUMENT, not training data: rows
  are never dropped (line alignment is the contract) so mates are CLAMPED to
  +-2500 — which hides them from any downstream |cp|-based filter (the exact
  trap that put 86k mate labels in the hardneg aux). Never feed non-curate
  output to bf-convert/training.
* WITH --curate the output is training-grade: the filter mirrors the
  trainer-parity criteria (ply>=16 from the FEN counters, not-in-check, mates
  and |cp|>=2500 dropped, best-PV-move non-capture/non-promotion).
"""
import argparse
import math
import multiprocessing as mp
import os
import sys

import chess
import chess.engine

SCORE_CAP = 2500


def parse(line):
    parts = [x.strip() for x in line.rstrip("\n").replace("|", "\t").split("\t") if x.strip()]
    if not parts:
        return None
    fen = parts[0]
    # result (white-relative WDL in {0,0.5,1}) is the last field when present.
    result = parts[-1] if len(parts) >= 3 else "0.5"
    return fen, result


def init(sf):
    init.eng = chess.engine.SimpleEngine.popen_uci(sf)
    init.eng.configure({"Threads": 1, "Hash": 16})


def work(args):
    idx, fen, result, nodes, curate = args
    try:
        board = chess.Board(fen)
    except ValueError:
        return (idx, None)
    if curate:
        if board.is_check() or not board.is_valid():
            return (idx, None)
        # trainer-parity ply>=16 gate (game ply from the FEN counters; the
        # inputs are real-game positions so the counters carry true game ply).
        ply = 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)
        if ply < 16:
            return (idx, None)
    try:
        info = init.eng.analyse(board, chess.engine.Limit(nodes=nodes))
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        return (idx, None)
    sc = info["score"].white()
    if sc.is_mate():
        if curate:
            return (idx, None)
        cp = SCORE_CAP if sc.mate() > 0 else -SCORE_CAP
        return (idx, f"{fen} | {cp} | {result}")
    cp = sc.score()
    if cp is None:
        return (idx, None)
    if curate:
        if abs(cp) >= SCORE_CAP:
            return (idx, None)
        pv = info.get("pv")
        if not pv or board.is_capture(pv[0]) or pv[0].promotion:
            return (idx, None)
    cp = max(-SCORE_CAP, min(SCORE_CAP, cp))
    return (idx, f"{fen} | {cp} | {result}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--sf", required=True)
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit-fens")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--curate", action="store_true")
    ap.add_argument("--max-rows", type=int)
    a = ap.parse_args()

    rows = []
    with open(a.input) as fh:
        for i, ln in enumerate(fh):
            if not ln.strip() or ln.startswith("#"):
                continue
            parsed = parse(ln)
            if parsed is None:
                continue
            fen, result = parsed
            rows.append((i, fen, result, a.nodes, a.curate))
            if a.max_rows and len(rows) >= a.max_rows:
                break

    if not a.curate:
        print(
            "relabel_depth: --curate OFF -> pilot-instrument output (mates clamped to "
            "+-2500, no quiet filter). NOT training-grade; never bf-convert this.",
            file=sys.stderr,
        )
    os.nice(10)
    with mp.Pool(a.workers, initializer=init, initargs=(a.sf,)) as pool:
        res = pool.map(work, rows, chunksize=256)
    res.sort(key=lambda x: x[0])
    kept = [line for _, line in res if line]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(kept) + ("\n" if kept else ""))
    if a.emit_fens:
        with open(a.emit_fens, "w") as fh:
            fh.write("\n".join(l.split(" | ")[0] for l in kept) + ("\n" if kept else ""))
    print(f"wrote {len(kept)}/{len(rows)} rows -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
