#!/usr/bin/env python3
"""Build a small crossed LC0-position corpus with fresh Stockfish value labels.

The output preserves the selected LC0 features, policy, result, and original
``eval.npy`` exactly, and adds ``eval_sf.npy``. This permits a strict training
A/B in which only the value teacher changes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np


MATE_SCORE = 2500
ARRAYS = ("feats", "moves", "probs", "best", "eval", "result")


def board_from_features(row: np.ndarray) -> chess.Board:
    board = chess.Board.empty()
    for raw in row:
        feature = int(raw)
        if feature < 0:
            break
        colour, rem = divmod(feature, 384)
        piece_type, square = divmod(rem, 64)
        board.set_piece_at(square, chess.Piece(piece_type + 1, colour == 0))
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    if not board.is_valid():
        raise ValueError(f"invalid reconstructed board: {board.fen()}")
    return board


def label_chunk(args: tuple[str, int, np.ndarray]) -> np.ndarray:
    engine_path, nodes, features = args
    scores = np.empty(len(features), dtype=np.int16)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=30)
    engine.configure({"Threads": 1, "Hash": 32})
    try:
        for index, row in enumerate(features):
            board = board_from_features(row)
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
            score = info["score"].pov(chess.WHITE).score(mate_score=MATE_SCORE)
            if score is None:
                raise RuntimeError(f"Stockfish returned no score for {board.fen()}")
            scores[index] = int(np.clip(score, -MATE_SCORE, MATE_SCORE))
    finally:
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows-per-side", type=int, default=25_000)
    parser.add_argument("--source-rows-per-side", type=int, default=50_000_000)
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--stockfish", type=Path, default=Path("engines/stockfish-datagen"))
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report: dict[str, object] = {
        "source": str(args.source.resolve()),
        "stockfish": str(args.stockfish.resolve()),
        "rows_per_side": args.rows_per_side,
        "source_rows_per_side": args.source_rows_per_side,
        "nodes": args.nodes,
        "workers": args.workers,
        "seed": args.seed,
        "sides": {},
    }

    for side in ("white", "black"):
        source_dir = args.source / side
        output_dir = args.output / side
        output_dir.mkdir(parents=True)
        available = min(
            args.source_rows_per_side,
            len(np.load(source_dir / "feats.npy", mmap_mode="r")),
        )
        indices = np.sort(
            rng.choice(available, size=args.rows_per_side, replace=False)
        )
        selected: dict[str, np.ndarray] = {}
        for name in ARRAYS:
            values = np.load(source_dir / f"{name}.npy", mmap_mode="r")
            selected[name] = np.asarray(values[indices])
            np.save(output_dir / f"{name}.npy", selected[name], allow_pickle=False)
        np.save(output_dir / "source_index.npy", indices, allow_pickle=False)

        chunks = [chunk for chunk in np.array_split(selected["feats"], args.workers) if len(chunk)]
        jobs = [(str(args.stockfish.resolve()), args.nodes, chunk) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            scores = np.concatenate(list(executor.map(label_chunk, jobs)))
        np.save(output_dir / "eval_sf.npy", scores, allow_pickle=False)

        lc0 = selected["eval"].astype(np.float64)
        sf = scores.astype(np.float64)
        side_report = {
            "rows": len(indices),
            "first_source_index": int(indices[0]),
            "last_source_index": int(indices[-1]),
            "lc0_sf_pearson": float(np.corrcoef(lc0, sf)[0, 1]),
            "lc0_sf_mae_cp": float(np.mean(np.abs(lc0 - sf))),
            "sf_min_cp": int(scores.min()),
            "sf_max_cp": int(scores.max()),
        }
        report["sides"][side] = side_report
        print(f"{side}: {side_report}", flush=True)

    (args.output / "meta.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
