#!/usr/bin/env python3
"""Audit LC0 value labels against fresh Stockfish on identical positions.

The policy corpus stores every position normalized to White-to-move, so the
stored eval and Stockfish POV are directly comparable. Castling rights are not
recoverable from piece placement; the `castle_geometry` subgroup identifies
positions where a king and home rook make that omission potentially relevant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import chess
import chess.engine
import numpy as np


MATE_SCORE = 2500


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
    return board


def possible_castle_geometry(board: chess.Board) -> bool:
    for colour, king_square, rooks in (
        (chess.WHITE, chess.E1, (chess.A1, chess.H1)),
        (chess.BLACK, chess.E8, (chess.A8, chess.H8)),
    ):
        king = board.piece_at(king_square)
        if king == chess.Piece(chess.KING, colour):
            for square in rooks:
                if board.piece_at(square) == chess.Piece(chess.ROOK, colour):
                    return True
    return False


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranked


def stats(lc0: np.ndarray, sf: np.ndarray, outcome: np.ndarray) -> dict:
    if len(lc0) < 2:
        return {"rows": int(len(lc0))}
    residual = sf - lc0
    slope, intercept = np.polyfit(lc0, sf, 1)
    decisive = (np.abs(lc0) >= 50) & (np.abs(sf) >= 50)
    sigmoid_lc0 = 1.0 / (1.0 + np.exp(-lc0 / 400.0))
    sigmoid_sf = 1.0 / (1.0 + np.exp(-sf / 400.0))
    blended = 0.3 * outcome + 0.7 * sigmoid_lc0
    scale_brier = {}
    for scale in (100, 150, 200, 250, 300, 400, 600, 800):
        probability = 1.0 / (1.0 + np.exp(-lc0 / scale))
        scale_brier[str(scale)] = {
            "value_only": float(np.mean((probability - outcome) ** 2)),
            "wdl03_blend": float(np.mean((0.3 * outcome + 0.7 * probability - outcome) ** 2)),
        }
    return {
        "rows": int(len(lc0)),
        "pearson": float(np.corrcoef(lc0, sf)[0, 1]),
        "spearman": float(np.corrcoef(ranks(lc0), ranks(sf))[0, 1]),
        "sf_on_lc0_slope": float(slope),
        "sf_on_lc0_intercept": float(intercept),
        "mae_cp": float(np.mean(np.abs(residual))),
        "rmse_cp": float(np.sqrt(np.mean(residual ** 2))),
        "median_abs_error_cp": float(np.median(np.abs(residual))),
        "p90_abs_error_cp": float(np.quantile(np.abs(residual), 0.9)),
        "sign_agreement_all": float(np.mean(np.sign(lc0) == np.sign(sf))),
        "sign_agreement_both_abs_ge_50": (
            float(np.mean(np.sign(lc0[decisive]) == np.sign(sf[decisive])))
            if decisive.any() else None
        ),
        "decisive_rows": int(decisive.sum()),
        "lc0_sigmoid_outcome_brier": float(np.mean((sigmoid_lc0 - outcome) ** 2)),
        "sf_sigmoid_outcome_brier": float(np.mean((sigmoid_sf - outcome) ** 2)),
        "wdl03_blended_outcome_brier": float(np.mean((blended - outcome) ** 2)),
        "target_scale_brier": scale_brier,
    }


def comparison_stats(reference: np.ndarray, predicted: np.ndarray, outcome: np.ndarray) -> dict:
    residual = predicted - reference
    slope, intercept = np.polyfit(reference, predicted, 1)
    decisive = (np.abs(reference) >= 50) & (np.abs(predicted) >= 50)
    probability = 1.0 / (1.0 + np.exp(-predicted / 400.0))
    reference_probability = 1.0 / (1.0 + np.exp(-reference / 400.0))
    blended_training_target = 0.3 * outcome + 0.7 * reference_probability
    bands = {}
    for label, mask in (
        ("abs_lt_50", np.abs(reference) < 50),
        ("abs_50_to_199", (np.abs(reference) >= 50) & (np.abs(reference) < 200)),
        ("abs_ge_200", np.abs(reference) >= 200),
    ):
        band_residual = residual[mask]
        bands[label] = {
            "rows": int(mask.sum()),
            "mae_cp": float(np.mean(np.abs(band_residual))) if mask.any() else None,
            "rmse_cp": float(np.sqrt(np.mean(band_residual ** 2))) if mask.any() else None,
            "sign_agreement": (
                float(np.mean(np.sign(reference[mask]) == np.sign(predicted[mask])))
                if mask.any() else None
            ),
        }
    return {
        "rows": int(len(reference)),
        "pearson": float(np.corrcoef(reference, predicted)[0, 1]),
        "spearman": float(np.corrcoef(ranks(reference), ranks(predicted))[0, 1]),
        "predicted_on_reference_slope": float(slope),
        "predicted_on_reference_intercept": float(intercept),
        "mae_cp": float(np.mean(np.abs(residual))),
        "rmse_cp": float(np.sqrt(np.mean(residual ** 2))),
        "median_abs_error_cp": float(np.median(np.abs(residual))),
        "reference_std_cp": float(np.std(reference)),
        "predicted_std_cp": float(np.std(predicted)),
        "sign_agreement_all": float(np.mean(np.sign(reference) == np.sign(predicted))),
        "sign_agreement_both_abs_ge_50": (
            float(np.mean(np.sign(reference[decisive]) == np.sign(predicted[decisive])))
            if decisive.any() else None
        ),
        "decisive_rows": int(decisive.sum()),
        "outcome_brier": float(np.mean((probability - outcome) ** 2)),
        "blended_training_target_mse": float(
            np.mean((probability - blended_training_target) ** 2)
        ),
        "by_abs_reference": bands,
    }


def evaluate_net(engine: Path, net: Path, boards: list[chess.Board]) -> np.ndarray:
    result = subprocess.run(
        [str(engine), "nnue-eval", str(net)],
        input="\n".join(board.fen() for board in boards) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    scores = np.asarray([int(value) for value in result.stdout.split()], dtype=np.float64)
    if len(scores) != len(boards):
        raise SystemExit(f"{net}: engine returned {len(scores)} scores for {len(boards)} boards")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--stockfish", type=Path, default=Path("engines/stockfish-datagen"))
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument(
        "--rows-per-side-limit", type=int,
        help="sample only this valid prefix from preallocated corpus arrays",
    )
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", type=Path, default=Path("target/release/chessengine"))
    parser.add_argument(
        "--net", action="append", default=[], metavar="NAME=PATH",
        help="also compare an engine NNUE's predictions with LC0 and Stockfish",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    selected: list[tuple[chess.Board, int, float, str, int]] = []
    per_side = max(1, math.ceil(args.rows / 2))
    for side in ("white", "black"):
        root = args.data / side
        features = np.load(root / "feats.npy", mmap_mode="r")
        evals = np.load(root / "eval.npy", mmap_mode="r")
        results = np.load(root / "result.npy", mmap_mode="r")
        available = min(len(features), args.rows_per_side_limit or len(features))
        chosen = rng.choice(available, size=min(per_side, available), replace=False)
        for index in chosen:
            board = board_from_features(features[index])
            if not board.is_valid():
                raise SystemExit(f"invalid reconstructed board at {side}:{index}: {board.fen()}")
            selected.append((board, int(evals[index]), float(results[index]) / 2.0, side, int(index)))
    selected = selected[: args.rows]

    sf_scores: list[int] = []
    castle_geometry: list[bool] = []
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish), timeout=30)
    engine.configure({"Threads": 1, "Hash": 64})
    try:
        for number, (board, _, _, _, _) in enumerate(selected, 1):
            info = engine.analyse(board, chess.engine.Limit(nodes=args.nodes), game=object())
            score = info["score"].pov(chess.WHITE).score(mate_score=MATE_SCORE)
            if score is None:
                raise SystemExit(f"Stockfish returned no score for {board.fen()}")
            sf_scores.append(int(np.clip(score, -MATE_SCORE, MATE_SCORE)))
            castle_geometry.append(possible_castle_geometry(board))
            if number % 100 == 0:
                print(f"audited {number}/{len(selected)}", flush=True)
    finally:
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    lc0 = np.asarray([row[1] for row in selected], dtype=np.float64)
    outcome = np.asarray([row[2] for row in selected], dtype=np.float64)
    sf = np.asarray(sf_scores, dtype=np.float64)
    castle = np.asarray(castle_geometry, dtype=bool)
    report = {
        "data": str(args.data.resolve()),
        "stockfish": str(args.stockfish.resolve()),
        "nodes": args.nodes,
        "seed": args.seed,
        "all": stats(lc0, sf, outcome),
        "castle_geometry": stats(lc0[castle], sf[castle], outcome[castle]),
        "no_castle_geometry": stats(lc0[~castle], sf[~castle], outcome[~castle]),
    }
    if args.net:
        report["networks"] = {}
        boards = [row[0] for row in selected]
        for specification in args.net:
            if "=" not in specification:
                raise SystemExit(f"--net requires NAME=PATH, got {specification!r}")
            name, path = specification.split("=", 1)
            predicted = evaluate_net(args.engine, Path(path), boards)
            report["networks"][name] = {
                "path": str(Path(path).resolve()),
                "vs_lc0": comparison_stats(lc0, predicted, outcome),
                "vs_stockfish": comparison_stats(sf, predicted, outcome),
            }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
