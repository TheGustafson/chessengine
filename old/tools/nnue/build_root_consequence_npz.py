#!/usr/bin/env python3
"""Build a leakage-safe root action-consequence tensor from oracle shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

import nnue_ref


def group_is_validation(source_id: str, per_mille: int) -> bool:
    value = int.from_bytes(hashlib.blake2b(source_id.encode(), digest_size=8).digest(), "little")
    return value % 1000 < per_mille


def move_features(board: chess.Board, move: chess.Move, legal_count: int) -> np.ndarray:
    out = np.zeros(128 + 6 + 7 + 5 + 5, dtype=np.float32)
    from_square = move.from_square if board.turn == chess.WHITE else move.from_square ^ 56
    to_square = move.to_square if board.turn == chess.WHITE else move.to_square ^ 56
    out[from_square] = 1.0
    out[64 + to_square] = 1.0
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise RuntimeError(f"move has no source piece: {move} {board.fen()}")
    out[128 + piece.piece_type - 1] = 1.0
    captured = board.piece_at(move.to_square)
    captured_type = 0 if captured is None else captured.piece_type
    out[134 + captured_type] = 1.0
    promotion = 0 if move.promotion is None else move.promotion
    promotion_slot = {0: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[promotion]
    out[141 + promotion_slot] = 1.0
    tail = 146
    out[tail] = float(board.is_capture(move))
    out[tail + 1] = float(board.gives_check(move))
    out[tail + 2] = legal_count / 64.0
    out[tail + 3] = from_square / 63.0
    out[tail + 4] = to_square / 63.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-cp", type=int, default=50)
    parser.add_argument("--val-per-mille", type=int, default=200)
    args = parser.parse_args()
    if args.error_cp <= 0 or not 0 < args.val_per_mille < 1000:
        parser.error("error-cp must be positive and val-per-mille in 1..999")

    manifest = json.loads((args.oracle / "manifest.json").read_text())
    records = []
    for path in sorted((args.oracle / "shards").glob("*.json")):
        shard = json.loads(path.read_text())
        if shard["config_hash"] != manifest["config_hash"]:
            raise RuntimeError(f"config mismatch: {path}")
        records.extend(shard["records"])
    if len(records) != manifest["total_positions"]:
        raise RuntimeError("incomplete oracle coverage")

    net = nnue_ref.load_net(str(args.eval))
    low_limit = manifest["config"]["search"]["low"]
    high_limit = manifest["config"]["search"]["high"]
    tie_cp = manifest["config"]["search"]["tie_cp"]
    features = []
    context_only = []
    labels = {key: [] for key in ("error", "best", "score_delta_cp", "log_low_nodes", "log_high_nodes",
                                  "log_low_time_ms", "log_high_time_ms", "high_score_cp")}
    groups = []
    validation = []
    moves = []
    for group, record in enumerate(records):
        board = chess.Board(record["fen"])
        pieces, stm = nnue_ref.parse_placement(record["fen"])
        acc_white = np.asarray(nnue_ref.refresh(net, pieces, nnue_ref.WHITE), dtype=np.float32)
        acc_black = np.asarray(nnue_ref.refresh(net, pieces, nnue_ref.BLACK), dtype=np.float32)
        stm_acc, ntm_acc = (acc_white, acc_black) if stm == nnue_ref.WHITE else (acc_black, acc_white)
        latent = np.concatenate([np.clip(stm_acc, 0, 255), np.clip(ntm_acc, 0, 255)]) / 255.0
        high_scores = [arm["high"]["score_cp"] for arm in record["arms"]]
        high_best = max(high_scores)
        static_eval = nnue_ref.forward(net, stm_acc.astype(int).tolist(), ntm_acc.astype(int).tolist())
        for arm in record["arms"]:
            move = chess.Move.from_uci(arm["move"])
            move_ctx = move_features(board, move, len(record["arms"]))
            numeric = np.asarray(
                [static_eval / 1000.0, low_limit / 16.0, high_limit / 16.0],
                dtype=np.float32,
            )
            context = np.concatenate([move_ctx, numeric])
            delta = arm["low"]["score_cp"] - arm["high"]["score_cp"]
            features.append(np.concatenate([latent, context]))
            context_only.append(context)
            labels["error"].append(abs(delta) > args.error_cp)
            labels["best"].append(arm["high"]["score_cp"] >= high_best - tie_cp)
            labels["score_delta_cp"].append(delta)
            labels["log_low_nodes"].append(np.log1p(arm["low"]["nodes"]))
            labels["log_high_nodes"].append(np.log1p(arm["high"]["nodes"]))
            labels["log_low_time_ms"].append(np.log1p(arm["low"].get("time_ms", 0)))
            labels["log_high_time_ms"].append(np.log1p(arm["high"].get("time_ms", 0)))
            labels["high_score_cp"].append(arm["high"]["score_cp"])
            groups.append(group)
            validation.append(group_is_validation(record["source_id"], args.val_per_mille))
            moves.append(arm["move"])

    metadata = {
        "schema": "archon.root-action-consequence.v1",
        "oracle": str(args.oracle.resolve()),
        "oracle_config_hash": manifest["config_hash"],
        "eval": str(args.eval.resolve()),
        "eval_sha256": hashlib.sha256(args.eval.read_bytes()).hexdigest(),
        "rows": len(features),
        "positions": len(records),
        "error_definition": f"abs(low_score - high_score) > {args.error_cp} cp",
        "best_definition": f"high_score within {tie_cp} cp of position maximum",
        "split": f"validation iff blake2b64(source_id) % 1000 < {args.val_per_mille}",
        "feature_contract": "protected classic [stm;ntm] clipped accumulator /255 + pre-action move/context",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
        features=np.asarray(features, dtype=np.float32),
        context_only=np.asarray(context_only, dtype=np.float32),
        group=np.asarray(groups, dtype=np.uint16),
        is_validation=np.asarray(validation, dtype=np.bool_),
        move=np.asarray(moves, dtype="S5"),
        **{key: np.asarray(value) for key, value in labels.items()},
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
