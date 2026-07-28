#!/usr/bin/env python3
"""Export paired-teacher best moves as frozen-classic policy NPY data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import chess
import numpy as np


NO_MOVE = 0xFFFF
FEATURES = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_move(value: int) -> chess.Move | None:
    if value == NO_MOVE:
        return None
    promotion = (value >> 12) & 7
    return chess.Move(value & 63, (value >> 6) & 63, promotion or None)


def canonical_square(square: int, turn: chess.Color) -> int:
    return square if turn == chess.WHITE else square ^ 56


def policy_index(move: chess.Move, turn: chess.Color) -> int:
    return canonical_square(move.from_square, turn) * 64 + canonical_square(move.to_square, turn)


def encode_position(board: chess.Board, best: chess.Move, support: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    legal = list(board.legal_moves)
    if best not in legal:
        raise RuntimeError(f"teacher best move {best.uci()} is illegal for {board.fen()}")
    legal_indices = sorted({policy_index(move, board.turn) for move in legal})
    if len(legal_indices) > support:
        return None
    best_index = policy_index(best, board.turn)
    if best_index not in legal_indices:
        raise AssertionError("best index missing from legal support")

    features = np.full(FEATURES, -1, dtype=np.int16)
    encoded = []
    for square, piece in board.piece_map().items():
        relative_colour = 0 if piece.color == board.turn else 1
        normalized_square = canonical_square(square, board.turn)
        encoded.append(relative_colour * 384 + (piece.piece_type - 1) * 64 + normalized_square)
    encoded.sort()
    if len(encoded) > FEATURES:
        raise RuntimeError("position has more than 32 pieces")
    features[: len(encoded)] = encoded

    moves = np.full(support, -1, dtype=np.int16)
    probabilities = np.zeros(support, dtype=np.float16)
    moves[: len(legal_indices)] = legal_indices
    probabilities[legal_indices.index(best_index)] = 1.0
    return features, moves, probabilities


def source_rows(source: Path, start: int, stop: int):
    with source.open(encoding="utf-8") as handle:
        for _ in range(start):
            if not handle.readline():
                raise RuntimeError(f"source ended before {start}: {source}")
        for line_number in range(start, stop):
            raw = handle.readline()
            if not raw:
                raise RuntimeError(f"source ended at {line_number}: {source}")
            yield line_number, raw.split("|", 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=("reckless", "stockfish"), required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--rows-per-side", type=int, required=True)
    parser.add_argument("--skip-per-side", type=int, default=0)
    parser.add_argument("--support", type=int, default=64)
    args = parser.parse_args()
    if min(args.chunks, args.rows_per_side, args.support) <= 0 or args.support > 218:
        parser.error("chunks/rows/support must be positive and support <= 218")
    if args.skip_per_side < 0:
        parser.error("skip-per-side must be non-negative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output must be absent or empty: {args.output}")

    source_manifest = json.loads((args.labels / "manifest.json").read_text())
    chunks = source_manifest["chunks"][: args.chunks]
    if len(chunks) != args.chunks:
        parser.error("not enough completed chunks")
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for side in ("white", "black"):
        directory = args.output / side
        directory.mkdir()
        arrays[side] = {
            "features": np.lib.format.open_memmap(directory / "feats.npy", mode="w+", dtype=np.int16,
                                                   shape=(args.rows_per_side, FEATURES)),
            "moves": np.lib.format.open_memmap(directory / "moves.npy", mode="w+", dtype=np.int16,
                                                shape=(args.rows_per_side, args.support)),
            "probabilities": np.lib.format.open_memmap(directory / "probs.npy", mode="w+", dtype=np.float16,
                                                        shape=(args.rows_per_side, args.support)),
        }
    counts = {"white": 0, "black": 0}
    eligible_seen = {"white": 0, "black": 0}
    rejected = {"no_move": 0, "support_overflow": 0}
    snapshot = []

    for entry in chunks:
        path = args.labels / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"chunk SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as values:
            metadata = json.loads(values["metadata_json"].tobytes().decode())
            source_lines = values["source_line"].copy()
            best_moves = values[f"{args.teacher}_best_move"].copy()
        source = Path(metadata["source"]["path"])
        if sha256(source) != metadata["source"]["sha256"]:
            raise RuntimeError(f"source SHA mismatch: {source}")
        start, stop = metadata["source_start_line"], metadata["source_stop_line"]
        snapshot.append({"path": entry["path"], "sha256": entry["sha256"], "rows": stop - start})
        for row, (line_number, fen) in enumerate(source_rows(source, start, stop)):
            if int(source_lines[row]) != line_number:
                raise RuntimeError(f"source identity mismatch: {path}:{row}")
            board = chess.Board(fen)
            side = "white" if board.turn == chess.WHITE else "black"
            if counts[side] >= args.rows_per_side:
                if all(value >= args.rows_per_side for value in counts.values()):
                    break
                continue
            best = decode_move(int(best_moves[row]))
            if best is None:
                rejected["no_move"] += 1
                continue
            encoded = encode_position(board, best, args.support)
            if encoded is None:
                rejected["support_overflow"] += 1
                continue
            eligible_seen[side] += 1
            if eligible_seen[side] <= args.skip_per_side:
                continue
            index = counts[side]
            arrays[side]["features"][index], arrays[side]["moves"][index], arrays[side]["probabilities"][index] = encoded
            counts[side] += 1
        if all(value >= args.rows_per_side for value in counts.values()):
            break

    if any(value != args.rows_per_side for value in counts.values()):
        raise RuntimeError(f"insufficient eligible rows: {counts}")
    for side_arrays in arrays.values():
        for array in side_arrays.values():
            array.flush()
    result = {
        "schema": "archon.native-bestmove-policy-npy.v1",
        "source": str(args.labels.resolve()),
        "source_config_hash": source_manifest["config_hash"],
        "teacher": args.teacher,
        "rows_per_side": args.rows_per_side,
        "skip_per_side": args.skip_per_side,
        "legal_support": args.support,
        "normalization": "side-to-move becomes white; black-to-move squares xor 56",
        "promotion_target": "collapsed to from-to 4096 action",
        "counts": counts,
        "rejected": rejected,
        "chunks_snapshot": snapshot,
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
