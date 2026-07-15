#!/usr/bin/env python3
"""Resumably relabel the existing normalized LC0 position arrays with a UCI engine.

Completed shards are immediately consumable and use deterministic source row
IDs. A seeded affine permutation gives broad early coverage without allocating
a corpus-sized index permutation. Re-running the same command validates and
skips completed shards.

The source feature arrays contain normalized piece placement, not full chess
state. Reconstructed boards therefore have white to move, no castling/ep,
rule50 zero, and no repetition history. The manifest marks these labels
``piece_state_bootstrap_only``: they may bootstrap a value representation, but
must never be presented as full D41 ARCHON-oracle records.

Run with an ephemeral dependency environment if needed:

    uv run --no-project --with numpy --with python-chess python \
      tools/nnue/relabel_lc0_uci.py SOURCE OUTPUT --engine ENGINE ...
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import chess
import chess.engine
import numpy as np


MATE_SCORE = 2500
SCHEMA = "archon.external-value-shards.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uci_name(engine: Path) -> str:
    result = subprocess.run(
        [str(engine)],
        input="uci\nquit\n",
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("id name "):
            return line.removeprefix("id name ")
    raise RuntimeError(f"engine did not report an id name: {engine}")


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
    board.halfmove_clock = 0
    board.fullmove_number = 1
    if not board.is_valid():
        raise ValueError(f"invalid reconstructed board: {board.fen()}")
    return board


def encode_move(move: chess.Move | None) -> int:
    if move is None:
        return 0xFFFF
    promotion = 0 if move.promotion is None else move.promotion
    return move.from_square | (move.to_square << 6) | (promotion << 12)


@dataclass(frozen=True)
class LabelJob:
    engine: str
    nodes: int
    hash_mb: int
    features: np.ndarray


def label_rows(job: LabelJob) -> dict[str, np.ndarray]:
    count = len(job.features)
    scores = np.empty(count, dtype=np.int16)
    best_moves = np.empty(count, dtype=np.uint16)
    depths = np.zeros(count, dtype=np.uint8)
    seldepths = np.zeros(count, dtype=np.uint8)
    searched_nodes = np.zeros(count, dtype=np.uint32)
    engine = chess.engine.SimpleEngine.popen_uci(job.engine, timeout=30)
    engine.configure({"Threads": 1, "Hash": job.hash_mb})
    try:
        for index, row in enumerate(job.features):
            board = board_from_features(row)
            info = engine.analyse(
                board,
                chess.engine.Limit(nodes=job.nodes),
                game=object(),
                info=chess.engine.INFO_ALL,
            )
            score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
            if score is None:
                raise RuntimeError(f"engine returned no score for {board.fen()}")
            pv = info.get("pv", [])
            scores[index] = int(np.clip(score, -MATE_SCORE, MATE_SCORE))
            best_moves[index] = encode_move(pv[0] if pv else None)
            depths[index] = min(int(info.get("depth", 0)), 255)
            seldepths[index] = min(int(info.get("seldepth", 0)), 255)
            searched_nodes[index] = min(int(info.get("nodes", 0)), 0xFFFFFFFF)
    finally:
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
    return {
        "score_cp": scores,
        "best_move": best_moves,
        "depth": depths,
        "seldepth": seldepths,
        "nodes": searched_nodes,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temp = Path(handle.name)
    try:
        np.savez(temp, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def affine_parameters(rows: int, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    multiplier = int(rng.integers(1, rows))
    while math.gcd(multiplier, rows) != 1:
        multiplier = (multiplier + 1) % rows
        if multiplier == 0:
            multiplier = 1
    offset = int(rng.integers(0, rows))
    return multiplier, offset


def source_indices(
    start: int, stop: int, rows: int, multiplier: int, offset: int
) -> np.ndarray:
    ordinal = np.arange(start, stop, dtype=np.uint64)
    return ((ordinal * multiplier + offset) % rows).astype(np.uint64)


def validate_shard(path: Path, expected_indices: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as shard:
        required = {"source_index", "score_cp", "best_move", "depth", "seldepth", "nodes"}
        missing = required.difference(shard.files)
        if missing:
            raise RuntimeError(f"{path}: missing arrays {sorted(missing)}")
        if not np.array_equal(shard["source_index"], expected_indices):
            raise RuntimeError(f"{path}: source indices do not match deterministic schedule")
        if any(len(shard[name]) != len(expected_indices) for name in required):
            raise RuntimeError(f"{path}: inconsistent row counts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard-rows", type=int, default=2_048)
    parser.add_argument("--max-rows-per-side", type=int)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    engine = args.engine.resolve()
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise SystemExit(f"engine is not executable: {engine}")
    if args.nodes <= 0 or args.workers <= 0 or args.shard_rows <= 0:
        raise SystemExit("nodes, workers, and shard-rows must be positive")

    identity = {
        "path": str(engine),
        "uci_name": uci_name(engine),
        "sha256": sha256(engine),
    }
    config: dict[str, Any] = {
        "schema": SCHEMA,
        "source": str(source),
        "teacher": args.teacher,
        "engine": identity,
        "nodes": args.nodes,
        "hash_mb_per_worker": args.hash_mb,
        "workers": args.workers,
        "shard_rows": args.shard_rows,
        "max_rows_per_side": args.max_rows_per_side,
        "seed": args.seed,
        "state_completeness": "piece_state_bootstrap_only",
        "reconstruction": {
            "side_to_move": "white (source is side-normalized)",
            "castling": "unavailable; forced none",
            "en_passant": "unavailable; forced none",
            "rule50": "unavailable; forced zero",
            "repetition": "unavailable",
        },
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("config") != config:
            raise SystemExit("existing manifest configuration differs; use a new output path")
        manifest = existing
    else:
        manifest = {"config": config, "sides": {}, "started_unix": time.time()}
        atomic_json(manifest_path, manifest)

    for side_index, side in enumerate(("white", "black")):
        features = np.load(source / side / "feats.npy", mmap_mode="r")
        available = len(features)
        target = available if args.max_rows_per_side is None else min(
            available, args.max_rows_per_side
        )
        multiplier, offset = affine_parameters(available, args.seed + side_index)
        side_dir = output / side / "shards"
        side_dir.mkdir(parents=True, exist_ok=True)
        total_shards = (target + args.shard_rows - 1) // args.shard_rows
        completed_rows = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for batch_start in range(0, total_shards, args.workers):
                pending: list[tuple[int, np.ndarray, Any]] = []
                for shard_number in range(
                    batch_start, min(batch_start + args.workers, total_shards)
                ):
                    start = shard_number * args.shard_rows
                    stop = min(start + args.shard_rows, target)
                    indices = source_indices(start, stop, available, multiplier, offset)
                    shard_path = side_dir / f"{shard_number:08d}.npz"
                    if shard_path.exists():
                        validate_shard(shard_path, indices)
                        completed_rows += len(indices)
                        continue
                    selected = np.asarray(features[indices])
                    future = executor.submit(
                        label_rows,
                        LabelJob(str(engine), args.nodes, args.hash_mb, selected),
                    )
                    pending.append((shard_number, indices, future))

                for shard_number, indices, future in pending:
                    labels = future.result()
                    shard_path = side_dir / f"{shard_number:08d}.npz"
                    atomic_npz(shard_path, source_index=indices, **labels)
                    completed_rows += len(indices)

                manifest["sides"][side] = {
                    "available_rows": available,
                    "target_rows": target,
                    "completed_rows": completed_rows,
                    "completed_shards": (completed_rows + args.shard_rows - 1)
                    // args.shard_rows,
                    "total_shards": total_shards,
                    "permutation": {"multiplier": multiplier, "offset": offset},
                    "updated_unix": time.time(),
                }
                atomic_json(manifest_path, manifest)
                print(
                    f"{side}: {completed_rows}/{target} rows "
                    f"({manifest['sides'][side]['completed_shards']}/{total_shards} shards)",
                    flush=True,
                )

    manifest["completed_unix"] = time.time()
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
