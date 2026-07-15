#!/usr/bin/env python3
"""Attach paired Reckless and Stockfish labels to one immutable position shard."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import chess
import chess.engine
from google.cloud import storage
import numpy as np


SCHEMA = "archon.paired-value-labels.v1"
MATE_SCORE = 30_000
NO_MATE = 32_767
NO_MOVE = 0xFFFF
_ENGINES: tuple[chess.engine.SimpleEngine, chess.engine.SimpleEngine] | None = None
_LIMIT: chess.engine.Limit | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(uri)
    bucket, _, name = uri[5:].partition("/")
    if not bucket or not name:
        raise ValueError(uri)
    return bucket, name


def download(client: storage.Client, uri: str, destination: Path) -> storage.Blob:
    bucket_name, name = parse_gs(uri)
    blob = client.bucket(bucket_name).blob(name)
    blob.reload()
    blob.download_to_filename(destination, checksum="crc32c")
    return blob


def read_json(client: storage.Client, uri: str) -> dict[str, Any]:
    bucket_name, name = parse_gs(uri)
    return json.loads(client.bucket(bucket_name).blob(name).download_as_text())


def engine_identity(path: Path) -> dict[str, Any]:
    engine = chess.engine.SimpleEngine.popen_uci(str(path), timeout=30)
    try:
        return {
            "path": str(path),
            "sha256": sha256(path),
            "uci_name": engine.id.get("name"),
            "uci_author": engine.id.get("author"),
            "uci_options": sorted(engine.options),
        }
    finally:
        engine.quit()


def board_from_features(row: np.ndarray) -> chess.Board:
    board = chess.Board.empty()
    for raw in row:
        feature = int(raw)
        if feature < 0:
            break
        colour, remainder = divmod(feature, 384)
        piece_type, square = divmod(remainder, 64)
        if colour not in (0, 1) or piece_type >= 6:
            raise ValueError(f"invalid feature: {feature}")
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
        return NO_MOVE
    promotion = 0 if move.promotion is None else move.promotion
    return move.from_square | (move.to_square << 6) | (promotion << 12)


def close_engines() -> None:
    global _ENGINES, _LIMIT
    if _ENGINES is None:
        _LIMIT = None
        return
    for engine in _ENGINES:
        try:
            engine.quit()
        except (chess.engine.EngineError, OSError):
            pass
    _ENGINES = None
    _LIMIT = None


def initialize_worker(
    reckless: str,
    stockfish: str,
    nodes: int | None,
    depth: int | None,
    hash_mb: int,
) -> None:
    global _ENGINES, _LIMIT
    if (nodes is None) == (depth is None):
        raise ValueError("set exactly one of nodes or depth")
    first = chess.engine.SimpleEngine.popen_uci(reckless, timeout=30)
    second = chess.engine.SimpleEngine.popen_uci(stockfish, timeout=30)
    for engine in (first, second):
        settings: dict[str, Any] = {}
        if "Threads" in engine.options:
            settings["Threads"] = 1
        if "Hash" in engine.options:
            settings["Hash"] = hash_mb
        if "UCI_ShowWDL" in engine.options:
            settings["UCI_ShowWDL"] = True
        if settings:
            engine.configure(settings)
    _ENGINES = first, second
    _LIMIT = chess.engine.Limit(nodes=nodes) if nodes is not None else chess.engine.Limit(depth=depth)
    atexit.register(close_engines)


def analyse(engine: chess.engine.SimpleEngine, board: chess.Board) -> tuple[int, ...]:
    if _LIMIT is None:
        raise RuntimeError("worker search limit is not initialized")
    info = engine.analyse(
        board,
        _LIMIT,
        game=object(),
        info=chess.engine.INFO_ALL,
    )
    pov = info["score"].pov(board.turn)
    mate = pov.mate()
    score = pov.score(mate_score=MATE_SCORE)
    if score is None:
        raise RuntimeError("teacher returned no score")
    pv = info.get("pv", [])
    move = pv[0] if pv else None
    if move is not None and move not in board.legal_moves:
        raise RuntimeError(f"teacher returned illegal move {move} for {board.fen()}")
    bound = 1 if info.get("lowerbound") else (-1 if info.get("upperbound") else 0)
    wins = draws = losses = 0xFFFF
    if "wdl" in info:
        wdl = info["wdl"].pov(board.turn)
        wins, draws, losses = int(wdl.wins), int(wdl.draws), int(wdl.losses)
    return (
        int(np.clip(score, -MATE_SCORE, MATE_SCORE)),
        NO_MATE if mate is None else int(np.clip(mate, -32_000, 32_000)),
        encode_move(move),
        min(int(info.get("depth", 0)), 255),
        min(int(info.get("seldepth", 0)), 255),
        min(int(info.get("nodes", 0)), 0xFFFFFFFF),
        bound,
        wins,
        draws,
        losses,
    )


def label_batch(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _ENGINES is None:
        raise RuntimeError("worker engines are not initialized")
    result = [[], []]
    for row in features:
        board = board_from_features(row)
        for teacher, engine in enumerate(_ENGINES):
            result[teacher].append(analyse(engine, board))
    return np.asarray(result[0], dtype=np.int64), np.asarray(result[1], dtype=np.int64)


def label_partition(
    features: np.ndarray,
    reckless: str,
    stockfish: str,
    nodes: int | None,
    depth: int | None,
    hash_mb: int,
) -> tuple[np.ndarray, np.ndarray]:
    initialize_worker(reckless, stockfish, nodes, depth, hash_mb)
    try:
        return label_batch(features)
    finally:
        close_engines()


def teacher_arrays(prefix: str, values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_score_cp": values[:, 0].astype(np.int16),
        f"{prefix}_mate_plies": values[:, 1].astype(np.int16),
        f"{prefix}_best_move": values[:, 2].astype(np.uint16),
        f"{prefix}_depth": values[:, 3].astype(np.uint8),
        f"{prefix}_seldepth": values[:, 4].astype(np.uint8),
        f"{prefix}_nodes": values[:, 5].astype(np.uint32),
        f"{prefix}_bound": values[:, 6].astype(np.int8),
        f"{prefix}_wdl": values[:, 7:10].astype(np.uint16),
    }


def quality_summary(values: np.ndarray) -> dict[str, int]:
    bounds = values[:, 6]
    wdl = values[:, 7:10]
    return {
        "upper_bound_rows": int(np.count_nonzero(bounds == -1)),
        "exact_rows": int(np.count_nonzero(bounds == 0)),
        "lower_bound_rows": int(np.count_nonzero(bounds == 1)),
        "wdl_available_rows": int(np.count_nonzero(np.any(wdl != 0xFFFF, axis=1))),
    }


def validate_output(path: Path, rows: int, metadata: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as output:
        required = {
            "source_index", "metadata_json",
            *teacher_arrays("reckless", np.zeros((1, 10), dtype=np.int64)).keys(),
            *teacher_arrays("stockfish", np.zeros((1, 10), dtype=np.int64)).keys(),
        }
        missing = required.difference(output.files)
        if missing:
            raise RuntimeError(f"output missing arrays: {sorted(missing)}")
        for name in required.difference({"metadata_json"}):
            if len(output[name]) != rows:
                raise RuntimeError(f"output row mismatch in {name}")
        stored = json.loads(output["metadata_json"].tobytes().decode())
        if stored != metadata:
            raise RuntimeError("output metadata mismatch")
        for prefix in ("reckless", "stockfish"):
            moves = output[f"{prefix}_best_move"]
            if np.any((moves != NO_MOVE) & (moves > 0x5FFF)):
                raise RuntimeError(f"invalid encoded move in {prefix}")
            if np.any(output[f"{prefix}_nodes"] == 0):
                raise RuntimeError(f"zero searched nodes in {prefix}")
            bounds = output[f"{prefix}_bound"]
            if np.any((bounds < -1) | (bounds > 1)):
                raise RuntimeError(f"invalid bound code in {prefix}")
            expected_quality = metadata.get("label_quality", {}).get(prefix)
            if expected_quality is not None:
                values = np.column_stack([
                    output[f"{prefix}_score_cp"],
                    output[f"{prefix}_mate_plies"],
                    output[f"{prefix}_best_move"],
                    output[f"{prefix}_depth"],
                    output[f"{prefix}_seldepth"],
                    output[f"{prefix}_nodes"],
                    bounds,
                    output[f"{prefix}_wdl"],
                ])
                if quality_summary(values) != expected_quality:
                    raise RuntimeError(f"label-quality metadata mismatch in {prefix}")
            if metadata.get("search_limit", {}).get("kind") == "depth" and np.any(bounds):
                raise RuntimeError(f"completed-depth search returned a bound in {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--manifest")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reckless", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    limit = parser.add_mutually_exclusive_group(required=True)
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=16)
    args = parser.parse_args()
    if min(value for value in (args.nodes, args.depth, args.workers, args.hash_mb)
           if value is not None) <= 0:
        parser.error("search limit, workers, and hash-mb must be positive")
    for engine in (args.reckless, args.stockfish):
        if not engine.is_file() or not os.access(engine, os.X_OK):
            parser.error(f"engine is not executable: {engine}")

    client = storage.Client() if any(
        value and value.startswith("gs://")
        for value in (args.input, args.manifest, args.output)
    ) else None
    shard_detail: dict[str, Any] = {}
    input_uri = args.input
    if args.manifest:
        if client is None:
            raise AssertionError("GCS client missing")
        manifest = read_json(client, args.manifest)
        task_index = args.task_index
        if task_index is None:
            task_index = int(os.environ.get("BATCH_TASK_INDEX", "-1"))
        if not 0 <= task_index < len(manifest["shards"]):
            raise SystemExit(f"invalid task index {task_index}")
        shard_detail = manifest["shards"][task_index]
        input_uri = shard_detail["uri"]
    if input_uri is None:
        raise AssertionError("input URI missing")

    identities = {
        "reckless": engine_identity(args.reckless.resolve()),
        "stockfish": engine_identity(args.stockfish.resolve()),
    }
    config = {
        "schema": SCHEMA,
        "search_limit": {
            "kind": "nodes" if args.nodes is not None else "depth",
            "value": args.nodes if args.nodes is not None else args.depth,
            "completion_contract": (
                "may return lower/upper bounds; consumer must use bound-aware loss"
                if args.nodes is not None
                else "completed iteration; any returned bound fails shard validation"
            ),
        },
        "hash_mb_per_engine": args.hash_mb,
        "cold_position_state": "ucinewgame per position and teacher",
        "score_perspective": "side_to_move",
        "state_completeness": "piece_state_bootstrap_only",
        "teachers": identities,
    }
    config_hash = canonical_hash(config)

    with tempfile.TemporaryDirectory(prefix="archon-dual-label-") as temporary:
        root = Path(temporary)
        input_path = root / "input.npz"
        input_blob = None
        if input_uri.startswith("gs://"):
            if client is None:
                raise AssertionError("GCS client missing")
            input_blob = download(client, input_uri, input_path)
        else:
            input_path = Path(input_uri).resolve()
        input_hash = sha256(input_path)
        if shard_detail and input_hash != shard_detail["sha256"]:
            raise RuntimeError("input hash does not match manifest")
        if input_blob and input_blob.metadata and input_blob.metadata.get("sha256") != input_hash:
            raise RuntimeError("input hash does not match object metadata")
        with np.load(input_path, allow_pickle=False) as source_data:
            features = np.asarray(source_data["features"])
            source_index = np.asarray(source_data["source_index"])
        if features.shape != (len(source_index), 32) or features.dtype != np.int16:
            raise RuntimeError(f"invalid input arrays: {features.shape} {features.dtype}")

        started = time.monotonic()
        partitions = [part for part in np.array_split(features, min(args.workers, len(features))) if len(part)]
        with ProcessPoolExecutor(
            max_workers=len(partitions),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(
                    label_partition,
                    partition,
                    str(args.reckless.resolve()),
                    str(args.stockfish.resolve()),
                    args.nodes,
                    args.depth,
                    args.hash_mb,
                )
                for partition in partitions
            ]
            paired_parts = [future.result() for future in futures]
        elapsed = time.monotonic() - started
        reckless_values = np.concatenate([part[0] for part in paired_parts])
        stockfish_values = np.concatenate([part[1] for part in paired_parts])
        metadata = {
            **config,
            "config_hash": config_hash,
            "input_uri": input_uri,
            "input_sha256": input_hash,
            "source_shard": shard_detail,
            "rows": len(features),
            "label_quality": {
                "reckless": quality_summary(reckless_values),
                "stockfish": quality_summary(stockfish_values),
            },
        }
        arrays = {
            "source_index": source_index,
            "metadata_json": np.frombuffer(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
                dtype=np.uint8,
            ),
            **teacher_arrays("reckless", reckless_values),
            **teacher_arrays("stockfish", stockfish_values),
        }
        output_path = root / "labels.npz"
        np.savez_compressed(output_path, **arrays)
        validate_output(output_path, len(features), metadata)
        output_hash = sha256(output_path)

        if args.output.startswith("gs://"):
            if client is None:
                raise AssertionError("GCS client missing")
            bucket_name, prefix = parse_gs(args.output)
            ordinal = shard_detail.get("ordinal", args.task_index or 0)
            object_name = f"{prefix.rstrip('/')}/shards/{int(ordinal):06d}.npz"
            blob = client.bucket(bucket_name).blob(object_name)
            expected_metadata = {
                "sha256": output_hash,
                "schema": SCHEMA,
                "config_hash": config_hash,
                "input_sha256": input_hash,
                "rows": str(len(features)),
            }
            if blob.exists():
                blob.reload()
                if blob.metadata != expected_metadata or blob.size != output_path.stat().st_size:
                    raise RuntimeError(f"existing output differs: gs://{bucket_name}/{object_name}")
                status = "validated_existing"
            else:
                blob.metadata = expected_metadata
                blob.upload_from_filename(output_path, if_generation_match=0, checksum="crc32c")
                blob.reload()
                status = "uploaded"
            output_uri = f"gs://{bucket_name}/{object_name}"
        else:
            destination = Path(args.output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != output_hash:
                raise RuntimeError(f"existing local output differs: {destination}")
            if not destination.exists():
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    delete=False,
                ) as handle:
                    staged = Path(handle.name)
                try:
                    shutil.copy2(output_path, staged)
                    os.replace(staged, destination)
                finally:
                    staged.unlink(missing_ok=True)
            output_uri = str(destination)
            status = "written"

    print(json.dumps({
        "status": status,
        "output": output_uri,
        "rows": len(features),
        "seconds": elapsed,
        "paired_rows_per_second": len(features) / elapsed,
        "config_hash": config_hash,
        "sha256": output_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
