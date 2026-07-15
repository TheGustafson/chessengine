#!/usr/bin/env python3
"""Resumably dual-label bulletformat FEN text with Reckless and Stockfish."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.cloud import dual_label_shard as paired


SCHEMA = "archon.paired-fen-value-labels.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_source_line(line: str) -> tuple[str, int, float]:
    parts = line.rstrip("\n").split("|")
    if len(parts) != 3:
        raise ValueError("expected '<FEN> | <score> | <result>'")
    fen, score, result = (part.strip() for part in parts)
    board = chess.Board(fen)
    score_white = int(score)
    result_white = float(result)
    if result_white not in (0.0, 0.5, 1.0):
        raise ValueError(f"invalid result {result_white}")
    if board.is_game_over(claim_draw=False):
        raise ValueError("source contains a terminal position")
    return fen, score_white, result_white


def label_fens(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if paired._ENGINES is None:
        raise RuntimeError("worker engines are not initialized")
    results: list[list[tuple[int, ...]]] = [[], []]
    for fen in fens:
        board = chess.Board(fen)
        for teacher, engine in enumerate(paired._ENGINES):
            results[teacher].append(paired.analyse(engine, board))
    return (
        np.asarray(results[0], dtype=np.int64),
        np.asarray(results[1], dtype=np.int64),
    )


def label_fen_partition(
    fens: list[str], reckless: str, stockfish: str,
    nodes: int | None, depth: int | None, hash_mb: int,
) -> tuple[np.ndarray, np.ndarray]:
    paired.initialize_worker(reckless, stockfish, nodes, depth, hash_mb)
    try:
        return label_fens(fens)
    finally:
        paired.close_engines()


def validate_output(path: Path, expected: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata_json"].tobytes().decode())
        if metadata != expected:
            raise RuntimeError(f"metadata mismatch in {path}")
        rows = int(metadata["rows"])
        required = {
            "source_line", "fen_hash64", "sf16_score_stm", "result_stm",
            *paired.teacher_arrays("reckless", np.zeros((1, 10), dtype=np.int64)),
            *paired.teacher_arrays("stockfish", np.zeros((1, 10), dtype=np.int64)),
        }
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"missing arrays in {path}: {sorted(missing)}")
        for name in required:
            if len(data[name]) != rows:
                raise RuntimeError(f"row mismatch in {path}:{name}")
        for teacher in ("reckless", "stockfish"):
            if np.any(data[f"{teacher}_nodes"] == 0):
                raise RuntimeError(f"zero searched nodes in {path}:{teacher}")
            bounds = data[f"{teacher}_bound"]
            quality = {
                "upper_bound_rows": int(np.count_nonzero(bounds == -1)),
                "exact_rows": int(np.count_nonzero(bounds == 0)),
                "lower_bound_rows": int(np.count_nonzero(bounds == 1)),
                "wdl_available_rows": int(np.count_nonzero(
                    np.any(data[f"{teacher}_wdl"] != 0xFFFF, axis=1)
                )),
            }
            if metadata.get("label_quality", {}).get(teacher) != quality:
                raise RuntimeError(f"label-quality metadata mismatch in {path}:{teacher}")
            if metadata.get("search_limit", {}).get("kind") == "depth" and np.any(bounds):
                raise RuntimeError(f"completed-depth search returned a bound in {path}:{teacher}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage-meta", type=Path, required=True)
    parser.add_argument("--reckless", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    limit = parser.add_mutually_exclusive_group(required=True)
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if min(value for value in (args.nodes, args.depth, args.workers, args.hash_mb, args.chunk_rows)
           if value is not None) <= 0:
        parser.error("search limit, workers, hash-mb, and chunk-rows must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    for path in (*args.sources, args.lineage_meta, args.reckless, args.stockfish):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    identities = {
        "reckless": paired.engine_identity(args.reckless.resolve()),
        "stockfish": paired.engine_identity(args.stockfish.resolve()),
    }
    config = {
        "schema": SCHEMA,
        "search_limit": {
            "kind": "nodes" if args.nodes is not None else "depth",
            "value": args.nodes if args.nodes is not None else args.depth,
            "completion_contract": (
                "may return lower/upper bounds; consumer must use bound-aware loss"
                if args.nodes is not None
                else "completed iteration; any returned bound fails chunk validation"
            ),
        },
        "hash_mb_per_engine": args.hash_mb,
        "cold_position_state": "ucinewgame per position and teacher",
        "score_perspective": "side_to_move",
        "state_completeness": "fen_state_no_repetition_history",
        "result_encoding": "side-to-move outcome: loss=0, draw=1, win=2",
        "teachers": identities,
        "source_lineage": {
            "path": str(args.lineage_meta.resolve()),
            "sha256": sha256(args.lineage_meta),
            "metadata": json.loads(args.lineage_meta.read_text(encoding="utf-8")),
        },
        "sources": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in args.sources
        ],
        "chunk_rows": args.chunk_rows,
        "max_rows": args.max_rows,
    }
    config_hash = canonical_hash(config)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = {**config, "config_hash": config_hash, "chunks": []}
    total_rows = 0
    started_all = time.monotonic()

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
    ) as executor:
        stop = False
        for source_number, source in enumerate(args.sources):
            source_detail = config["sources"][source_number]
            with source.open("r", encoding="utf-8") as handle:
                line_number = 0
                chunk_number = 0
                while not stop:
                    rows: list[tuple[str, int, float, int]] = []
                    while len(rows) < args.chunk_rows:
                        line = handle.readline()
                        if not line:
                            break
                        try:
                            fen, score_white, result_white = parse_source_line(line)
                        except Exception as error:
                            raise RuntimeError(f"{source}:{line_number + 1}: {error}") from error
                        board = chess.Board(fen)
                        score_stm = score_white if board.turn == chess.WHITE else -score_white
                        result_stm = result_white if board.turn == chess.WHITE else 1.0 - result_white
                        fen_hash = int.from_bytes(
                            hashlib.blake2b(fen.encode(), digest_size=8).digest(), "little"
                        )
                        line_number += 1
                        rows.append((fen, score_stm, result_stm, fen_hash))
                        if args.max_rows is not None and total_rows + len(rows) >= args.max_rows:
                            stop = True
                            break
                    if not rows:
                        break

                    relative = Path(source.stem) / f"chunk-{chunk_number:06d}.npz"
                    output = args.output / relative
                    metadata = {
                        "schema": SCHEMA,
                        "config_hash": config_hash,
                        "source": source_detail,
                        "source_number": source_number,
                        "source_start_line": line_number - len(rows),
                        "source_stop_line": line_number,
                        "rows": len(rows),
                        "chunk": chunk_number,
                        "score_perspective": "side_to_move",
                        "state_completeness": "fen_state_no_repetition_history",
                        "result_encoding": "side-to-move outcome: loss=0, draw=1, win=2",
                        "search_limit": config["search_limit"],
                    }
                    if output.exists():
                        with np.load(output, allow_pickle=False) as existing:
                            stored = json.loads(existing["metadata_json"].tobytes().decode())
                        if "label_quality" in stored:
                            metadata["label_quality"] = stored["label_quality"]
                        validate_output(output, metadata)
                        status = "validated_existing"
                    else:
                        partitions = [
                            part for part in np.array_split(np.asarray([r[0] for r in rows], dtype=object),
                                                           min(args.workers, len(rows)))
                            if len(part)
                        ]
                        began = time.monotonic()
                        futures = [
                            executor.submit(
                                label_fen_partition,
                                part.tolist(),
                                str(args.reckless.resolve()),
                                str(args.stockfish.resolve()),
                                args.nodes,
                                args.depth,
                                args.hash_mb,
                            )
                            for part in partitions
                        ]
                        values = [future.result() for future in futures]
                        reckless_values = np.concatenate([value[0] for value in values])
                        stockfish_values = np.concatenate([value[1] for value in values])
                        metadata["label_quality"] = {
                            "reckless": paired.quality_summary(reckless_values),
                            "stockfish": paired.quality_summary(stockfish_values),
                        }
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
                        ) as temporary:
                            temporary_path = Path(temporary.name)
                        try:
                            with temporary_path.open("wb") as destination:
                                np.savez_compressed(
                                    destination,
                                    source_line=np.arange(
                                        metadata["source_start_line"], metadata["source_stop_line"],
                                        dtype=np.uint32,
                                    ),
                                    fen_hash64=np.asarray([r[3] for r in rows], dtype=np.uint64),
                                    sf16_score_stm=np.asarray([r[1] for r in rows], dtype=np.int16),
                                    result_stm=np.asarray([round(r[2] * 2) for r in rows], dtype=np.uint8),
                                    metadata_json=np.frombuffer(
                                        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
                                        dtype=np.uint8,
                                    ),
                                    **paired.teacher_arrays("reckless", reckless_values),
                                    **paired.teacher_arrays("stockfish", stockfish_values),
                                )
                            validate_output(temporary_path, metadata)
                            os.replace(temporary_path, output)
                        finally:
                            temporary_path.unlink(missing_ok=True)
                        seconds = time.monotonic() - began
                        status = "written"
                        print(json.dumps({
                            "chunk": str(relative), "rows": len(rows), "seconds": seconds,
                            "paired_rows_per_second": len(rows) / seconds, "status": status,
                        }, sort_keys=True), flush=True)
                    manifest["chunks"].append({
                        "path": str(relative), "sha256": sha256(output), **metadata,
                    })
                    total_rows += len(rows)
                    chunk_number += 1
                    manifest.update({
                        "total_rows": total_rows,
                        "total_chunks": len(manifest["chunks"]),
                        "elapsed_seconds": time.monotonic() - started_all,
                    })
                    temporary_manifest = manifest_path.with_suffix(".json.tmp")
                    temporary_manifest.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    os.replace(temporary_manifest, manifest_path)
                    if not line and not rows:
                        break
                if stop:
                    break
    print(json.dumps({
        "manifest": str(manifest_path), "rows": total_rows,
        "chunks": len(manifest["chunks"]), "config_hash": config_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
