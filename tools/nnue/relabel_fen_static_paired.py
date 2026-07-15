#!/usr/bin/env python3
"""Resumably attach semantics-explicit static evaluator outputs to FEN text.

This is deliberately separate from search labeling.  The pinned teacher builds
must expose the concise ``evalraw`` command; no node-limited search score can be
silently substituted for a static value label. Stockfish 18 is the production
bootstrap teacher; Reckless is an optional sparse audit control.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import chess
import numpy as np


SCHEMA = "archon.paired-fen-static-values.v2"
NO_VALUE = np.int32(-(2**31))
TEACHER_FIELDS = {
    "reckless": (
        "reckless_raw_nnue_units",
        "reckless_static_internal",
        "reckless_rule50",
    ),
    "stockfish18": (
        "stockfish18_final_internal",
        "stockfish18_final_uci_cp",
        "stockfish18_big_psqt_internal",
        "stockfish18_big_positional_internal",
        "stockfish18_big_sum_internal",
        "stockfish18_big_sum_uci_cp",
        "stockfish18_rule50",
    ),
}


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


def fen_hash64(fen: str) -> int:
    return int.from_bytes(hashlib.blake2b(fen.encode(), digest_size=8).digest(), "little")


def parse_source_line(line: str) -> tuple[str, int, float]:
    parts = [part.strip() for part in line.rstrip("\n").split("|")]
    if len(parts) != 3:
        raise ValueError("expected '<FEN> | <score> | <result>'")
    board = chess.Board(parts[0])
    result = float(parts[2])
    if result not in (0.0, 0.5, 1.0) or board.is_game_over(claim_draw=False):
        raise ValueError("invalid result or terminal position")
    return parts[0], int(parts[1]), result


class RawEvalEngine:
    def __init__(self, path: str):
        self.path = path
        self.process = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.send("uci")
        self.read_until("uciok")
        self.send("isready")
        self.read_until("readyok")

    def send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("teacher stdin unavailable")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, prefix: str) -> str:
        if self.process.stdout is None:
            raise RuntimeError("teacher stdout unavailable")
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = "" if self.process.stderr is None else self.process.stderr.read()
                raise RuntimeError(f"teacher exited before {prefix!r}: {stderr[-2000:]}")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def evaluate(self, fen: str) -> dict[str, int] | None:
        self.send(f"position fen {fen}")
        self.send("evalraw")
        fields = self.read_until("evalraw").split()
        parsed: dict[str, str] = {}
        for field in fields[1:]:
            if "=" not in field:
                raise RuntimeError(f"invalid evalraw field: {field!r}")
            key, value = field.split("=", 1)
            parsed[key] = value
        if parsed.get("status") == "invalid_check":
            return None
        if parsed.pop("status", None) != "ok" or not parsed:
            raise RuntimeError(f"invalid evalraw response: {fields}")
        try:
            return {key: int(value) for key, value in parsed.items()}
        except ValueError as error:
            raise RuntimeError(f"non-integer evalraw response: {fields}") from error

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
        self.process.wait(timeout=10)


def engine_identity(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [str(path)], input="uci\nquit\n", text=True, capture_output=True, check=True,
    )
    name = author = None
    for line in process.stdout.splitlines():
        if line.startswith("id name "):
            name = line[8:]
        elif line.startswith("id author "):
            author = line[10:]
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "uci_name": name, "uci_author": author, "command": "evalraw key=value protocol v2",
    }


def label_partition(
    fens: list[str], reckless: str | None, stockfish: str,
) -> dict[str, np.ndarray]:
    engines = {"stockfish18": RawEvalEngine(stockfish)}
    if reckless is not None:
        engines["reckless"] = RawEvalEngine(reckless)
    values = {
        field: np.full(len(fens), NO_VALUE, dtype=np.int32)
        for teacher in engines for field in TEACHER_FIELDS[teacher]
    }
    try:
        for row, fen in enumerate(fens):
            row_results = {teacher: engine.evaluate(fen) for teacher, engine in engines.items()}
            # A row is either valid for both teachers or excluded for both.  In
            # particular, Stockfish deliberately has no static eval in check.
            if any(result is None for result in row_results.values()):
                continue
            for teacher, result in row_results.items():
                assert result is not None
                expected = {
                    field.removeprefix(f"{teacher}_") for field in TEACHER_FIELDS[teacher]
                }
                if set(result) != expected:
                    raise RuntimeError(
                        f"{teacher} evalraw keys differ: expected={sorted(expected)} got={sorted(result)}"
                    )
                for field in TEACHER_FIELDS[teacher]:
                    values[field][row] = result[field.removeprefix(f"{teacher}_")]
    finally:
        for engine in engines.values():
            engine.close()
    return values


def quality(values: dict[str, np.ndarray], teacher: str) -> dict[str, int]:
    arrays = [values[field] for field in TEACHER_FIELDS[teacher]]
    valid = np.logical_and.reduce([array != NO_VALUE for array in arrays])
    if any(np.any((array != NO_VALUE) != valid) for array in arrays):
        raise RuntimeError(f"partial {teacher} static-label row")
    return {
        "valid_rows": int(np.count_nonzero(valid)),
        "excluded_in_check_rows": int(np.count_nonzero(~valid)),
    }


def validate_chunk(path: Path, expected: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        stored = json.loads(data["metadata_json"].tobytes().decode())
        if stored != expected:
            raise RuntimeError(f"metadata mismatch: {path}")
        rows = expected["rows"]
        active_teachers = expected["active_teachers"]
        active_fields = [field for teacher in active_teachers for field in TEACHER_FIELDS[teacher]]
        for name in ("source_line", "fen_hash64", "sf16_score_stm", "result_stm", *active_fields):
            if name not in data.files or len(data[name]) != rows:
                raise RuntimeError(f"missing or mis-sized {name}: {path}")
        values = {field: data[field] for field in active_fields}
        for teacher in active_teachers:
            if quality(values, teacher) != expected["label_quality"][teacher]:
                raise RuntimeError(f"quality metadata mismatch: {path}:{teacher}")
        valid = values["stockfish18_rule50"] != NO_VALUE
        if np.any(values["stockfish18_big_sum_internal"][valid] !=
                  values["stockfish18_big_psqt_internal"][valid] +
                  values["stockfish18_big_positional_internal"][valid]):
            raise RuntimeError(f"Stockfish big-net decomposition mismatch: {path}")
        if "reckless" in active_teachers and np.any(
            values["stockfish18_rule50"] != values["reckless_rule50"]
        ):
            raise RuntimeError(f"teacher rule-50 mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage-meta", type=Path, required=True)
    parser.add_argument("--reckless", type=Path,
                        help="optional sparse audit teacher; omitted for production SF18 bootstrap")
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if min(value for value in (args.workers, args.chunk_rows, args.max_rows or 1)) <= 0:
        parser.error("workers, chunk-rows, and max-rows must be positive")
    for path in [*args.sources, args.lineage_meta, args.stockfish, *([args.reckless] if args.reckless else [])]:
        if not path.is_file():
            parser.error(f"missing input: {path}")

    config = {
        "schema": SCHEMA,
        "label_kind": "explicit cold static-evaluator components; never a search score",
        "score_perspective": "side_to_move",
        "invalid_policy": "positions in check receive NO_VALUE and are excluded by consumers",
        "state_completeness": "full FEN except repetition history",
        "field_semantics": {
            "reckless_raw_nnue_units": "Reckless NNUE output-transformer internal units",
            "reckless_static_internal": "raw NNUE after deterministic material/rule50 scaling; zero optimism and correction history",
            "stockfish18_final_internal": "Stockfish Eval::evaluate internal Value with optimism=0",
            "stockfish18_final_uci_cp": "final_internal converted by Stockfish position-dependent UCI win-rate model",
            "stockfish18_big_psqt_internal": "Stockfish big-network PSQT component in internal Value units",
            "stockfish18_big_positional_internal": "Stockfish big-network positional component in internal Value units",
            "stockfish18_big_sum_internal": "unscaled sum of big-network PSQT and positional components",
            "stockfish18_big_sum_uci_cp": "big_sum_internal converted by Stockfish position-dependent UCI win-rate model",
            "*_rule50": "teacher-parsed FEN halfmove clock; cross-teacher equality is enforced",
        },
        "lineage": {
            "path": str(args.lineage_meta.resolve()), "sha256": sha256(args.lineage_meta),
        },
        "active_teachers": ["stockfish18"] + (["reckless"] if args.reckless else []),
        "teachers": {"stockfish18": engine_identity(args.stockfish.resolve())},
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in args.sources
        ],
    }
    if args.reckless:
        config["teachers"]["reckless"] = engine_identity(args.reckless.resolve())
    config_hash = canonical_hash(config)
    manifest: dict[str, Any] = {**config, "config_hash": config_hash, "chunks": []}
    args.output.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        stop = False
        for source_number, source in enumerate(args.sources):
            source_detail = config["sources"][source_number]
            with source.open("r", encoding="utf-8") as handle:
                line_number = chunk_number = 0
                while not stop:
                    rows: list[tuple[str, int, float, int]] = []
                    while len(rows) < args.chunk_rows:
                        line = handle.readline()
                        if not line:
                            break
                        fen, score_white, result_white = parse_source_line(line)
                        board = chess.Board(fen)
                        rows.append((
                            fen,
                            score_white if board.turn == chess.WHITE else -score_white,
                            result_white if board.turn == chess.WHITE else 1.0 - result_white,
                            fen_hash64(fen),
                        ))
                        line_number += 1
                        if args.max_rows is not None and total_rows + len(rows) >= args.max_rows:
                            stop = True
                            break
                    if not rows:
                        break

                    relative = Path(source.stem) / f"chunk-{chunk_number:06d}.npz"
                    output = args.output / relative
                    metadata: dict[str, Any] = {
                        "schema": SCHEMA, "config_hash": config_hash,
                        "source": source_detail, "source_number": source_number,
                        "source_start_line": line_number - len(rows), "source_stop_line": line_number,
                        "rows": len(rows), "chunk": chunk_number,
                        "score_perspective": "side_to_move",
                        "active_teachers": config["active_teachers"],
                    }
                    if output.exists():
                        with np.load(output, allow_pickle=False) as existing:
                            stored = json.loads(existing["metadata_json"].tobytes().decode())
                        metadata["label_quality"] = stored["label_quality"]
                        validate_chunk(output, metadata)
                        status = "validated_existing"
                    else:
                        fens = np.asarray([row[0] for row in rows], dtype=object)
                        partitions = [part for part in np.array_split(fens, min(args.workers, len(fens))) if len(part)]
                        began = time.monotonic()
                        futures = [executor.submit(
                            label_partition, part.tolist(),
                            None if args.reckless is None else str(args.reckless.resolve()),
                            str(args.stockfish.resolve()),
                        ) for part in partitions]
                        parts = [future.result() for future in futures]
                        values = {
                            field: np.concatenate([part[field] for part in parts])
                            for teacher in config["active_teachers"] for field in TEACHER_FIELDS[teacher]
                        }
                        metadata["label_quality"] = {
                            teacher: quality(values, teacher) for teacher in config["active_teachers"]
                        }
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as f:
                            staged = Path(f.name)
                        try:
                            with staged.open("wb") as destination:
                                np.savez_compressed(
                                    destination,
                                    source_line=np.arange(metadata["source_start_line"], metadata["source_stop_line"], dtype=np.uint32),
                                    fen_hash64=np.asarray([row[3] for row in rows], dtype=np.uint64),
                                    sf16_score_stm=np.asarray([row[1] for row in rows], dtype=np.int16),
                                    result_stm=np.asarray([round(row[2] * 2) for row in rows], dtype=np.uint8),
                                    **values,
                                    metadata_json=np.frombuffer(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8),
                                )
                            validate_chunk(staged, metadata)
                            os.replace(staged, output)
                        finally:
                            staged.unlink(missing_ok=True)
                        seconds = time.monotonic() - began
                        status = "written"
                        print(json.dumps({
                            "chunk": str(relative), "rows": len(rows), "seconds": seconds,
                            "paired_rows_per_second": len(rows) / seconds,
                            "label_quality": metadata["label_quality"], "status": status,
                        }, sort_keys=True), flush=True)
                    manifest["chunks"].append({
                        "path": str(relative), "sha256": sha256(output), **metadata,
                    })
                    total_rows += len(rows)
                    chunk_number += 1

    manifest["total_rows"] = total_rows
    manifest_path = args.output / "manifest.json"
    with tempfile.NamedTemporaryFile("w", dir=args.output, prefix=".manifest.", delete=False) as f:
        staged_manifest = Path(f.name)
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(staged_manifest, manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "rows": total_rows,
                      "chunks": len(manifest["chunks"]), "config_hash": config_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
