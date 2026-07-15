#!/usr/bin/env python3
"""Export row-identical Bullet text corpora from paired value-label chunks.

The exporter intentionally removes rows that cannot be represented by the
32-byte bulletformat ChessBoard score field used by Archon's proven trainer.
The same mask and deterministic FEN-hash split are applied to every teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA = "archon.matched-paired-value-corpora.v1"
TEACHERS = {
    "sf16": "sf16_score_stm",
    "reckless": "reckless_score_cp",
    "stockfish18": "stockfish_score_cp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def chunk_entries(root: Path, count: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = list(manifest.get("chunks", []))
    if count is not None:
        if len(chunks) < count:
            raise RuntimeError(f"requested {count} chunks, manifest has {len(chunks)}")
        chunks = chunks[:count]
    if not chunks:
        raise RuntimeError("manifest contains no completed chunks")
    return manifest, chunks


def parse_source(line: str) -> tuple[str, int, float]:
    parts = [part.strip() for part in line.rstrip("\n").split("|")]
    if len(parts) != 3:
        raise ValueError("expected '<FEN> | <score> | <result>'")
    result = float(parts[2])
    if result not in (0.0, 0.5, 1.0):
        raise ValueError(f"invalid result {result}")
    return parts[0], int(parts[1]), result


def fen_hash64(fen: str) -> int:
    return int.from_bytes(hashlib.blake2b(fen.encode(), digest_size=8).digest(), "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--val-per-mille", type=int, default=100)
    parser.add_argument("--score-limit", type=int, default=2500)
    args = parser.parse_args()
    if args.chunks is not None and args.chunks <= 0:
        parser.error("--chunks must be positive")
    if not 0 < args.val_per_mille < 1000:
        parser.error("--val-per-mille must be in 1..999")
    if not 0 < args.score_limit <= 32767:
        parser.error("--score-limit must be positive and at most 32767")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output must be absent or empty: {args.output}")

    manifest, entries = chunk_entries(args.labels, args.chunks)
    args.output.mkdir(parents=True, exist_ok=True)
    totals = {"input": 0, "retained": 0, "train": 0, "val": 0, "excluded_score": 0}
    exported: list[dict[str, Any]] = []

    for ordinal, entry in enumerate(entries):
        relative = Path(entry["path"])
        chunk_path = args.labels / relative
        if sha256(chunk_path) != entry["sha256"]:
            raise RuntimeError(f"chunk SHA mismatch: {chunk_path}")
        with np.load(chunk_path, allow_pickle=False) as values:
            metadata = json.loads(values["metadata_json"].tobytes().decode())
            arrays = {name: values[name].copy() for name in ["source_line", "fen_hash64", *TEACHERS.values()]}

        source = Path(metadata["source"]["path"])
        if sha256(source) != metadata["source"]["sha256"]:
            raise RuntimeError(f"source SHA mismatch: {source}")
        start, stop = metadata["source_start_line"], metadata["source_stop_line"]
        expected_rows = stop - start
        if expected_rows != len(arrays["source_line"]):
            raise RuntimeError(f"metadata/array row mismatch: {chunk_path}")

        scores = np.column_stack([arrays[name] for name in TEACHERS.values()]).astype(np.int64)
        retained = np.all(np.abs(scores) < args.score_limit, axis=1)
        is_val = (arrays["fen_hash64"].astype(np.uint64) % np.uint64(1000)) < args.val_per_mille
        output_handles: dict[tuple[str, str], Any] = {}
        output_paths: dict[tuple[str, str], Path] = {}
        try:
            for teacher in TEACHERS:
                for split in ("train", "val"):
                    path = args.output / teacher / split / f"shard-{ordinal:06d}.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    output_paths[(teacher, split)] = path
                    output_handles[(teacher, split)] = path.open("x", encoding="utf-8")

            with source.open("r", encoding="utf-8") as handle:
                for _ in range(start):
                    if not handle.readline():
                        raise RuntimeError(f"source ended before line {start}: {source}")
                for row in range(expected_rows):
                    raw = handle.readline()
                    if not raw:
                        raise RuntimeError(f"source ended within chunk: {source}:{start + row + 1}")
                    fen, sf16_white, result_white = parse_source(raw)
                    if int(arrays["source_line"][row]) != start + row:
                        raise RuntimeError(f"source_line mismatch: {chunk_path}:{row}")
                    if int(arrays["fen_hash64"][row]) != fen_hash64(fen):
                        raise RuntimeError(f"FEN hash mismatch: {chunk_path}:{row}")
                    black = fen.split()[1] == "b"
                    sf16_stm = -sf16_white if black else sf16_white
                    if sf16_stm != int(arrays["sf16_score_stm"][row]):
                        raise RuntimeError(f"SF16 source score mismatch: {chunk_path}:{row}")
                    if not retained[row]:
                        continue
                    split = "val" if is_val[row] else "train"
                    result_text = str(result_white)
                    for teacher, array_name in TEACHERS.items():
                        score_stm = int(arrays[array_name][row])
                        score_white = -score_stm if black else score_stm
                        output_handles[(teacher, split)].write(f"{fen} | {score_white} | {result_text}\n")
        finally:
            for handle in output_handles.values():
                handle.close()

        input_rows = expected_rows
        retained_rows = int(retained.sum())
        val_rows = int(np.count_nonzero(retained & is_val))
        train_rows = retained_rows - val_rows
        totals["input"] += input_rows
        totals["retained"] += retained_rows
        totals["excluded_score"] += input_rows - retained_rows
        totals["train"] += train_rows
        totals["val"] += val_rows
        exported.append({
            "input": str(relative), "input_sha256": entry["sha256"],
            "rows": input_rows, "retained": retained_rows,
            "train": train_rows, "val": val_rows,
            "outputs": {
                f"{teacher}/{split}": {"path": str(path.relative_to(args.output)), "sha256": sha256(path)}
                for (teacher, split), path in output_paths.items()
            },
        })

    result = {
        "schema": SCHEMA,
        "source": str(args.labels.resolve()),
        "source_config_hash": manifest["config_hash"],
        "source_manifest_sha256": sha256(args.labels / "manifest.json"),
        "chunks": len(entries),
        "score_rule": f"retain iff abs(score) < {args.score_limit} for every teacher",
        "split_rule": f"val iff fen_hash64 % 1000 < {args.val_per_mille}",
        "teachers": TEACHERS,
        "totals": totals,
        "exported": exported,
    }
    atomic_json(args.output / "manifest.json", result)
    print(json.dumps({key: result[key] for key in ("schema", "chunks", "totals")}, indent=2))


if __name__ == "__main__":
    main()
