#!/usr/bin/env python3
"""Reconcile and optionally replay-audit an Archon native-game task corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from google.cloud import storage

from native_game_generator import SCHEMA, sha256, validate_games
from shard_lc0_positions import parse_gs, storage_client


TASK_NAME = re.compile(r"(?:^|/)tasks/(?P<task>[0-9]{6})\.jsonl\.gz$")
LOCAL_NAME = re.compile(r"(?P<task>[0-9]{6})\.jsonl\.gz$")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=path.name + ".",
        suffix=".tmp", dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def required_metadata(metadata: dict[str, str], task: int) -> None:
    required = {
        "schema", "sha256", "config_hash", "corpus_id", "stratum",
        "task_index", "positions", "games",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise RuntimeError(f"task {task} lacks metadata: {missing}")
    if metadata["schema"] != SCHEMA or int(metadata["task_index"]) != task:
        raise RuntimeError(f"task {task} has inconsistent schema/index metadata")
    if int(metadata["positions"]) <= 0 or int(metadata["games"]) <= 0:
        raise RuntimeError(f"task {task} has non-positive row/game counts")


def sample_indices(tasks: list[int], count: int) -> list[int]:
    if count <= 0 or not tasks:
        return []
    count = min(count, len(tasks))
    return sorted({
        tasks[round(i * (len(tasks) - 1) / max(1, count - 1))]
        for i in range(count)
    })


def reconcile_gcs(args: argparse.Namespace) -> dict[str, Any]:
    client = storage_client(args.project)
    bucket_name, prefix = parse_gs(args.input)
    task_prefix = prefix.rstrip("/") + "/tasks/"
    objects: dict[int, storage.Blob] = {}
    metadata_by_task: dict[int, dict[str, str]] = {}
    total_bytes = total_positions = total_games = 0
    for blob in client.list_blobs(bucket_name, prefix=task_prefix):
        match = TASK_NAME.search(blob.name)
        if not match:
            raise RuntimeError(f"unexpected native object: gs://{bucket_name}/{blob.name}")
        task = int(match.group("task"))
        if task in objects:
            raise RuntimeError(f"duplicate native task index {task}")
        blob.reload()
        metadata = blob.metadata or {}
        required_metadata(metadata, task)
        objects[task] = blob
        metadata_by_task[task] = metadata
        total_bytes += int(blob.size)
        total_positions += int(metadata["positions"])
        total_games += int(metadata["games"])

    sampled = sample_indices(sorted(objects), args.download_samples)
    with tempfile.TemporaryDirectory(prefix="archon-native-audit-") as temporary:
        root = Path(temporary)
        for task in sampled:
            path = root / f"{task:06d}.jsonl.gz"
            blob = objects[task]
            blob.download_to_filename(path, checksum="crc32c")
            metadata = metadata_by_task[task]
            if sha256(path) != metadata["sha256"]:
                raise RuntimeError(f"downloaded SHA-256 mismatch for task {task}")
            validate_games(path, int(metadata["positions"]), int(metadata["games"]))
    return report(args, metadata_by_task, total_bytes, total_positions, total_games, sampled)


def reconcile_local(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.input)
    if not root.is_dir():
        raise RuntimeError(f"local native input is not a directory: {root}")
    metadata_by_task: dict[int, dict[str, str]] = {}
    paths: dict[int, Path] = {}
    total_bytes = total_positions = total_games = 0
    for path in sorted(root.glob("*.jsonl.gz")):
        match = LOCAL_NAME.fullmatch(path.name)
        if not match:
            raise RuntimeError(f"unexpected local native object: {path}")
        task = int(match.group("task"))
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if not metadata_path.is_file():
            raise RuntimeError(f"missing local metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required_metadata(metadata, task)
        if sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"local SHA-256 mismatch for task {task}")
        paths[task] = path
        metadata_by_task[task] = metadata
        total_bytes += path.stat().st_size
        total_positions += int(metadata["positions"])
        total_games += int(metadata["games"])
    sampled = sample_indices(sorted(paths), args.download_samples)
    for task in sampled:
        metadata = metadata_by_task[task]
        validate_games(paths[task], int(metadata["positions"]), int(metadata["games"]))
    return report(args, metadata_by_task, total_bytes, total_positions, total_games, sampled)


def report(
    args: argparse.Namespace,
    metadata_by_task: dict[int, dict[str, str]],
    total_bytes: int,
    total_positions: int,
    total_games: int,
    sampled: list[int],
) -> dict[str, Any]:
    expected = set(range(args.task_index_start, args.task_index_stop))
    present = set(metadata_by_task)
    unexpected = sorted(present.difference(expected))
    if unexpected:
        raise RuntimeError(f"unexpected task indices: {unexpected[:20]}")
    missing = sorted(expected.difference(present))
    config_hashes = sorted({item["config_hash"] for item in metadata_by_task.values()})
    corpus_ids = sorted({item["corpus_id"] for item in metadata_by_task.values()})
    strata = sorted({item["stratum"] for item in metadata_by_task.values()})
    if args.expected_corpus_id and corpus_ids and corpus_ids != [args.expected_corpus_id]:
        raise RuntimeError(f"unexpected corpus IDs: {corpus_ids}")
    if args.expected_stratum and strata and strata != [args.expected_stratum]:
        raise RuntimeError(f"unexpected strata: {strata}")
    if args.expected_config_hash and config_hashes and config_hashes != [args.expected_config_hash]:
        raise RuntimeError(f"unexpected config hashes: {config_hashes}")
    expected_positions = len(present) * args.positions_per_task if args.positions_per_task else None
    if expected_positions is not None and total_positions != expected_positions:
        raise RuntimeError(
            f"present position count {total_positions} != expected {expected_positions}"
        )
    return {
        "schema": "archon.native-games-reconciliation.v1",
        "input": args.input,
        "task_index_start": args.task_index_start,
        "task_index_stop": args.task_index_stop,
        "expected_tasks": len(expected),
        "present_tasks": len(present),
        "missing_tasks": len(missing),
        "first_missing": missing[:20],
        "positions": total_positions,
        "games": total_games,
        "bytes": total_bytes,
        "config_hashes": config_hashes,
        "corpus_ids": corpus_ids,
        "strata": strata,
        "replay_validated_tasks": sampled,
        "complete": not missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--task-index-start", type=int, required=True)
    parser.add_argument("--task-index-stop", type=int, required=True)
    parser.add_argument("--positions-per-task", type=int)
    parser.add_argument("--expected-corpus-id")
    parser.add_argument("--expected-stratum")
    parser.add_argument("--expected-config-hash")
    parser.add_argument("--download-samples", type=int, default=0)
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.task_index_start < 0 or args.task_index_stop <= args.task_index_start:
        parser.error("invalid task index interval")
    if args.positions_per_task is not None and args.positions_per_task <= 0:
        parser.error("--positions-per-task must be positive")
    if args.download_samples < 0:
        parser.error("--download-samples cannot be negative")
    value = reconcile_gcs(args) if args.input.startswith("gs://") else reconcile_local(args)
    if args.report:
        atomic_json(args.report, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
