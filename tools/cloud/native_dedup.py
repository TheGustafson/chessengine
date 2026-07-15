#!/usr/bin/env python3
"""Resumable hash-partitioned global deduplication for native-game samples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO

import numpy as np
from google.cloud import storage

from native_game_generator import SCHEMA as NATIVE_SCHEMA, canonical_hash, sha256
from shard_lc0_positions import parse_gs, storage_client


MAP_SCHEMA = "archon.native-dedup-map.v1"
REDUCE_SCHEMA = "archon.native-dedup-partition.v1"
FINAL_SCHEMA = "archon.native-dedup-manifest.v1"
RECORD = struct.Struct(">16sIII")
DTYPE = np.dtype([
    ("key", "V16"), ("task_index", ">u4"),
    ("game_index", ">u4"), ("ply", ">u4"),
])


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


def deterministic_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as raw_input, destination.open("wb") as raw_output:
        with gzip.GzipFile(
            fileobj=raw_output, mode="wb", filename="", mtime=0, compresslevel=6,
        ) as compressed:
            for block in iter(lambda: raw_input.read(8 * 1024 * 1024), b""):
                compressed.write(block)


class Objects:
    def __init__(self, root: str, project: str) -> None:
        self.root = root.rstrip("/")
        self.gcs = root.startswith("gs://")
        if self.gcs:
            self.client = storage_client(project)
            self.bucket_name, self.prefix = parse_gs(self.root)
            self.bucket = self.client.bucket(self.bucket_name)
        else:
            self.path = Path(root)

    def _name(self, relative: str) -> str:
        return f"{self.prefix.rstrip('/')}/{relative}" if self.gcs else relative

    def metadata(self, relative: str) -> tuple[dict[str, str], int] | None:
        if self.gcs:
            blob = self.bucket.blob(self._name(relative))
            if not blob.exists():
                return None
            blob.reload()
            return blob.metadata or {}, int(blob.size)
        path = self.path / relative
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            return None
        return json.loads(metadata_path.read_text(encoding="utf-8")), path.stat().st_size

    def download(self, relative: str, destination: Path, expected_sha: str | None = None) -> None:
        if self.gcs:
            self.bucket.blob(self._name(relative)).download_to_filename(
                destination, checksum="crc32c"
            )
        else:
            source = self.path / relative
            with source.open("rb") as reader, destination.open("wb") as writer:
                for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    writer.write(block)
        if expected_sha and sha256(destination) != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch while reading {self.root}/{relative}")

    def upload(self, relative: str, source: Path, metadata: dict[str, str]) -> None:
        if self.gcs:
            blob = self.bucket.blob(self._name(relative))
            if blob.exists():
                blob.reload()
                if blob.metadata != metadata or int(blob.size) != source.stat().st_size:
                    raise RuntimeError(f"existing object differs: {self.root}/{relative}")
                return
            blob.metadata = metadata
            blob.upload_from_filename(source, if_generation_match=0, checksum="crc32c")
            return
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if path.exists():
            if sha256(path) != metadata["sha256"] or not metadata_path.is_file():
                raise RuntimeError(f"existing local object differs: {path}")
            if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                raise RuntimeError(f"existing local metadata differs: {metadata_path}")
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        with source.open("rb") as reader, temporary.open("wb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                writer.write(block)
        os.replace(temporary, path)
        atomic_json(metadata_path, metadata)

    def write_json(self, relative: str, value: Any) -> None:
        with tempfile.TemporaryDirectory(prefix="archon-dedup-json-") as temporary:
            path = Path(temporary) / "value.json"
            atomic_json(path, value)
            digest = sha256(path)
            self.upload(relative, path, {
                "schema": str(value["schema"]), "sha256": digest,
            })

    def read_json(self, relative: str) -> dict[str, Any]:
        detail = self.metadata(relative)
        if detail is None:
            raise RuntimeError(f"missing object: {self.root}/{relative}")
        metadata, _ = detail
        with tempfile.TemporaryDirectory(prefix="archon-dedup-json-") as temporary:
            path = Path(temporary) / "value.json"
            self.download(relative, path, metadata.get("sha256"))
            return json.loads(path.read_text(encoding="utf-8"))


def validate_partition_count(partitions: int) -> None:
    if partitions <= 0 or partitions > 4096 or partitions & (partitions - 1):
        raise ValueError("partition count must be a power of two in [1, 4096]")


def partition_for(key: bytes, partitions: int) -> int:
    return int.from_bytes(key[:8], "big") & (partitions - 1)


def native_relative(task: int, local_flat: bool) -> str:
    name = f"{task:06d}.jsonl.gz"
    return name if local_flat else f"tasks/{name}"


def mapper(args: argparse.Namespace) -> dict[str, Any]:
    validate_partition_count(args.partitions)
    source = Objects(args.input, args.project)
    destination = Objects(args.output, args.project)
    manifest_name = f"maps/{args.map_index:04d}/manifest.json"
    existing = destination.metadata(manifest_name)
    if existing is not None:
        manifest = destination.read_json(manifest_name)
        expected = {
            "map_index": args.map_index, "task_index_start": args.task_index_start,
            "task_index_stop": args.task_index_stop, "partitions": args.partitions,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError("existing mapper manifest differs from requested work")
        for fragment in manifest.get("fragments", []):
            part = int(fragment["partition"])
            relative = f"maps/{args.map_index:04d}/part-{part:04d}.bin.gz"
            detail = destination.metadata(relative)
            if detail is None:
                raise RuntimeError(f"mapper manifest references missing fragment: {relative}")
            metadata, _ = detail
            if (
                metadata.get("sha256") != fragment["sha256"]
                or int(metadata.get("records", -1)) != int(fragment["records"])
            ):
                raise RuntimeError(f"mapper fragment differs from manifest: {relative}")
        print(json.dumps({"status": "validated_existing", **manifest}, sort_keys=True))
        return manifest

    with tempfile.TemporaryDirectory(prefix="archon-native-dedup-map-") as temporary:
        root = Path(temporary)
        raw_paths = [root / f"part-{part:04d}.bin" for part in range(args.partitions)]
        handles: list[BinaryIO] = [path.open("wb") for path in raw_paths]
        counts = [0] * args.partitions
        total = games = 0
        config_hashes: set[str] = set()
        corpus_ids: set[str] = set()
        strata: set[str] = set()
        input_hashes: list[str] = []
        try:
            for task in range(args.task_index_start, args.task_index_stop):
                relative = native_relative(task, not source.gcs and args.local_flat_input)
                detail = source.metadata(relative)
                if detail is None:
                    raise RuntimeError(f"missing native task {task}: {source.root}/{relative}")
                metadata, _ = detail
                required = {
                    "schema", "sha256", "config_hash", "corpus_id", "stratum",
                    "task_index", "positions", "games",
                }
                if required.difference(metadata) or metadata["schema"] != NATIVE_SCHEMA:
                    raise RuntimeError(f"invalid native metadata for task {task}")
                if int(metadata["task_index"]) != task:
                    raise RuntimeError(f"native task-index mismatch for task {task}")
                path = root / f"task-{task:06d}.jsonl.gz"
                source.download(relative, path, metadata["sha256"])
                task_positions = task_games = 0
                with gzip.open(path, "rt", encoding="utf-8") as reader:
                    for line in reader:
                        game = json.loads(line)
                        if game.get("schema") != NATIVE_SCHEMA or int(game["task_index"]) != task:
                            raise RuntimeError(f"invalid native game in task {task}")
                        game_index = int(game["game_index"])
                        for sample in game["samples"]:
                            key = bytes.fromhex(sample["state_key"])
                            if len(key) != 16:
                                raise RuntimeError(f"invalid state key in task {task}")
                            part = partition_for(key, args.partitions)
                            handles[part].write(RECORD.pack(
                                key, task, game_index, int(sample["ply"])
                            ))
                            counts[part] += 1
                            task_positions += 1
                        task_games += 1
                path.unlink()
                if (task_positions, task_games) != (
                    int(metadata["positions"]), int(metadata["games"]),
                ):
                    raise RuntimeError(f"native count mismatch for task {task}")
                total += task_positions
                games += task_games
                config_hashes.add(metadata["config_hash"])
                corpus_ids.add(metadata["corpus_id"])
                strata.add(metadata["stratum"])
                input_hashes.append(metadata["sha256"])
        finally:
            for handle in handles:
                handle.close()

        fragments = []
        source_set_hash = canonical_hash(input_hashes)
        for part, raw_path in enumerate(raw_paths):
            compressed = root / f"part-{part:04d}.bin.gz"
            deterministic_gzip(raw_path, compressed)
            raw_path.unlink()
            digest = sha256(compressed)
            relative = f"maps/{args.map_index:04d}/part-{part:04d}.bin.gz"
            metadata = {
                "schema": MAP_SCHEMA, "sha256": digest,
                "record_size": str(RECORD.size), "records": str(counts[part]),
                "partitions": str(args.partitions), "partition": str(part),
                "map_index": str(args.map_index),
                "task_index_start": str(args.task_index_start),
                "task_index_stop": str(args.task_index_stop),
                "source_set_hash": source_set_hash,
            }
            destination.upload(relative, compressed, metadata)
            fragments.append({"partition": part, "records": counts[part], "sha256": digest})

    manifest = {
        "schema": MAP_SCHEMA,
        "map_index": args.map_index,
        "task_index_start": args.task_index_start,
        "task_index_stop": args.task_index_stop,
        "partitions": args.partitions,
        "record_size": RECORD.size,
        "records": total,
        "games": games,
        "config_hashes": sorted(config_hashes),
        "corpus_ids": sorted(corpus_ids),
        "strata": sorted(strata),
        "source_set_hash": source_set_hash,
        "fragments": fragments,
    }
    destination.write_json(manifest_name, manifest)
    print(json.dumps({"status": "written", **manifest}, sort_keys=True))
    return manifest


def reducer(args: argparse.Namespace) -> dict[str, Any]:
    validate_partition_count(args.partitions)
    if not 0 <= args.partition_index < args.partitions:
        raise ValueError("partition index outside partition range")
    source = Objects(args.input, args.project)
    destination = Objects(args.output, args.project)
    output_relative = f"partitions/part-{args.partition_index:04d}.npz"
    existing = destination.metadata(output_relative)
    if existing is not None:
        metadata, _ = existing
        if (
            metadata.get("schema") != REDUCE_SCHEMA
            or int(metadata.get("partition", -1)) != args.partition_index
            or int(metadata.get("maps", -1)) != args.map_count
        ):
            raise RuntimeError("existing reduced partition differs from requested work")
        print(json.dumps({"status": "validated_existing", **metadata}, sort_keys=True))
        return metadata

    with tempfile.TemporaryDirectory(prefix="archon-native-dedup-reduce-") as temporary:
        root = Path(temporary)
        combined = root / "combined.bin"
        input_records = 0
        with combined.open("wb") as writer:
            for map_index in range(args.map_count):
                relative = f"maps/{map_index:04d}/part-{args.partition_index:04d}.bin.gz"
                detail = source.metadata(relative)
                if detail is None:
                    raise RuntimeError(f"missing map fragment {map_index}/{args.partition_index}")
                metadata, _ = detail
                required = {
                    "schema": MAP_SCHEMA, "record_size": str(RECORD.size),
                    "partitions": str(args.partitions),
                    "partition": str(args.partition_index), "map_index": str(map_index),
                }
                if any(metadata.get(key) != value for key, value in required.items()):
                    raise RuntimeError(f"map-fragment metadata mismatch: {relative}")
                compressed = root / f"map-{map_index:04d}.bin.gz"
                source.download(relative, compressed, metadata["sha256"])
                with gzip.open(compressed, "rb") as reader:
                    for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        writer.write(block)
                compressed.unlink()
                input_records += int(metadata["records"])
        if combined.stat().st_size != input_records * RECORD.size:
            raise RuntimeError("combined map-fragment byte count mismatch")
        records = np.fromfile(combined, dtype=DTYPE)
        records.sort(order=["key", "task_index", "game_index", "ply"], kind="stable")
        keep = np.ones(len(records), dtype=np.bool_)
        if len(records) > 1:
            keep[1:] = records["key"][1:] != records["key"][:-1]
        unique = records[keep]
        output_path = root / f"part-{args.partition_index:04d}.npz"
        # A field projected from a structured array is strided by the full
        # record width; make it contiguous before reinterpreting 16-byte keys.
        keys = np.ascontiguousarray(unique["key"]).view(np.uint8).reshape(-1, 16)
        np.savez_compressed(
            output_path,
            state_key=keys,
            task_index=unique["task_index"].astype(np.uint32),
            game_index=unique["game_index"].astype(np.uint32),
            ply=unique["ply"].astype(np.uint32),
        )
        digest = sha256(output_path)
        metadata = {
            "schema": REDUCE_SCHEMA, "sha256": digest,
            "partition": str(args.partition_index), "partitions": str(args.partitions),
            "maps": str(args.map_count), "input_records": str(input_records),
            "unique_records": str(len(unique)),
            "duplicate_records": str(input_records - len(unique)),
        }
        destination.upload(output_relative, output_path, metadata)
    print(json.dumps({"status": "written", **metadata}, sort_keys=True))
    return metadata


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    validate_partition_count(args.partitions)
    source = Objects(args.input, args.project)
    maps = Objects(args.maps_input, args.project)
    map_manifests = []
    mapped_records = 0
    intervals = []
    for map_index in range(args.map_count):
        manifest = maps.read_json(f"maps/{map_index:04d}/manifest.json")
        if (
            manifest.get("schema") != MAP_SCHEMA
            or int(manifest.get("map_index", -1)) != map_index
            or int(manifest.get("partitions", -1)) != args.partitions
        ):
            raise RuntimeError(f"mapper manifest mismatch for map {map_index}")
        if sum(int(item["records"]) for item in manifest["fragments"]) != int(manifest["records"]):
            raise RuntimeError(f"mapper fragment counts do not reconcile for map {map_index}")
        intervals.append((
            int(manifest["task_index_start"]), int(manifest["task_index_stop"]), map_index,
        ))
        mapped_records += int(manifest["records"])
        map_manifests.append({
            "map_index": map_index,
            "task_index_start": int(manifest["task_index_start"]),
            "task_index_stop": int(manifest["task_index_stop"]),
            "records": int(manifest["records"]),
            "source_set_hash": manifest["source_set_hash"],
        })
    cursor = args.task_index_start
    for start, stop, map_index in sorted(intervals):
        if start != cursor or stop <= start:
            raise RuntimeError(
                f"mapper task coverage gap/overlap at map {map_index}: expected {cursor}, got {start}:{stop}"
            )
        cursor = stop
    if cursor != args.task_index_stop:
        raise RuntimeError(
            f"mapper task coverage ends at {cursor}, expected {args.task_index_stop}"
        )
    partitions = []
    input_records = unique_records = duplicate_records = total_bytes = 0
    for part in range(args.partitions):
        relative = f"partitions/part-{part:04d}.npz"
        detail = source.metadata(relative)
        if detail is None:
            raise RuntimeError(f"missing reduced partition {part}")
        metadata, size = detail
        required = {
            "schema": REDUCE_SCHEMA, "partition": str(part),
            "partitions": str(args.partitions), "maps": str(args.map_count),
        }
        if any(metadata.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"reduced-partition metadata mismatch: {relative}")
        item = {
            "partition": part, "sha256": metadata["sha256"], "bytes": size,
            "input_records": int(metadata["input_records"]),
            "unique_records": int(metadata["unique_records"]),
            "duplicate_records": int(metadata["duplicate_records"]),
        }
        partitions.append(item)
        input_records += item["input_records"]
        unique_records += item["unique_records"]
        duplicate_records += item["duplicate_records"]
        total_bytes += size
    if input_records != unique_records + duplicate_records:
        raise RuntimeError("global dedup count reconciliation failed")
    if input_records != mapped_records:
        raise RuntimeError(
            f"mapper/reducer record count mismatch: {mapped_records} != {input_records}"
        )
    manifest = {
        "schema": FINAL_SCHEMA,
        "input": args.input,
        "partitions": args.partitions,
        "maps": args.map_count,
        "maps_input": args.maps_input,
        "task_index_start": args.task_index_start,
        "task_index_stop": args.task_index_stop,
        "map_manifests": map_manifests,
        "input_records": input_records,
        "unique_records": unique_records,
        "duplicate_records": duplicate_records,
        "duplicate_fraction": duplicate_records / input_records if input_records else 0.0,
        "bytes": total_bytes,
        "partition_objects": partitions,
        "complete": True,
    }
    source.write_json("manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    map_parser = sub.add_parser("map")
    map_parser.add_argument("--input", required=True)
    map_parser.add_argument("--output", required=True)
    map_parser.add_argument("--map-index", type=int, required=True)
    map_parser.add_argument("--task-index-start", type=int, required=True)
    map_parser.add_argument("--task-index-stop", type=int, required=True)
    map_parser.add_argument("--partitions", type=int, default=64)
    map_parser.add_argument("--local-flat-input", action="store_true")
    reduce_parser = sub.add_parser("reduce")
    reduce_parser.add_argument("--input", required=True)
    reduce_parser.add_argument("--output", required=True)
    reduce_parser.add_argument("--partition-index", type=int, required=True)
    reduce_parser.add_argument("--partitions", type=int, default=64)
    reduce_parser.add_argument("--map-count", type=int, required=True)
    final_parser = sub.add_parser("finalize")
    final_parser.add_argument("--input", required=True)
    final_parser.add_argument("--maps-input", required=True)
    final_parser.add_argument("--partitions", type=int, default=64)
    final_parser.add_argument("--map-count", type=int, required=True)
    final_parser.add_argument("--task-index-start", type=int, required=True)
    final_parser.add_argument("--task-index-stop", type=int, required=True)
    for child in (map_parser, reduce_parser, final_parser):
        child.add_argument("--project", default="archon-502500")
    args = parser.parse_args()
    if getattr(args, "map_index", 0) < 0 or getattr(args, "map_count", 1) <= 0:
        parser.error("map indices/counts must be non-negative/positive")
    if args.command == "map":
        if args.task_index_start < 0 or args.task_index_stop <= args.task_index_start:
            parser.error("invalid task interval")
        mapper(args)
    elif args.command == "reduce":
        reducer(args)
    else:
        if args.task_index_start < 0 or args.task_index_stop <= args.task_index_start:
            parser.error("invalid task interval")
        finalize(args)


if __name__ == "__main__":
    main()
