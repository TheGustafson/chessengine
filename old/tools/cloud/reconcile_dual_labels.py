#!/usr/bin/env python3
"""Reconcile immutable paired-label objects against a position manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from google.cloud import storage
import numpy as np

from shard_lc0_positions import parse_gs, storage_client


OUTPUT_SCHEMA = "archon.paired-value-labels.v1"
OUTPUT_NAME = re.compile(r"^(?P<prefix>.+)/shards/(?P<ordinal>[0-9]{6})\.npz$")
NO_MOVE = 0xFFFF


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(client: storage.Client, uri: str) -> dict:
    bucket_name, name = parse_gs(uri)
    return json.loads(client.bucket(bucket_name).blob(name).download_as_text())


def validate_download(path: Path, shard: dict, blob: storage.Blob) -> None:
    if sha256(path) != blob.metadata["sha256"]:
        raise RuntimeError(f"download hash mismatch for {blob.name}")
    with np.load(path, allow_pickle=False) as data:
        required = {"source_index", "metadata_json"}
        for teacher in ("reckless", "stockfish"):
            required.update({
                f"{teacher}_score_cp", f"{teacher}_mate_plies",
                f"{teacher}_best_move", f"{teacher}_depth",
                f"{teacher}_seldepth", f"{teacher}_nodes",
                f"{teacher}_bound", f"{teacher}_wdl",
            })
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"missing arrays in {blob.name}: {sorted(missing)}")
        rows = int(shard["rows"])
        for name in required.difference({"metadata_json"}):
            if len(data[name]) != rows:
                raise RuntimeError(f"row mismatch in {blob.name}:{name}")
        metadata = json.loads(data["metadata_json"].tobytes().decode())
        if metadata["input_sha256"] != shard["sha256"]:
            raise RuntimeError(f"embedded input hash mismatch in {blob.name}")
        if metadata["source_shard"] != shard:
            raise RuntimeError(f"embedded source shard mismatch in {blob.name}")
        if metadata["config_hash"] != blob.metadata["config_hash"]:
            raise RuntimeError(f"embedded config hash mismatch in {blob.name}")
        if int(data["source_index"][0]) != int(shard["source_start"]):
            raise RuntimeError(f"first source index mismatch in {blob.name}")
        if int(data["source_index"][-1]) != int(shard["source_stop"]) - 1:
            raise RuntimeError(f"last source index mismatch in {blob.name}")
        for teacher in ("reckless", "stockfish"):
            if np.any(data[f"{teacher}_nodes"] == 0):
                raise RuntimeError(f"zero searched nodes in {blob.name}:{teacher}")
            moves = data[f"{teacher}_best_move"]
            if np.any((moves != NO_MOVE) & (moves > 0x5FFF)):
                raise RuntimeError(f"invalid move encoding in {blob.name}:{teacher}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--expected-config-hash")
    parser.add_argument("--download-samples", type=int, default=0)
    args = parser.parse_args()
    if args.download_samples < 0:
        parser.error("--download-samples cannot be negative")

    client = storage_client(args.project)
    manifest = read_json(client, args.manifest)
    expected = {int(shard["ordinal"]): shard for shard in manifest["shards"]}
    bucket_name, prefix = parse_gs(args.output)
    object_prefix = prefix.rstrip("/") + "/shards/"
    blobs: dict[int, storage.Blob] = {}
    config_hashes: set[str] = set()
    rows = 0
    bytes_total = 0
    for blob in client.list_blobs(bucket_name, prefix=object_prefix):
        match = OUTPUT_NAME.match(blob.name)
        if not match or match.group("prefix") != prefix.rstrip("/"):
            raise RuntimeError(f"unexpected output object: gs://{bucket_name}/{blob.name}")
        ordinal = int(match.group("ordinal"))
        if ordinal not in expected or ordinal in blobs:
            raise RuntimeError(f"unexpected or duplicate ordinal {ordinal}")
        blob.reload()
        metadata = blob.metadata or {}
        shard = expected[ordinal]
        required_metadata = {
            "schema": OUTPUT_SCHEMA,
            "input_sha256": shard["sha256"],
            "rows": str(shard["rows"]),
        }
        for key, value in required_metadata.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"metadata {key} mismatch for ordinal {ordinal}")
        if not metadata.get("sha256") or not metadata.get("config_hash"):
            raise RuntimeError(f"missing hash metadata for ordinal {ordinal}")
        config_hashes.add(metadata["config_hash"])
        blobs[ordinal] = blob
        rows += int(metadata["rows"])
        bytes_total += int(blob.size)

    if len(config_hashes) > 1:
        raise RuntimeError(f"multiple teacher config hashes: {sorted(config_hashes)}")
    if args.expected_config_hash and config_hashes and config_hashes != {args.expected_config_hash}:
        raise RuntimeError(f"unexpected teacher config hash: {sorted(config_hashes)}")

    sampled: list[int] = []
    if args.download_samples and blobs:
        available = sorted(blobs)
        count = min(args.download_samples, len(available))
        sampled = sorted({available[round(i * (len(available) - 1) / max(1, count - 1))]
                          for i in range(count)})
        with tempfile.TemporaryDirectory(prefix="archon-label-audit-") as temporary:
            root = Path(temporary)
            for ordinal in sampled:
                path = root / f"{ordinal:06d}.npz"
                blobs[ordinal].download_to_filename(path, checksum="crc32c")
                validate_download(path, expected[ordinal], blobs[ordinal])

    missing = sorted(set(expected).difference(blobs))
    print(json.dumps({
        "schema": "archon.paired-label-reconciliation.v1",
        "manifest": args.manifest,
        "output": args.output,
        "expected_shards": len(expected),
        "present_shards": len(blobs),
        "missing_shards": len(missing),
        "first_missing": missing[:20],
        "present_rows": rows,
        "present_bytes": bytes_total,
        "config_hashes": sorted(config_hashes),
        "download_validated_ordinals": sampled,
        "complete": not missing,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
